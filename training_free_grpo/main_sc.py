import os
import json
import argparse
import asyncio
import copy
import time
import traceback
import numpy as np
from functools import partial

from tqdm import tqdm
from collections import defaultdict, Counter
from typing import Callable

from utu.agents import SimpleAgent
from utu.config import ConfigLoader
from utu.utils import AgentsUtils
from utu.agents.common import TaskRecorder
from training_free_grpo.llm import LLM
from openai import OpenAI

from training_free_grpo.medical.dataset import load_data
from training_free_grpo.medical.verify import (
    verify_func,
    extract_answer,
    classify_answer,
    eval_for_multiple_choice,
)
from training_free_grpo.medical.prompts import PROBLEM_WITH_EXPERIENCE_TEMPLATE


def calculate_and_update_self_consistency(
    rollouts: list[dict], experiment_name: str, rollout_filename: str
) -> tuple[list[dict], dict]:
    """
    Calculates self-consistency, updates each rollout with SC info, and returns SC stats.
    """
    domain = rollout_filename.split("/")[1]
    problem_to_rollouts = defaultdict(list)
    for r in rollouts:
        problem_to_rollouts[r["problem"]].append(r)

    sc_rewards = []
    sc_answers_log = {}  # For saving to a separate file

    # First pass: classify all answers and add to each rollout
    for problem, p_rollouts in problem_to_rollouts.items():
        for r in p_rollouts:
            extracted_answer = extract_answer(r["response"])
            classified_answer = classify_answer(r["problem"], extracted_answer)
            r["classified_answer"] = classified_answer

    # Second pass: calculate SC answer and reward, and update all related rollouts
    for problem, p_rollouts in problem_to_rollouts.items():
        answers = [r["classified_answer"] for r in p_rollouts if r["classified_answer"]]

        most_common_answer = ""
        sc_reward = 0.0

        if answers:
            most_common_answer_tuple = Counter(answers).most_common(1)
            if most_common_answer_tuple:
                most_common_answer = most_common_answer_tuple[0][0]
                ground_truth = p_rollouts[0]["groundtruth"]
                reward = eval_for_multiple_choice(
                    problem, most_common_answer, ground_truth
                )
                sc_reward = float(reward)
                sc_rewards.append(sc_reward)

        # Log for the separate file
        sc_answers_log[problem] = {
            "self_consistent_answer": most_common_answer,
            "reward": sc_reward,
            "all_classified_answers": answers,
        }

        # Update each rollout for this problem with the final SC info
        for r in p_rollouts:
            r["self_consistent_answer"] = most_common_answer
            r["self_consistent_reward"] = sc_reward

    # Save the separate log file
    # sc_filename = f"data/{domain}/eval/sc_answers/{experiment_name}_sc_answers.json"
    # with open(sc_filename, "w", encoding="utf-8") as f:
    #     json.dump(sc_answers_log, f, indent=2, ensure_ascii=False)
    # print(f"Saved self-consistency answers to {sc_filename}")

    sc_stats = {}
    if sc_rewards:
        sc_stats["self_consistency_reward"] = np.mean(sc_rewards)

    return rollouts, sc_stats


def bootstrap_metric(
    data: list[any],
    subset_size: int,
    reduce_fns: list[Callable[[np.ndarray], float]],
    n_bootstrap: int = 1000,
    seed: int = 42,
) -> list[tuple[float, float]]:
    """Performs bootstrap resampling to estimate statistics of metrics."""
    np.random.seed(seed)

    bootstrap_metric_lsts = [[] for _ in range(len(reduce_fns))]
    for _ in range(n_bootstrap):
        bootstrap_idxs = np.random.choice(len(data), size=subset_size, replace=True)
        bootstrap_data = [data[i] for i in bootstrap_idxs]
        for i, reduce_fn in enumerate(reduce_fns):
            bootstrap_metric_lsts[i].append(reduce_fn(bootstrap_data))
    return [(np.mean(lst), np.std(lst)) for lst in bootstrap_metric_lsts]


def _calc_pass_at_k(scores: list[float], k: int) -> float:
    """Estimates pass@k given a list of scores."""
    m = len(scores)
    c = sum(1 for s in scores if s > 0)

    if m < k:
        # If number of samples is less than k, we can't form a k-subset.
        # This case might indicate an issue, but we return 0 as a safeguard.
        return 0.0

    if m - c < k:
        return 1.0

    # calculate 1 - C(m-c, k) / C(m, k)
    return 1.0 - np.prod(1.0 - k / np.arange(m - c + 1, m + 1))


def load_rollouts(rollout_filename: str) -> list[dict]:
    results = []
    if os.path.exists(rollout_filename):
        with open(rollout_filename, encoding="utf-8") as f:
            for line in f:
                results.append(json.loads(line))
    return results


def save_rollouts(results: list[dict], rollout_filename: str):
    with open(rollout_filename, "w", encoding="utf-8") as f:
        for result in results:
            f.write(json.dumps(result, ensure_ascii=False) + "\n")


async def rollout_dataset(
    worker_agent: SimpleAgent | None,
    data: list[dict],
    rollouts: list[dict],
    rollout_filename: str,
    verify_func: callable,
    pass_k: int,
    experiment_name: str,
    rollout_concurrency: int = 5,
    task_timeout: float = 3600,
    max_retries: int = 3,
    temperature: float = 0.0,
    max_tokens: int = 2048,
) -> list[dict]:
    """Rollout the dataset using the worker agent with concurrency control, timeout, error handling, and retries."""

    # base_url = "https://api.nuwaapi.com/v1"
    # api_key = "sk-Jp2KWjbRTJniXopDRSkUjTWGyCRUDF15aSvhDbRnrdACMS9C"
    # concurrency_limit = 128
    # client = OpenAI(api_key=api_key, base_url=base_url)
    # examine data and existing rollouts
    if len(rollouts) > 0:
        for each in rollouts:
            assert "runid" in each
        data_problems = [each["problem"] for each in data]
        rollouts_problems = [each["problem"] for each in rollouts]
        assert data_problems == rollouts_problems, (
            f"The problems in data should be the same as existing rollouts {rollout_filename}"
        )
    else:
        for sample in data:
            assert "problem" in sample and "groundtruth" in sample
        rollouts = [{"runid": i, **sample} for i, sample in enumerate(data)]
    save_rollouts(rollouts, rollout_filename)

    # create task queue
    task_queue = asyncio.Queue()
    pending_tasks_count = 0
    for sample in rollouts:
        if "trajectories" not in sample or len(sample["trajectories"]) == 0:
            sample_with_retry = copy.deepcopy(sample)
            sample_with_retry["retry_count"] = 0
            await task_queue.put(sample_with_retry)
            pending_tasks_count += 1
    pbar = tqdm(total=pending_tasks_count, desc="Rolling out")

    async def worker(name: str):
        while not task_queue.empty():
            sample = await task_queue.get()
            task_start_time = time.time()
            try:
                if worker_agent is None:
                    llm = LLM()
                    system_prompt='''Solve the following problem step by step. 
    The last part of your final response should be in the following format:
    <answer>
    'The final answer goes here.'
    </answer>'''
                    system_prompt2='''Please answer the following question.

-----

## FINAL ANSWER FORMAT

ALWAYS present your final answer in the following format:

FINAL ANSWER:
<answer>
(final answer)
</answer>

N.B. Make sure that the final answer is properly wrapped inside the <answer> block.

* For multiple-choice questions: Only provide the letter choice (e.g., (A))
* For numerical answers: Only provide the final number (e.g., 42)
* For other types of answers, including free-response answers: Provide the complete final answer

Example:
Q: What is the meaning of life?
A: [...]
FINAL ANSWER:
<answer>
42
</answer>'''        
                    messages_or_prompt=[{"role": "system", "content": system_prompt2}, {"role": "user", "content": sample["prompt"]}]
                    
                    coro = asyncio.to_thread(llm.chat, messages_or_prompt, temperature=0.0, max_tokens=2048)
                    
                    res = await asyncio.wait_for(coro, timeout=task_timeout)     
                     
                    #res = llm.chat(messages_or_prompt,temperature=0,max_tokens=2048)               
                    res = TaskRecorder(
                            final_output=res,
                            trajectories=[{
                                "trajectory": [
                                    {"role": "system", "content": system_prompt2},
                                    {"role": "user", "content": sample["prompt"]},
                                    {"role": "assistant", "content": res}
                                ]
                            }],
                        )
                    print('sample["prompt"]',sample["prompt"])
                else:
                    async with worker_agent as agent:
                        async def rollout_streamed(sample) -> TaskRecorder:
                            prompt = sample.get("prompt", sample["problem"])
                            res = agent.run_streamed(prompt)
                            async for _ in res.stream_events(): pass
                            traj = AgentsUtils.get_trajectory_from_agent_result(res)
                            return TaskRecorder(
                                final_output=res.final_output,
                                trajectories=[traj],
                            )
                        res = await asyncio.wait_for(rollout_streamed(sample), timeout=task_timeout)
                
                task_end_time = time.time()
                sample.update(
                    {
                        "response": res.final_output,
                        "trajectories": res.trajectories,
                        "error": None,
                        "rollout_time": task_end_time - task_start_time,
                    }
                )
                sample["reward"] = verify_func(sample, sample["groundtruth"])
                
                # Task succeeded
                rollouts[sample["runid"]] = sample
                save_rollouts(rollouts, rollout_filename)
                pbar.update(1)

            except Exception as e:
                task_end_time = time.time()
                sample["retry_count"] += 1
                error_info = traceback.format_exc()
                print(f"> error: {error_info}")
                
                if sample["retry_count"] <= max_retries:
                    tqdm.write(f"Worker {name}: Task runid={sample['runid']} failed with {type(e).__name__}. Retrying ({sample['retry_count']}/{max_retries})...")
                    await task_queue.put(sample) # Re-queue the task
                else:
                    tqdm.write(f"Worker {name}: Task runid={sample['runid']} failed after {max_retries} retries. Error: {e}. Traceback: {error_info}")
                    sample.update(
                        {
                            "response": f"Error: {str(e)} after {max_retries} retries.",
                            "trajectories": [],
                            "error": error_info,
                            "reward": 0,
                            "rollout_time": task_end_time - task_start_time,
                        }
                    )
                    
                    # Task failed permanently
                    rollouts[sample["runid"]] = sample
                    save_rollouts(rollouts, rollout_filename)
                    pbar.update(1)
            finally:
                task_queue.task_done()

    # run all tasks
    workers = [asyncio.create_task(worker(f"worker-{i}")) for i in range(rollout_concurrency)]
    await task_queue.join()

    # clean up
    for w in workers:
        w.cancel()
    await asyncio.gather(*workers, return_exceptions=True)
    pbar.close()
    print(f"Successfully processed {len(rollouts)} samples.")

    # Calculate self-consistency and update rollouts
    sc_stats = {}
    if pass_k > 1:
        rollouts, sc_stats = calculate_and_update_self_consistency(
            rollouts, experiment_name, rollout_filename
        )
        save_rollouts(rollouts, rollout_filename)

    # stats
    all_rewards = []
    problem_to_scores = defaultdict(list)
    num_tool_calls = []
    for rollout in rollouts:
        all_rewards.append(rollout.get("reward", 0))
        problem_to_scores[rollout["problem"]].append(rollout.get("reward", 0))
        if "trajectories" in rollout and rollout["trajectories"]:
            num_tool_calls.append(
                len([each for each in rollout["trajectories"][0]["trajectory"] if each["role"] == "tool"])
            )

    problem_metrics = defaultdict(dict)
    for problem, scores in problem_to_scores.items():
        if not scores:
            continue

        metric = {}
        n_resps = len(scores)
        metric[f"mean@{n_resps}"] = np.mean(scores)

        if n_resps > 1:
            metric[f"std@{n_resps}"] = np.std(scores)

            ns = []
            n = 1
            while n < n_resps:
                ns.append(n)
                n *= 2
            if n_resps not in ns:
                ns.append(n_resps)
            
            for n in ns:
                if n > n_resps: continue

                # bootstrap for best@n and worst@n
                [(bon_mean, bon_std), (won_mean, won_std)] = bootstrap_metric(
                    data=scores, subset_size=n, reduce_fns=[np.max, np.min], seed=42
                )
                metric[f"best@{n}/mean"] = bon_mean
                metric[f"best@{n}/std"] = bon_std
                metric[f"worst@{n}/mean"] = won_mean
                metric[f"worst@{n}/std"] = won_std

                # pass@n
                metric[f"pass@{n}"] = _calc_pass_at_k(scores, n)

        problem_metrics[problem] = metric
    
    # Aggregate metrics
    agg_metrics = defaultdict(list)
    for problem, p_metrics in problem_metrics.items():
        for metric_name, metric_val in p_metrics.items():
            agg_metrics[metric_name].append(metric_val)

    stats = {name: np.mean(vals) for name, vals in agg_metrics.items()}
    
    # Calculate and add old pass rate
    if problem_to_scores:
        problem_to_max_score = {problem: max(scores) for problem, scores in problem_to_scores.items() if scores}
        max_K = max((len(scores) for scores in problem_to_scores.values()), default=0)
        if problem_to_max_score:
            stats[f"pass_old@{max_K}"] = sum(max_reward > 0 for max_reward in problem_to_max_score.values()) / len(problem_to_max_score)
        else:
            stats[f"pass_old@{max_K}"] = 0.0
            
    stats["avg_reward"] = sum(all_rewards) / len(all_rewards) if all_rewards else 0
    stats["avg_tool_call"] = sum(num_tool_calls) / len(num_tool_calls) if num_tool_calls else 0
    
    stats.update(sc_stats)

    for k, v in stats.items():
        print(f"- {k}: {v}")
    return rollouts, stats


async def main(args):
    # Set up domain-specific variables
    if args.domain == "medical":
        config_name = "simple/base_medical.yaml"
    else:
        raise ValueError(f"Unsupported domain: {args.domain}")

    # Set up the agent
    if args.mode == "prompt":
        worker_agent = None
    elif args.mode == "agent":
        config = ConfigLoader.load_agent_config(config_name)
        config.model.model_settings.temperature = args.rollout_temperature
        worker_agent = SimpleAgent(config=config)
        await worker_agent.build()
    else:
        raise ValueError(f"Unsupported inference mode: {args.mode}")

    # Load the dataset
    test_data = load_data(args.dataset)
    print(f"Loaded {len(test_data)} records from dataset")
    if args.dataset_truncate is not None:
        print(f"- truncated to {args.dataset_truncate}")
        test_data = test_data[: args.dataset_truncate]
    
    # Insert experiences
    if args.experience_file:
        experiences = json.load(open(args.experience_file))
        formatted_experiences = "\n".join([ f"[{i}]. {e}" for i, e in experiences.items() ])
        formatted_test_data = [{
            "prompt": PROBLEM_WITH_EXPERIENCE_TEMPLATE.format(
                experiences=formatted_experiences if formatted_experiences else "None",
                problem=each["problem"],
            ),
            **each
        } for each in test_data]
    else:
        formatted_test_data = [{
            "prompt": each["problem"],
            **each
        } for each in test_data]
    
    # Duplicate for Pass@k evaluation
    formatted_test_data = formatted_test_data * args.pass_k
    print(f"Duplicated to {len(formatted_test_data)} records for Pass@{args.pass_k} evaluation")

    # Load existing rollouts
    os.makedirs(f"data/{args.domain}/eval", exist_ok=True)
    rollout_filename = f"data/{args.domain}/eval/{args.experiment_name}.jsonl"
    rollouts = load_rollouts(rollout_filename)

    # Rollout the dataset
    rollouts, stats = await rollout_dataset(
        worker_agent=worker_agent,
        data=formatted_test_data,
        rollouts=rollouts,
        verify_func=verify_func,
        rollout_filename=rollout_filename,
        pass_k=args.pass_k,
        experiment_name=args.experiment_name,
        rollout_concurrency=args.rollout_concurrency,
        task_timeout=args.task_timeout,
        max_tokens=args.rollout_max_tokens,
    )

    #保存stats
    os.makedirs(f"data/{args.domain}/eval", exist_ok=True)
    stats_filename = f"data/{args.domain}/eval/{args.experiment_name}_stats.json"
    json.dump(stats, open(stats_filename, "w"), indent=2)
    print(f"Saved stats to {stats_filename}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Training-Free GRPO Evaluation")
    parser.add_argument("--mode", type=str, default="agent", required=True, choices=["prompt", "agent"], help="Mode of inference")
    parser.add_argument("--domain", type=str, required=True, choices=["medical"], help="The domain of the experiment")
    parser.add_argument("--experiment_name", type=str, required=True, help="Name of the experiment run")
    parser.add_argument("--dataset", type=str, required=True, help="Name of dataset")
    parser.add_argument("--dataset_truncate", type=int, default=None, help="Truncate dataset to first N samples")
    parser.add_argument("--experience_file", type=str, default=None)
    parser.add_argument("--rollout_concurrency", type=int, default=5, help="Concurrency level for rollouts")
    parser.add_argument("--rollout_temperature", type=float, default=0.0, help="Temperature for the LLM")
    parser.add_argument("--rollout_max_tokens", type=int, default=8192, help="Max tokens for each rollout")
    parser.add_argument("--pass_k", type=int, default=1, help="Pass@k metric")
    parser.add_argument("--task_timeout", type=float, default=3600, help="Timeout for each individual task in seconds")

    args = parser.parse_args()
    asyncio.run(main(args))