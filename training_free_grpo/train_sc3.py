import argparse
import asyncio
import copy
import json
import os
import random
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm

from training_free_grpo.main_sc2 import rollout_dataset, load_rollouts
from training_free_grpo.llm_sc import LLM
from utu.agents import SimpleAgent
from utu.config import ConfigLoader

random.seed(42)


async def main(args):
    # Set up domain-specific variables
    if args.domain == "medical":
        from training_free_grpo.medical.dataset import load_data
        from training_free_grpo.medical.verify import verify_func
        from training_free_grpo.medical.prompts import PROBLEM_WITH_EXPERIENCE_TEMPLATE
        from training_free_grpo.medical.experience import ExperienceUpdater
        config_name = "simple/base_medical.yaml"
    else:
        raise ValueError(f"Unsupported domain: {args.domain}")
    
    # Create experiment directory
    experiment_dir = os.path.join("data", args.domain, "train", args.experiment_name)
    os.makedirs(experiment_dir, exist_ok=True)

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

    # Load the dataset and prepare for retrieval
    train_data = load_data(args.dataset)
    print(f"Loaded {len(train_data)} records from dataset")
    if args.dataset_truncate is not None:
        print(f"- randomly sampling {args.dataset_truncate} records (seed=42)")
        random.seed(42)
        train_data = random.sample(train_data, args.dataset_truncate)
    
    for i, sample in enumerate(train_data):
        sample["problem_id"] = sample["id"]

    print("Generating embeddings for all problems for retrieval...")
    llm = LLM()
    all_problems_text = [p['problem'] for p in train_data]
    all_problem_embeddings = llm.get_embeddings(all_problems_text)
    
    problem_store = {
        train_data[i]["problem_id"]: {
            "data": train_data[i],
            "embedding": all_problem_embeddings[i],
        }
        for i in range(len(train_data)) if all_problem_embeddings[i] is not None
    }
    print(f"Successfully created embeddings for {len(problem_store)} problems.")


    # Set up the stats
    stats_filename = os.path.join(experiment_dir, "stats.json")
    if os.path.exists(stats_filename):
        stats = json.load(open(stats_filename))
    else:
        stats = {"arg": vars(args)}

    # Train
    for epoch in range(args.epochs):
        # Init
        print("=" * 30 + f"\nEpoch {epoch}\n" + "=" * 30)
        cur_epoch_dir = os.path.join(experiment_dir, f"epoch_{epoch}")
        os.makedirs(cur_epoch_dir, exist_ok=True)
        
        shuffled_ids_filename = os.path.join(cur_epoch_dir, "shuffled_problem_ids.json")
        if os.path.exists(shuffled_ids_filename):
            with open(shuffled_ids_filename) as f:
                problem_ids_in_epoch = json.load(f)
            print(f"Loaded {len(problem_ids_in_epoch)} shuffled problem IDs for epoch {epoch}")
        else:
            problem_ids_in_epoch = list(problem_store.keys())
            random.shuffle(problem_ids_in_epoch)
            with open(shuffled_ids_filename, "w") as f:
                json.dump(problem_ids_in_epoch, f)
            print(f"Shuffled and saved {len(problem_ids_in_epoch)} problem IDs for epoch {epoch}")

        num_batches = (len(problem_ids_in_epoch) + args.batchsize - 1) // args.batchsize
        for batch_idx in range(num_batches):
            step = epoch * num_batches + batch_idx
            if stats.get(f"step_{step}", {}).get("complete"):
                continue

            print(f"Step {step} (Epoch {epoch}, Batch {batch_idx})")
            cur_step_dir = os.path.join(experiment_dir, f"step_{step}")
            os.makedirs(cur_step_dir, exist_ok=True)

            experience_store_filename = os.path.join(cur_step_dir, "per_problem_experiences.json")
            if os.path.exists(experience_store_filename):
                with open(experience_store_filename) as f:
                    per_problem_experiences = {int(k): v for k, v in json.load(f).items()}
                print(f"Loaded {len(per_problem_experiences)} existing per-problem experience dbs.")
            else:
                per_problem_experiences = {}
            
            batch_ids = problem_ids_in_epoch[batch_idx * args.batchsize : (batch_idx + 1) * args.batchsize]
            batch_data = [problem_store[pid]['data'] for pid in batch_ids]

            # Retrieval and Synthesis for the batch
            formatted_batch_data = []
            experience_updater = ExperienceUpdater()
            
            problem_embeddings_matrix = np.array([p["embedding"] for p in problem_store.values()])
            problem_ids_for_retrieval = list(problem_store.keys())

            if per_problem_experiences != {}:
                synthesis_requests = []
                
                for problem_sample in batch_data:
                    current_problem_id = problem_sample['problem_id']
                    current_problem_embedding = problem_store[current_problem_id]['embedding']
                    
                    similarities = cosine_similarity([current_problem_embedding], problem_embeddings_matrix)[0]
                    sorted_indices = np.argsort(similarities)[::-1]  # Sort all problems by similarity, descending
                    
                    retrieved_experiences = {}
                    for idx in sorted_indices:
                        retrieved_problem_id = problem_ids_for_retrieval[idx]
                        
                        # Skip the problem itself to avoid retrieving its own experiences
                        if retrieved_problem_id == current_problem_id:
                            continue

                        # If a similar problem has experiences, add them.
                        if retrieved_problem_id in per_problem_experiences:
                            retrieved_experiences[retrieved_problem_id] = per_problem_experiences[retrieved_problem_id]
                        
                        # Stop once we have collected experiences from 3 distinct problems.
                        if len(retrieved_experiences) >= 3:
                            break
                    
                    synthesis_requests.append({
                        "target_problem": problem_sample["problem"],
                        "retrieved_experiences": retrieved_experiences,
                    })
                
                print(f"Synthesizing experiences for {len(synthesis_requests)} problems in batch...")
                batch_synthesized_experiences, synthesis_prompts = await experience_updater.synthesize_retrieved_experiences_batch_async(
                    synthesis_requests, 
                    concurrency=args.rollout_concurrency
                )

                # Log synthesis data for the entire batch
                synthesis_log_data = []
                for i, request in enumerate(synthesis_requests):
                    synthesis_log_data.append({
                        "target_problem": request["target_problem"],
                        "retrieved_experiences": request["retrieved_experiences"],
                        "synthesis_prompt": synthesis_prompts[i],
                        "synthesized_experiences": batch_synthesized_experiences[i]
                    })
                
                synthesis_log_filename = os.path.join(cur_step_dir, "synthesis_log.json")
                with open(synthesis_log_filename, "w") as f:
                    json.dump(synthesis_log_data, f, indent=2)
                print(f"Saved synthesis log for {len(synthesis_log_data)} problems to {synthesis_log_filename}")

                for i, problem_sample in enumerate(batch_data):
                    synthesized_experiences = batch_synthesized_experiences[i]
                    formatted_experiences = "\n".join([f"[{k}]. {v}" for k, v in synthesized_experiences.items()])
                    
                    formatted_sample = {
                        "prompt": PROBLEM_WITH_EXPERIENCE_TEMPLATE.format(
                            experiences=formatted_experiences,
                            problem=problem_sample["problem"],
                        ),
                        **problem_sample
                    }
                    formatted_batch_data.append(formatted_sample)
            else:
                for i, problem_sample in enumerate(batch_data):
                    formatted_sample = {
                        "prompt": PROBLEM_WITH_EXPERIENCE_TEMPLATE.format(
                            experiences="None",
                            problem=problem_sample["problem"],
                        ),
                        **problem_sample
                    }
                    formatted_batch_data.append(formatted_sample)
            
            # Rollout the dataset
            rollout_filename = os.path.join(cur_step_dir, "rollout.jsonl")
            rollouts = load_rollouts(rollout_filename)
            
            rollouts, rollout_stats = await rollout_dataset(
                worker_agent=worker_agent,
                data=formatted_batch_data * args.grpo_n,
                rollouts=rollouts,
                verify_func=verify_func,
                rollout_filename=rollout_filename,
                rollout_concurrency=args.rollout_concurrency,
                task_timeout=args.task_timeout,
                temperature=args.rollout_temperature,
                max_tokens=args.rollout_max_tokens,
                formatted_experiences=None, # Experiences are now in the prompt
                pass_k=args.grpo_n,
                experiment_name=f"{args.experiment_name}_step_{step}",
                use_trajectory_summary_for_synthesis=args.use_trajectory_summary_for_synthesis,
                cur_step_dir=cur_step_dir,
            )
            
            if f"step_{step}" not in stats:
                stats[f"step_{step}"] = {"epoch": epoch, "batch": batch_idx}
            stats[f"step_{step}"]["rollout"] = rollout_stats

            next_step_dir = os.path.join(experiment_dir, f"step_{step+1}")
            os.makedirs(next_step_dir, exist_ok=True)
            next_experience_filename = os.path.join(next_step_dir, "per_problem_experiences.json")
            if os.path.exists(next_experience_filename):
                print(f"Experiences already exist for step {step}, skipping experience update")
            else:
                # Generate/Update per-problem experiences
                newly_generated_experiences = await experience_updater.run2(
                    rollouts=rollouts, 
                    existing_experiences_per_problem=per_problem_experiences,
                    save_dir=cur_step_dir,
                    max_workers=args.rollout_concurrency,
                    given_ground_truth=(args.given_ground_truth == "True"),
                    only_partial_correct=args.grpo_n > 1,
                )
                per_problem_experiences.update(newly_generated_experiences)
        
                with open(next_experience_filename, "w") as f:
                    json.dump(per_problem_experiences, f, indent=2)
                print(f"Saved {len(newly_generated_experiences)} newly generated experiences to {next_experience_filename}")
            
            stats[f"step_{step}"]["complete"] = True
            json.dump(stats, open(stats_filename, "w"), indent=2)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Training-free GRPO with Per-Problem Experiences")
    parser.add_argument("--mode", type=str, default="agent", required=True, choices=["prompt", "agent"])
    parser.add_argument("--domain", type=str, required=True, choices=["medical"])
    parser.add_argument("--experiment_name", type=str, required=True)
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--dataset_truncate", type=int, default=None)
    parser.add_argument("--given_ground_truth", type=str, default="True")
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batchsize", type=int, default=10)
    parser.add_argument("--grpo_n", type=int, default=5)
    parser.add_argument("--rollout_concurrency", type=int, default=16)
    parser.add_argument("--rollout_temperature", type=float, default=0.7)
    parser.add_argument("--rollout_max_tokens", type=int, default=8192)
    parser.add_argument("--task_timeout", type=float, default=3600)
    parser.add_argument("--use_trajectory_summary_for_synthesis", action="store_true")

    args = parser.parse_args()
    asyncio.run(main(args))
