import json
import copy
import os

import numpy as np
from sklearn.cluster import AgglomerativeClustering
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
from training_free_grpo.llm_sc import LLM
from training_free_grpo.medical.prompts import (
    SINGLE_QUERY_CRITIQUE_TEMPLATE, 
    SINGLE_QUERY_CRITIQUE_NO_GT_TEMPLATE,
    SINGLE_QUERY_CRITIQUE_NO_GT_TEMPLATE2,
    SINGLE_ROLLOUT_SUMMARY_NO_GT_TEMPLATE3,    
    SINGLE_ROLLOUT_SUMMARY_TEMPLATE,
    SINGLE_ROLLOUT_SUMMARY_NO_GT_TEMPLATE,
    SINGLE_ROLLOUT_SUMMARY_NO_GT_TEMPLATE2,
    BATCH_EXPERIENCE_UPDATE_TEMPLATE,
    BATCH_EXPERIENCE_UPDATE_TEMPLATE2,
    SYNTHESIZE_EXPERIENCES_TEMPLATE,
    PER_PROBLEM_EXPERIENCE_UPDATE_TEMPLATE,
    PER_PROBLEM_EXPERIENCE_UPDATE_TEMPLATE2,
    PER_PROBLEM_EXPERIENCE_UPDATE_NO_UNIFIED_TEMPLATE,
    PER_PROBLEM_EXPERIENCE_UPDATE_TEMPLATE_0,
    PER_PROBLEM_EXPERIENCE_UPDATE_TEMPLATE2_0,
    PER_PROBLEM_EXPERIENCE_UPDATE_NO_UNIFIED_TEMPLATE_0,
    SYNTHESIZE_EXPERIENCES_TEMPLATE2,
)

import asyncio

class ExperienceUpdater:
    def __init__(self):
        self.llm = LLM()

    def run(self, rollouts, experiences, save_dir, max_workers=16, given_ground_truth=True, only_partial_correct=True):
        # 1. Summarize trajectory for each rollout
        problem_to_summarized_rollouts = self._single_rollout_summary(
            rollouts=rollouts, 
            save_dir=save_dir, 
            max_workers=max_workers,
            given_ground_truth=given_ground_truth,
            only_partial_correct=only_partial_correct
        )

        # 2. Generate critique for each query
        critiques = self._single_query_critique(
            problem_to_summarized_rollouts=problem_to_summarized_rollouts, 
            experiences=experiences,
            save_dir=save_dir, 
            max_workers=max_workers,
            given_ground_truth=given_ground_truth,
            only_partial_correct=only_partial_correct
        )

        # 3. batch update experiences
        new_experiences = self._batch_update(
            experiences=experiences, 
            critiques=critiques, 
            save_dir=save_dir
        )

        # 4. assign new experience IDs
        new_experiences = {
            f"G{i}": exp for i, exp in enumerate(new_experiences.values())
        }
        return new_experiences

    async def run2(self, rollouts, existing_experiences_per_problem, save_dir, max_workers=16, given_ground_truth=True, only_partial_correct=True):
        """
        New run method to generate experiences on a per-problem basis.
        """
        # 1. Summarize trajectory for each rollout (if not already done)
        problem_to_summarized_rollouts = self._single_rollout_summary(
            rollouts=rollouts, 
            save_dir=save_dir, 
            max_workers=max_workers,
            given_ground_truth=given_ground_truth,
            only_partial_correct=only_partial_correct
        )

        # Group rollouts by problem ID for processing
        problem_id_to_rollouts = defaultdict(list)
        for r in rollouts:
            if r.get("problem_id") is not None:
                problem_id_to_rollouts[r["problem_id"]].append(r)

        # 2. Batch update experiences for all problems
        updated_experiences = await self._update_experiences_for_problem_batch(
            problem_id_to_rollouts=problem_id_to_rollouts,
            problem_to_summarized_rollouts=problem_to_summarized_rollouts,
            existing_experiences_per_problem=existing_experiences_per_problem,
            max_workers=max_workers,
            given_ground_truth=given_ground_truth
        )

        return updated_experiences

    async def _update_experiences_for_problem_batch(self, problem_id_to_rollouts, problem_to_summarized_rollouts, existing_experiences_per_problem, max_workers, given_ground_truth=True):
        prompts = []
        problem_ids_order = []

        for problem_id, p_rollouts in problem_id_to_rollouts.items():
            problem_text = p_rollouts[0]["problem"]
            summarized_rollouts = problem_to_summarized_rollouts.get(problem_text, [])
            
            if not summarized_rollouts:
                continue

            experiences = existing_experiences_per_problem.get(str(problem_id), {})
            formatted_experiences = "\n".join([f"[{i}]. {e}" for i, e in experiences.items()])
            


            representative_rollout = summarized_rollouts[0]
            if given_ground_truth:
                formatted_trajectories = "\n\n".join([
                    f"Trajectory {i+1} (Answer {'correct' if each["reward"] else 'wrong'}):\n{each['trajectory_summary']}"
                    for i, each in enumerate(summarized_rollouts)
                ])
                prompt = PER_PROBLEM_EXPERIENCE_UPDATE_TEMPLATE_0.format(
                    problem=problem_text,
                    experiences=formatted_experiences,
                    trajectories=formatted_trajectories,
                    answer=representative_rollout.get("groundtruth", ""),
                )
            elif representative_rollout.get("is_confident_unified_answer"):
                formatted_trajectories = "\n\n".join([
                    f"Trajectory {i+1} (Answer {'correct' if each["pseudo_reward_from_unified"] else 'wrong'}):\n{each['trajectory_summary']}"
                    for i, each in enumerate(summarized_rollouts)
                ])
                prompt = PER_PROBLEM_EXPERIENCE_UPDATE_TEMPLATE_0.format(
                    problem=problem_text,
                    experiences=formatted_experiences,
                    trajectories=formatted_trajectories,
                    answer=representative_rollout.get("unified_classified_answer", ""),
                )
            elif "unified_response" in representative_rollout:
                formatted_trajectories = "\n\n".join(
                [f"Trajectory {i+1}:\n{r['trajectory_summary']}" for i, r in enumerate(summarized_rollouts)]
            )
                prompt = PER_PROBLEM_EXPERIENCE_UPDATE_TEMPLATE2_0.format(
                    problem=problem_text,
                    experiences=formatted_experiences,
                    trajectories=formatted_trajectories,
                    unified_response=representative_rollout.get("unified_response", ""),
                )
            else:
                formatted_trajectories = "\n\n".join(
                [f"Trajectory {i+1}:\n{r['trajectory_summary']}" for i, r in enumerate(summarized_rollouts)]
            )
                prompt = PER_PROBLEM_EXPERIENCE_UPDATE_NO_UNIFIED_TEMPLATE_0.format(
                    problem=problem_text,
                    experiences=formatted_experiences,
                    trajectories=formatted_trajectories,
                )
            
            prompts.append(prompt)
            problem_ids_order.append(problem_id)

        if not prompts:
            return {}

        responses = await self.llm.chat_batch_async(prompts, concurrency=128)

        new_experiences_per_problem = {}
        for i, response in enumerate(responses):
            problem_id = problem_ids_order[i]
            if response:
                try:
                    data = json.loads(response.split("```json")[-1].split("```")[0])
                    updated_list = data["updated_experiences"]
                    new_experiences_per_problem[problem_id] = {f"G{i}": exp for i, exp in enumerate(updated_list)}
                except (json.JSONDecodeError, KeyError) as e:
                    print(f"Warning: Failed to parse experience update for problem {problem_id}: {e}")
                    new_experiences_per_problem[problem_id] = existing_experiences_per_problem.get(str(problem_id), {})
            else:
                # If LLM call fails, retain the old experiences for that problem
                new_experiences_per_problem[problem_id] = existing_experiences_per_problem.get(str(problem_id), {})

        return new_experiences_per_problem

    async def synthesize_retrieved_experiences_batch_async(self, synthesis_requests, concurrency):
        """
        Synthesizes a new set of experiences from retrieved ones for a batch of target problems.
        """
        prompts = []
        for request in synthesis_requests:
            try:
                formatted_retrieved_experiences = json.dumps(request['retrieved_experiences'], indent=2)
                prompt = SYNTHESIZE_EXPERIENCES_TEMPLATE2.format(
                    target_problem=request['target_problem'],
                    retrieved_experiences=formatted_retrieved_experiences
                )
                prompts.append(prompt)
            except Exception as e:
                print(f"Warning: failed to format synthesis prompt: {e}")
                prompts.append(None)

        # Create a list of valid prompts and their original indices
        indexed_prompts = [(i, p) for i, p in enumerate(prompts) if p is not None]
        indices, valid_prompts = zip(*indexed_prompts) if indexed_prompts else ([], [])
        
        # Call LLM for valid prompts
        llm_responses = await self.llm.chat_batch_async(list(valid_prompts), concurrency=concurrency) if valid_prompts else []

        # Create a full list of responses, with None for failed prompts
        responses = [None] * len(prompts)
        for i, response in zip(indices, llm_responses):
            responses[i] = response

        all_synthesized_experiences = []
        for i, response in enumerate(responses):
            try:
                if response is None:
                    raise ValueError("API call failed or prompt was invalid")
                
                synthesized_experiences_list = json.loads(response.split("```json")[-1].split("```")[0])
                synthesized_experiences = {f"S{j}": exp for j, exp in enumerate(synthesized_experiences_list)}
                all_synthesized_experiences.append(synthesized_experiences)

            except Exception as e:
                print(f"Warning: failed to synthesize experiences for problem {i}: {e}")
                # Fallback: return the first 10 retrieved experiences without synthesis
                retrieved_experiences = synthesis_requests[i]['retrieved_experiences']
                flat_experiences = []
                if isinstance(retrieved_experiences, dict):
                    for exp_dict in retrieved_experiences.values():
                        if isinstance(exp_dict, dict):
                            flat_experiences.extend(exp_dict.values())
                
                all_synthesized_experiences.append({f"F{j}": exp for j, exp in enumerate(flat_experiences[:10])})
        
        return all_synthesized_experiences, prompts


    def _single_rollout_summary(
        self,
        rollouts, 
        save_dir, 
        max_workers,
        given_ground_truth=True,
        only_partial_correct=True
    ):
        # check file existence
        filename = os.path.join(save_dir, "single_rollout_summary.json")
        if os.path.exists(filename):
            try:
                with open(filename) as f:
                    results = json.load(f)
                if len(results) > 0:
                    print("Single rollout summary")
                    print("- File exists, loaded from:", filename)
                    
                    # Merge with current rollouts to get new fields
                    original_rollout_map = {}
                    for r in rollouts:
                        if "trajectories" in r and len(r["trajectories"]) > 0:
                            # The trajectory can be a list, which is not hashable. Convert to JSON string for key.
                            trajectory_str = json.dumps(r['trajectories'][0]['trajectory'], sort_keys=True)
                            key = (r['problem'], trajectory_str)
                            original_rollout_map[key] = r

                    updated_results = defaultdict(list)
                    for problem, summarized_rollouts in results.items():
                        new_summarized_rollouts_for_problem = []
                        for summarized_rollout in summarized_rollouts:
                            # The trajectory can be a list, which is not hashable. Convert to JSON string for key.
                            trajectory_str = json.dumps(summarized_rollout['trajectories'][0]['trajectory'], sort_keys=True)
                            key = (summarized_rollout['problem'], trajectory_str)
                            if key in original_rollout_map:
                                original_rollout = original_rollout_map[key]
                                new_rollout = original_rollout.copy()
                                if 'trajectory_summary' in summarized_rollout:
                                    new_rollout['trajectory_summary'] = summarized_rollout['trajectory_summary']
                                new_summarized_rollouts_for_problem.append(new_rollout)
                        
                        if new_summarized_rollouts_for_problem:
                            updated_results[problem] = new_summarized_rollouts_for_problem
                    
                    if len(updated_results) > 0:
                        print("- Merged cached summaries with current rollout data.")
                        return dict(updated_results)
                    else:
                        print("- Cache is stale, re-summarizing.")
            except (json.JSONDecodeError, KeyError) as e:
                print(f"- Cache file {filename} is corrupted or has an invalid format, re-summarizing. Error: {e}")

        # group by problems
        problems_to_rollouts = defaultdict(list)
        for each in rollouts:
            if "trajectories" in each and len(each["trajectories"]) > 0:
                problems_to_rollouts[each["problem"]].append(each)
        results = defaultdict(list)

        all_rollouts_to_process = []
        for rollouts in problems_to_rollouts.values():
            if  only_partial_correct:
                # only for those partially correct
                if given_ground_truth:
                    scores = [each["reward"] for each in rollouts]
                    avg_score = sum(scores) / len(scores)
                    if avg_score > 0 and avg_score < 1:
                        all_rollouts_to_process.extend(rollouts)
                else:
                    #只要他们的答案不是完全一样的
                    # classified_answers = [r["classified_answer"] for r in rollouts if "classified_answer" in r]
                    # if len(set(classified_answers)) > 1:


                    #只要是高置信度样本
                    # is_confident = [r["is_confident_unified_answer"] for r in rollouts]
                    # if any(is_confident):

                    #只要伪答案奖励partially correct
                    # pseudo_rewards = [r["pseudo_reward_from_unified"] for r in rollouts]
                    # avg_score = sum(pseudo_rewards) / len(pseudo_rewards)
                    # if avg_score > 0 and avg_score < 1:
                        all_rollouts_to_process.extend(rollouts)   
            else:
                all_rollouts_to_process.extend(rollouts)

        def process(cur):
            try:
                if given_ground_truth:
                    prompt = SINGLE_ROLLOUT_SUMMARY_TEMPLATE.format(
                        problem=cur["problem"],
                        trajectory=cur["trajectories"][0]["trajectory"],
                        grade="This trajectory delivers **" + ("correct" if cur.get("reward") else "wrong") + "** answer",
                        answer=cur.get("groundtruth", "")
                    )
                elif cur.get("is_confident_unified_answer"):
                    prompt = SINGLE_ROLLOUT_SUMMARY_TEMPLATE.format(
                        problem=cur["problem"],
                        trajectory=cur["trajectories"][0]["trajectory"],
                        grade="This trajectory delivers **" + ("correct" if cur.get("pseudo_reward_from_unified") else "wrong") + "** answer",
                        answer=cur.get("unified_classified_answer", "")
                    )
                elif "unified_response" in cur:
                    prompt = SINGLE_ROLLOUT_SUMMARY_NO_GT_TEMPLATE2.format(
                        problem=cur["problem"],
                        trajectory=cur["trajectories"][0]["trajectory"],
                        unified_response=cur.get("unified_response", "")
                    )
                else:
                    prompt = SINGLE_ROLLOUT_SUMMARY_NO_GT_TEMPLATE3.format(
                        problem=cur["problem"],
                        trajectory=cur["trajectories"][0]["trajectory"]
                    )

                response = self.llm.chat(prompt)
                return {"trajectory_summary": response, **cur}
            except Exception as e:
                print(f"Warning: failed in single rollout summary, {e}")
                return None

        # parallel running
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_rollout = {executor.submit(process, cur): cur for cur in all_rollouts_to_process}
            for future in tqdm(
                as_completed(future_to_rollout), total=len(all_rollouts_to_process), desc="Single rollout summary"
            ):
                result = future.result()
                if result is not None:
                    problem = result["problem"]
                    results[problem].append(result)

        # write to file
        with open(filename, "w") as f:
            json.dump(results, f, indent=2)
        return results


    def _single_query_critique(
        self,
        problem_to_summarized_rollouts, 
        experiences, 
        save_dir, 
        max_workers, 
        max_operations=1,
        given_ground_truth=True,
        only_partial_correct=True
    ):

        # check file existence
        filename = os.path.join(save_dir, "single_query_critique.json")
        if os.path.exists(filename):
            with open(filename) as f:
                results = json.load(f)
                if len(results) > 0:
                    print("Single query critique")
                    print("- File exists, loaded from:", filename)
                    return results

        all_rollouts = []
        for rollouts in problem_to_summarized_rollouts.values():
            if given_ground_truth and only_partial_correct:
                # only for those partially correct
                if given_ground_truth:
                    scores = [each["reward"] for each in rollouts]
                    avg_score = sum(scores) / len(scores)
                    if avg_score > 0 and avg_score < 1:
                        all_rollouts.append(rollouts)
                else:
                    #只要他们的答案不是完全一样的
                    # classified_answers = [r["classified_answer"] for r in rollouts if "classified_answer" in r]
                    # if len(set(classified_answers)) > 1:
                        all_rollouts.append(rollouts)   
            else:
                all_rollouts.append(rollouts)

        def process(rollouts_per_problem):
            try:
                problem = rollouts_per_problem[0]["problem"]
                answer = rollouts_per_problem[0]["groundtruth"]
                formatted_trajectories = "\n\n".join([
                    f"Trajectory {i+1} (Answer {'correct' if each["reward"] else 'wrong'}):\n{each['trajectory_summary']}"
                    for i, each in enumerate(rollouts_per_problem)
                ])
                formatted_experiences = "\n".join([ f"[{i}]. {e}" for i, e in experiences.items() ]) if experiences else "None"
                response = self.llm.chat(
                    SINGLE_QUERY_CRITIQUE_TEMPLATE.format(
                        max_operations=max_operations,
                        problem=problem,
                        trajectories=formatted_trajectories,
                        answer=answer,
                        experiences=formatted_experiences,
                    ) if given_ground_truth else
                    SINGLE_QUERY_CRITIQUE_NO_GT_TEMPLATE2.format(
                        max_operations=max_operations,
                        problem=problem,
                        trajectories="\n\n".join([
                            f"Trajectory {i+1}:\n{each['trajectory_summary']}" for i, each in enumerate(rollouts_per_problem)
                        ]),
                        experiences=formatted_experiences,
                        unified_response=rollouts_per_problem[0]["unified_response"],
                    ) if rollouts_per_problem[0].get("unified_response") else 
                    SINGLE_QUERY_CRITIQUE_NO_GT_TEMPLATE.format(
                        max_operations=max_operations,
                        problem=problem,
                        trajectories="\n\n".join([
                            f"Trajectory {i+1}:\n{each['trajectory_summary']}" for i, each in enumerate(rollouts_per_problem)
                        ]),
                        experiences=formatted_experiences,
                    )
                )
                response = response.split("```json")[-1].split("```")[0]
                operations = json.loads(response)
                return {"rollouts": rollouts_per_problem, "critique": response, "operations": operations[:max_operations]}
            except Exception as e:
                print(f"Warning: failed in single query critique, {e}")
                return None

        # parallel running
        results = []
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_case = {
                executor.submit(process, rollouts_per_problem): rollouts_per_problem
                for rollouts_per_problem in all_rollouts
            }
            for future in tqdm(as_completed(future_to_case), total=len(all_rollouts), desc="Single query critique"):
                result = future.result()
                if result is not None:
                    results.append(result)

        # write results
        with open(filename, "w") as f:
            json.dump(results, f, indent=2)
        return results


    def _batch_update(
        self,
        experiences, 
        critiques, 
        save_dir,
        max_retries=3
    ):
        print("Batch update")
        filename = os.path.join(save_dir, "batch_update.json")
        if os.path.exists(filename):
            results = json.load(open(filename))
            print("- File exists, loaded from:", filename)
            return results
        
        # collect operations
        all_operations = []
        for each in critiques:
            try:
                all_operations.extend(each["operations"])
            except:
                print(f"Warning: failed to decode operation: {each}")
        print("- Num of operations to process:", len(all_operations))

        # split experiences
        candidate_experiences = copy.deepcopy(experiences)
        to_modify = []
        max_ID = 0
        for operation in all_operations:
            try:
                if operation["option"] == "modify":
                    if operation["modified_from"] in candidate_experiences:
                        to_modify.append(operation)
                elif operation["option"] == "add":
                    candidate_experiences[f"C{max_ID}"] = operation["experience"]
                    max_ID += 1
            except:
                print(f"Warning: failed to decode operation: {operation}")

        print("- Num of added experiences:", max_ID)
        print("- Num of experiences to be modified:", len(to_modify))
        print("- Num of candidate experiences:", len(candidate_experiences))

        # use LLM to get the revision plan
        revision_plan = []
        for _ in range(max_retries):
            try:
                response = self.llm.chat(
                    BATCH_EXPERIENCE_UPDATE_TEMPLATE2.format(
                        experiences=candidate_experiences, 
                        updates=to_modify
                    )
                )
                revision_plan = json.loads(response.split("```json")[-1].split("```")[0])
                break
            except Exception:
                print("Warning: failed to decode in updating general experiences")

        # modify candidate experiences
        new_experiences = copy.deepcopy(candidate_experiences)
        for operation in revision_plan:
            try:
                if operation["option"] == "modify":
                    new_experiences[operation["modified_from"]] = operation["experience"]
                elif operation["option"] == "merge":
                    for ID in operation["merged_from"]:
                        if ID not in new_experiences:
                            raise Exception(f"ID {ID} not found for merging")
                    for ID in operation["merged_from"]:
                        if ID in new_experiences:
                            del new_experiences[ID]
                    new_experiences[f"C{max_ID}"] = operation["experience"]
                    max_ID += 1
            except Exception as e:
                print("Error: failed to complete experience update:", operation, "|", e)
        print("- Num of revised candidate experiences:", len(new_experiences))

        # Deduplicate if experiences are more than 40
        if len(new_experiences) > 40:
            print(f"- Experience count ({len(new_experiences)}) is over 40, starting deduplication.")
            
            ids = list(new_experiences.keys())
            texts = list(new_experiences.values())
            
            # Get embeddings
            embeddings = self.llm.get_embeddings(texts)
            
            # Filter out None embeddings
            valid_embeddings = [emb for emb in embeddings if emb is not None]
            valid_indices = [i for i, emb in enumerate(embeddings) if emb is not None]
            
            if len(valid_embeddings) > 1:
                # Perform clustering
                clustering = AgglomerativeClustering(
                    n_clusters=min(40, len(valid_embeddings)), linkage='average', metric='cosine'
                )
                labels = clustering.fit_predict(np.array(valid_embeddings))
                
                # Keep one experience per cluster
                final_experiences = {}
                kept_clusters = set()
                for i, label in enumerate(labels):
                    if label not in kept_clusters:
                        original_index = valid_indices[i]
                        final_experiences[ids[original_index]] = texts[original_index]
                        kept_clusters.add(label)
                
                new_experiences = final_experiences
                print(f"- Reduced experience count to {len(new_experiences)} after deduplication.")
            else:
                print("- Not enough valid embeddings to perform deduplication.")

        # write to file
        with open(filename, "w") as f:
            json.dump(
                {
                    "operations": all_operations,
                    "response": response,
                    "revision_plan": revision_plan,
                    "new_experiences": new_experiences,
                },
                f,
                indent=2,
            )
        return new_experiences
