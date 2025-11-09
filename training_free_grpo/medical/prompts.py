PROBLEM_WITH_EXPERIENCE_TEMPLATE = """Please solve the problem:
{problem}

When solving problems, you MUST first carefully read and understand the helpful instructions and experiences:
{experiences}"""

PROBLEM_WITH_EXPERIENCE_AND_EXPERT_TEMPLATE = """You are an expert with a specific perspective.
Your persona is: {expert_persona}

Please solve the following problem based on your assigned persona.

Problem:
{problem}

When solving problems, you MUST first carefully read and understand the helpful instructions and experiences, and integrate them with your persona's perspective:
{experiences}"""

DIVERSE_EXPERTS_PERSONAS = [
    "You are a cautious, detail-oriented expert. Your priority is to rule out all worst-case scenarios and ensure every step is backed by strong evidence. You double-check every piece of information.",
    "You are a pragmatic, cost-conscious expert. You focus on finding the most efficient and economically viable solution without compromising essential quality or safety. You weigh the costs and benefits of each action.",
    "You are an evidence-based expert who strictly follows the latest clinical guidelines and research. You prioritize well-established, scientifically-proven methods over anecdotal or outdated practices.",
    "You are a holistic, patient-centered expert. You consider the patient's overall well-being, including their comfort, preferences, and emotional state. You aim for solutions that are not just technically correct but also compassionate.",
    "You are an innovative, forward-thinking expert. You are open to considering cutting-edge techniques, experimental approaches, and novel solutions that may offer superior outcomes, even if they are not yet standard practice."
]

GENERATE_MEDICAL_FIELDS_TEMPLATE = """You are a medical expert who specializes in categorizing a specific medical scenario into specific areas of medicine.
Based on the medical scenario in the question below, classify the question into {num_experts} different subfields of medicine.

<problem>
{problem}
</problem>

You should output in exactly the same format as 'Medical Field: Field1 | Field2 | ...'. For example: 'Medical Field: Cardiology | Neurology | Oncology'.
"""


SINGLE_ROLLOUT_SUMMARY_TEMPLATE = """An agent system may be provided with some experiences, and then it produces the following trajectory to solve the given problem. Please summarize the trajectory step-by-step:

1. For each step, describe **what action is being taken**, and which experience has been used in this step.
2. Given the grading of this rollout and the correct answer, identify and explain any steps that **represent detours, errors, or backtracking**, highlighting why they might have occurred and what their impact was on the trajectory's progress. 
3. Maintain **all the core outcome of each step**, even if it was part of a flawed process.

<problem>
{problem}
</problem>

<trajectory>
{trajectory}
</trajectory>

<evaluation>
{grade}
</evaluation>

<groundtruth>
{answer}
</groundtruth>

Only return the trajectory summary of each step, e.g.,
1. what happened in the first step and the core outcomes
2. what happened in the second step and the core outcomes
3. ..."""

SINGLE_ROLLOUT_SUMMARY_TEMPLATE2 = """An agent system may be provided with some experiences, and then it produces the following trajectory to solve the given problem. Please summarize the trajectory step-by-step:

1. For each step, describe **what action is being taken**, and which experience has been used in this step.
2. Given the grading of this rollout and the correct answer, identify and explain any steps that **represent detours, errors, or backtracking**, highlighting why they might have occurred and what their impact was on the trajectory's progress. 
3. Maintain **all the core outcome of each step**, even if it was part of a flawed process.

<problem>
{problem}
</problem>

<trajectory>
{trajectory}
</trajectory>

<evaluation>
{grade}
</evaluation>

<groundtruth>
{answer}
</groundtruth>

Only return the trajectory summary of each step, e.g.,
1. what happened in the first step and the core outcomes
2. what happened in the second step and the core outcomes
3. ..."""


SINGLE_QUERY_CRITIQUE_TEMPLATE = """An agent system is provided with a set of experiences and has tried to solve the problem multiple times with both successful and wrong solutions. Review these problem-solving attempt and extract generalizable experiences. Follow these steps:

1. Trajectory Analysis:
    - For successful steps: Identify key correct decisions and insights
    - For errors: Pinpoint where and why the reasoning went wrong
    - Note any important patterns or strategies used/missed
    - Review why some trajectories fail? Is there any existing experiences are missed, or experiences do not provide enough guidance?

2. Update Existing Experiences
    - Some trajectories may be correct and others may be wrong, you should ensure there are experiences can help to run correctly
    - You have two options: [modify, add]
        * modify: You can modify current experiences to make it helpful
        * add: You can introduce new experiences may need to be 
    - You can update at most {max_operations} clear, generalizable lessons for this case
    - Before updating each experience, you need to:
        * Specify when it would be most relevant
        * List key problem features that make this experience applicable
        * Identify similar problem patterns where this advice applies
    
3. Requirements for each experience that is modified or added.
    - Begin with general background with several words in the experience
    - Focus on strategic thinking patterns, not specific calculations
    - Emphasize decision points that could apply to similar problems

Please provide reasoning in details under the guidance of the above 3 steps.
After the step-by-step reasoning, you will finish by returning in this JSON format as follows:
```json
[
    {{
        "option": "modify",
        "experience": "the modified experience",
        "modified_from": "G17" # specify the ID of experience that is modified
    }},
    {{
        "option": "add",
        "experience": "the added experience",
    }},
    ...
]
```
Note that your updated experiences may not need to cover all two options. Only using one type of updates is also very good.

<problem> 
{problem}
</problem>

<trajectories>
{trajectories}
</trajectories>

<groundtruth>
{answer}
</groundtruth>

<experience>
{experiences}
</experience>"""


BATCH_EXPERIENCE_UPDATE_TEMPLATE = """An agent system is provided with a set of experiences and has tried to solve the problem multiple times. From the reflections, some suggestions on the existing experiences have been posed. Your task is to collect and think for the final experience revision plan. Each final experience must satisfy the following requirements.
1. It must be clear, generalizable lessons for this case, with no more than 32 words
2. Begin with general background with several words in the experience
3. Focus on strategic thinking patterns, not specific calculations
4. Emphasize decision points that could apply to similar problems
5. Avoid repeating saying similar experience in multiple different experiences

<existing_experiences> 
{experiences}
</existing_experiences>

<suggested_updates>
{updates}
</suggested_updates>

Please provide reasoning in each of the suggestions, and think for how to update existing experiences 
You have two update options: [modify, merge]
* modify: You can modify current experiences to make it helpful
* merge: You can merge some similar experiences into a more general forms to reduce duplication

After generating the step-by-step reasoning, you need to give the final experience revision details by returning in this JSON format as follows:
```json
[
    {{
        "option": "modify",
        "experience": "the modified experience",
        "modified_from": "C1" # specify the str ID of experience that is modified
    }},
    {{
        "option": "merge",
        "experience": "the merged experience",
        "merged_from": ["C1", "C3", "S4", ...] # specify the str IDs of experiences that is merged from, at least 2 IDs are needed
    }},
    ...
]
```

Your updated experiences may not need to cover all two options. Only using one type of updates is OK."""

BATCH_EXPERIENCE_UPDATE_TEMPLATE2 = """An agent system is provided with a set of experiences and has tried to solve the problem multiple times. From the reflections, some suggestions on the existing experiences have been posed. Your task is to collect and think for the final experience revision plan. Each final experience must satisfy the following requirements.
1. It must be clear, generalizable lessons for this case, with no more than 32 words
2. Begin with general background with several words in the experience
3. Focus on strategic thinking patterns, not specific calculations
4. Emphasize decision points that could apply to similar problems
5. Avoid repeating saying similar experience in multiple different experiences
6. IMPORTANT: If the number of existing experiences is greater than 40, you must merge some relevant experiences into a more general forms to reduce the total number to below 40.

<existing_experiences> 
{experiences}
</existing_experiences>

<suggested_updates>
{updates}
</suggested_updates>

Please provide reasoning in each of the suggestions, and think for how to update existing experiences 
You have two update options: [modify, merge]
* modify: You can modify current experiences to make it helpful
* merge: You can merge some similar experiences into a more general forms to reduce duplication

After generating the step-by-step reasoning, you need to give the final experience revision details by returning in this JSON format as follows:
```json
[
    {{
        "option": "modify",
        "experience": "the modified experience",
        "modified_from": "C1" # specify the str ID of experience that is modified
    }},
    {{
        "option": "merge",
        "experience": "the merged experience",
        "merged_from": ["C1", "C3", "S4", ...] # specify the str IDs of experiences that is merged from, at least 2 IDs are needed
    }},
    ...
]
```

Your updated experiences may not need to cover all two options. Only using one type of updates is OK."""


SINGLE_ROLLOUT_SUMMARY_NO_GT_TEMPLATE = """An agent system may be provided with some experiences, and then it produces the following trajectory to solve the given problem. Please summarize the trajectory step-by-step:

1. For each step, describe **what action is being taken**, and which experience has been used in this step.
2. Given the grading of this rollout and the correct answer, identify and explain any steps that **represent detours, errors, or backtracking**, highlighting why they might have occurred and what their impact was on the trajectory's progress. 
3. Maintain **all the core outcome of each step**, even if it was part of a flawed process.

<problem>
{problem}
</problem>

<trajectory>
{trajectory}
</trajectory>

Only return the trajectory summary of each step, e.g.,
1. what happened in the first step and the core outcomes
2. what happened in the second step and the core outcomes
3. ..."""


SINGLE_ROLLOUT_SUMMARY_NO_GT_TEMPLATE2 = """An agent system may be provided with some experiences, and then it produces the following trajectory to solve the given problem. Please summarize the trajectory step-by-step:

1. For each step, describe **what action is being taken**, and which experience has been used in this step.
2. Given the unified response synthesized from different trajectories, identify and explain any steps that **represent detours, errors, or backtracking**, highlighting why they might have occurred and what their impact was on the trajectory's progress. 
3. Maintain **all the core outcome of each step**, even if it was part of a flawed process.

<problem>
{problem}
</problem>

<trajectory>
{trajectory}
</trajectory>

<unified_response>
{unified_response}
</unified_response>

Only return the trajectory summary of each step, e.g.,
1. what happened in the first step and the core outcomes
2. what happened in the second step and the core outcomes
3. ..."""


SINGLE_ROLLOUT_SUMMARY_NO_GT_TEMPLATE3 = """An agent system may be provided with some experiences, and then it produces the following trajectory to solve the given problem. Please summarize the trajectory step-by-step:

1. For each step, describe **what action is being taken**, and which experience has been used in this step.
2. Based on the trajectory, identify and explain any steps that **represent detours, errors, or backtracking**, highlighting why they might have occurred and what their impact was on the trajectory's progress.
3. Maintain **all the core outcome of each step**, even if it was part of a flawed process.

<problem>
{problem}
</problem>

<trajectory>
{trajectory}
</trajectory>

Only return the trajectory summary of each step, e.g.,
1. what happened in the first step and the core outcomes
2. what happened in the second step and the core outcomes
3. ..."""


SINGLE_QUERY_CRITIQUE_NO_GT_TEMPLATE = """An agent system is provided with a set of experiences and has tried to solve the problem multiple times. Review these problem-solving attempt and extract generalizable experiences. Follow these steps:

1. Trajectory Analysis:
    - Identify key correct decisions and insights
    - Pinpoint where and why the reasoning went wrong
    - Note any important patterns or strategies used/missed
    - Review why some trajectories seems to fail? Is there any existing experiences are missed, or experiences do not provide enough guidance?

2. Update Existing Experiences
    - Ensure there are experiences can help to run correctly
    - You have two options: [modify, add]
        * modify: You can modify current experiences to make it helpful
        * add: You can introduce new experiences may need to be 
    - You can update at most {max_operations} clear, generalizable lessons for this case
    - Before updating each experience, you need to:
        * Specify when it would be most relevant
        * List key problem features that make this experience applicable
        * Identify similar problem patterns where this advice applies
    
3. Requirements for each experience that is modified or added.
    - Begin with general background with several words in the experience
    - Focus on strategic thinking patterns, not specific calculations
    - Emphasize decision points that could apply to similar problems

Please provide reasoning in details under the guidance of the above 3 steps.
After the step-by-step reasoning, you will finish by returning in this JSON format as follows:
```json
[
    {{
        "option": "modify",
        "experience": "the modified experience",
        "modified_from": "G17" # specify the ID of experience that is modified
    }},
    {{
        "option": "add",
        "experience": "the added experience",
    }},
    ...
]
```
Note that your updated experiences may not need to cover all two options. Only using one type of updates is also very good.

<problem> 
{problem}
</problem>

<trajectories>
{trajectories}
</trajectories>

<experience>
{experiences}
</experience>"""


SINGLE_QUERY_CRITIQUE_NO_GT_TEMPLATE2 = """An agent system is provided with a set of experiences and has tried to solve the problem multiple times. A unified response, which synthesizes the individual attempts, is also provided. Review these problem-solving attempts and the unified response to extract generalizable experiences. Follow these steps:

1. Trajectory and Unified Response Analysis:
    - Compare the individual trajectories with the unified response. Does the unified response provide a better solution?
    - Identify key correct decisions and insights from both the trajectories and the unified response.
    - Pinpoint where and why the reasoning went wrong in the individual trajectories.
    - Note any important patterns or strategies used/missed.
    - Review why some trajectories seem to fail. Are there existing experiences that were missed, or do the experiences not provide enough guidance?

2. Update Existing Experiences
    - Ensure there are experiences can help to run correctly
    - You have two options: [modify, add]
        * modify: You can modify current experiences to make it helpful
        * add: You can introduce new experiences may need to be 
    - You can update at most {max_operations} clear, generalizable lessons for this case
    - Before updating each experience, you need to:
        * Specify when it would be most relevant
        * List key problem features that make this experience applicable
        * Identify similar problem patterns where this advice applies
    
3. Requirements for each experience that is modified or added.
    - Begin with general background with several words in the experience
    - Focus on strategic thinking patterns, not specific calculations
    - Emphasize decision points that could apply to similar problems

Please provide reasoning in details under the guidance of the above 3 steps.
After the step-by-step reasoning, you will finish by returning in this JSON format as follows:
```json
[
    {{
        "option": "modify",
        "experience": "the modified experience",
        "modified_from": "G17" # specify the ID of experience that is modified
    }},
    {{
        "option": "add",
        "experience": "the added experience",
    }},
    ...
]
```
Note that your updated experiences may not need to cover all two options. Only using one type of updates is also very good.

<problem> 
{problem}
</problem>

<trajectories>
{trajectories}
</trajectories>

<unified_response>
{unified_response}
</unified_response>

<experience>
{experiences}
</experience>"""



UNIFIED_RESPONSE_TEMPLATE1 = """
You are tasked with combining multiple responses for the same problem into a single, cohesive response.
Below, I will provide several responses.
Your goal is to identify common themes, reconcile differences, and combine the information
into a unified response.
Be sure to preserve all key insights from each trajectory and ensure the final output is logically
consistent and comprehensive.

<problem> 
{problem}
</problem>

<trajectories>
{trajectories}
</trajectories>


Output Format:
Combine all the provided responses into a new, comprehensive, complete, and unified response, prefixed by “# UNIFIED RESPONSE”.
Your response should not be much longer than the original responses.
"""

UNIFIED_RESPONSE_TEMPLATE2 = """
You are tasked with aggregating multiple responses for the same problem into a single, cohesive response.
Below, I will provide several responses.
Your goal is to identify common themes, reconcile differences, and synthesize the information into a unified response.
Be sure to preserve key insights from each trajectory and ensure the final output is logically consistent and comprehensive.
Avoid discarding unique or contradictory insights; highlight and address them where possible.

<problem> 
{problem}
</problem>

<trajectories>
{trajectories}
</trajectories>


Output Format:
Provide a detailed, aggregated explanation or summary that integrates the information from
the traces above, prefixed by “# SUMMARY”
If there are contradictions or unresolved aspects, clearly state them and propose a way to reconcile them.
Next, based on your summary and all of the prior responses, provide a new, comprehensive,
complete, and unified response, prefixed by “# UNIFIED RESPONSE”.
ALWAYS present your final answer in the following format:
<answer>
(final answer)
</answer>

N.B. Make sure that the final answer is properly wrapped inside the <answer> block.For multiple-choice questions: Only provide the letter choice (e.g., (A))
"""

UNIFIED_RESPONSE_TEMPLATE3 = """
You are tasked with aggregating multiple responses for the same problem into a single, cohesive response.
Below, I will provide several responses.
Your goal is to identify common themes, reconcile differences, and synthesize the information into a unified response.
Be sure to preserve key insights from each trajectory and ensure the final output is logically consistent and comprehensive.
Avoid discarding unique or contradictory insights; highlight and address them where possible.

<problem> 
{problem}
</problem>

<trajectories>
{trajectories}
</trajectories>

When sanalyzing trajectories, you MUST first carefully read and understand the helpful instructions and experiences:

<experience>
{experiences}
</experience>

Output Format:
Provide a detailed, aggregated explanation or summary that integrates the information from
the trajectories above, prefixed by “# SUMMARY”
If there are contradictions or unresolved aspects, clearly state them and propose a way to reconcile them.
Next, based on your summary and all of the prior responses, provide a new, comprehensive,
complete, and unified response, prefixed by “# UNIFIED RESPONSE”.
ALWAYS present your final answer in the following format:
<answer>
(final answer)
</answer>

N.B. Make sure that the final answer is properly wrapped inside the <answer> block.For multiple-choice questions: Only provide the letter choice (e.g., (A))
"""

SYNTHESIZE_EXPERIENCES_TEMPLATE = """You are an expert assistant responsible for creating a concise, highly relevant set of problem-solving experiences for a target problem.
You will be given a target problem and a collection of experiences retrieved from similar past problems.

Your task is to analyze all the retrieved experiences and synthesize them into a new, smaller set of experiences (no more than 10) that are most likely to help solve the **target problem**.

Follow these steps:
1.  **Analyze the Target Problem**: First, deeply understand the core challenge, required knowledge, and potential pitfalls of the target problem.
2.  **Review Retrieved Experiences**: Go through each retrieved experience. Identify which ones are directly applicable, which are partially relevant, and which are likely noise for the target problem.
3.  **Synthesize and Refine**:
    -   Merge experiences that share the same core idea into a single, more general, and clearer experience.
    -   Prioritize strategies and decision-making processes over simple facts.
    -   Rephrase experiences to be more concise and directly applicable to the target problem's context.
    -   Discard any experiences that are redundant, too specific to the original problem, or irrelevant to the target problem.

**Final Output Requirements**:
-   The output must be a JSON list of strings.
-   The list must contain **at most 10** experiences.
-   Each experience in the list should be a clear, actionable string, following the original format.

Below is the target problem and the retrieved experiences.

<target_problem>
{target_problem}
</target_problem>

<retrieved_experiences>
{retrieved_experiences}
</retrieved_experiences>

Now, provide the synthesized list of experiences in the following JSON format:
```json
[
    "Synthesized experience 1...",
    "Synthesized experience 2...",
    ...
]
```
"""


SYNTHESIZE_EXPERIENCES_TEMPLATE2 = """You are an expert assistant skilled at abstracting and generalizing problem-solving strategies.
Your goal is to synthesize a set of highly transferable and generalizable experiences from a collection of retrieved examples that can guide the solving of a new target problem.

You will be given a target problem and a collection of experiences retrieved from similar past problems.

Your task is to **distill universal principles and strategies** from the retrieved experiences. Instead of creating solutions specific to the target problem, you must produce a new, smaller set of experiences (no more than 10) that represent generalizable wisdom.

Follow these steps:
1.  **Analyze the Target Problem**: First, deeply understand the core challenge and the category of the target problem. This helps in identifying which retrieved experiences are most relevant for generalization.
2.  **Review Retrieved Experiences for Underlying Principles**: For each retrieved experience, ask "What is the general principle or common pitfall being illustrated here?". Look for patterns and recurring themes across all experiences.
3.  **Synthesize and Generalize**:
    -   **Abstract**: Convert specific instances into general rules. For example, if an experience says "For symptom A, consider disease X," a more general rule might be "When a primary symptom is present, consider its most common associated diseases first."
    -   **De-contextualize**: Remove details that are only relevant to the original problem. The synthesized experience should be broadly applicable.
    -   **Merge and Refine**: Combine experiences that share the same core idea into a single, more powerful and broadly applicable principle.
    -   **Prioritize**: Focus on creating experiences about strategic thinking, diagnostic processes, and common pitfalls over simple factual knowledge.
    -   **Discard**: Eliminate any experiences that are too narrow, redundant, or irrelevant to the general principles of the problem domain.

**Final Output Requirements**:
-   The output must be a JSON list of strings.
-   The list must contain **at most 10** experiences.
-   Each experience must be a **general and transferable principle or strategy**, not just a solution step for the target problem.

Below is the target problem and the retrieved experiences.

<target_problem>
{target_problem}
</target_problem>

<retrieved_experiences>
{retrieved_experiences}
</retrieved_experiences>

Now, provide the synthesized list of experiences in the following JSON format:
```json
[
    "Synthesized experience 1...",
    "Synthesized experience 2...",
    ...
]
```
"""



PER_PROBLEM_EXPERIENCE_UPDATE_TEMPLATE = """You are an expert assistant for refining a knowledge base of problem-solving experiences.
For a single problem, you will be given the ground-truth answer, existing experiences, and several new solution attempts (trajectories).

Your task is to **update the existing experiences** based on the new information from the trajectories. The goal is to produce a new, improved set of experiences for this specific problem.

Follow these steps:

1.  **Analyze the Attempts**: Review the `<trajectories>` and compare them against the `<ground_truth_answer>`. Identify:
    *   **Correct Strategies**: What reasoning paths in successful trajectories should be preserved or added as new experiences?
    *   **Errors & Pitfalls**: What mistakes were made in incorrect trajectories that could be prevented with a new or modified experience?
    *   **Redundancy**: Are any existing experiences verbose, duplicated, or less effective than what's shown in the new attempts?

2.  **Plan the Update**: Based on your analysis, decide on a set of update operations. You have three options:
    *   `add`: Introduce a new, generalizable experience learned from the new attempts.
    *   `modify`: Rewrite an existing experience to be clearer, more accurate, or more general.
    *   `merge`: Combine multiple existing experiences that are similar or related into a single, more powerful one.

3.  **Generate the Final Experience Set**: Apply your update plan to the `<existing_experiences>` and produce the final, updated list of experiences.

**Final Output Requirements**:
*   The output must be a single JSON object.
*   The object should contain a key `updated_experiences`, which is a list of strings.
*   Each string in the list is a complete, refined experience.

<problem>
{problem}
</problem>

<ground_truth_answer>
{answer}
</ground_truth_answer>

<existing_experiences>
{experiences}
</existing_experiences>

<trajectories>
{trajectories}
</trajectories>

Now, provide the final, updated set of experiences for this problem in the following JSON format:
```json
{{
    "updated_experiences": [
        "Updated experience 1...",
        "A newly added experience...",
        "A merged and refined experience...",
        ...
    ]
}}
```
"""

PER_PROBLEM_EXPERIENCE_UPDATE_TEMPLATE2 = """You are an expert assistant for refining a knowledge base of problem-solving experiences.
For a single problem, you will be given the existing experiences, several new solution attempts (trajectories), and a high-quality unified response synthesized from these attempts.

Your task is to **update the existing experiences** based on the new information from the trajectories and the unified response. The goal is to produce a new, improved set of experiences for this specific problem.

Follow these steps:

1.  **Analyze the Attempts**: Review the `<trajectories>` and the `<unified_response>`. The unified response often represents a better path. Identify:
    *   **Correct Strategies**: What reasoning paths in the unified response or successful trajectories should be preserved or added as new experiences?
    *   **Errors & Pitfalls**: What mistakes were made in the trajectories that could be prevented with a new or modified experience?
    *   **Redundancy**: Are any existing experiences verbose, duplicated, or less effective than what's shown in the new attempts?

2.  **Plan the Update**: Based on your analysis, decide on a set of update operations. You have three options:
    *   `add`: Introduce a new, generalizable experience learned from the new attempts.
    *   `modify`: Rewrite an existing experience to be clearer, more accurate, or more general.
    *   `merge`: Combine multiple existing experiences that are similar or related into a single, more powerful one.

3.  **Generate the Final Experience Set**: Apply your update plan to the `<existing_experiences>` and produce the final, updated list of experiences.

**Final Output Requirements**:
*   The output must be a single JSON object.
*   The object should contain a key `updated_experiences`, which is a list of strings.
*   Each string in the list is a complete, refined experience.

<problem>
{problem}
</problem>

<existing_experiences>
{experiences}
</existing_experiences>

<trajectories>
{trajectories}
</trajectories>

<unified_response>
{unified_response}
</unified_response>

Now, provide the final, updated set of experiences for this problem in the following JSON format:
```json
{{
    "updated_experiences": [
        "Updated experience 1...",
        "A newly added experience from the unified response...",
        "A merged and refined experience...",
        ...
    ]
}}
```
"""

PER_PROBLEM_EXPERIENCE_UPDATE_NO_UNIFIED_TEMPLATE = """You are an expert assistant for refining a knowledge base of problem-solving experiences.
For a single problem, you will be given the existing experiences and several new solution attempts (trajectories).

Your task is to **update the existing experiences** based on the new information from the trajectories. The goal is to produce a new, improved set of experiences for this specific problem.

Follow these steps:

1.  **Analyze the Attempts**: Review the `<trajectories>`. Identify:
    *   **Common Strategies**: What are the common reasoning paths in the trajectories?
    *   **Potential Errors**: What are common mistakes that could be prevented?
    *   **Redundancy**: Are any existing experiences verbose, duplicated, or less effective than what's shown in the new attempts?

2.  **Plan the Update**: Based on your analysis, decide on a set of update operations. You have three options:
    *   `add`: Introduce a new, generalizable experience learned from the new attempts.
    *   `modify`: Rewrite an existing experience to be clearer, more accurate, or more general.
    *   `merge`: Combine multiple existing experiences that are similar or related into a single, more powerful one.

3.  **Generate the Final Experience Set**: Apply your update plan to the `<existing_experiences>` and produce the final, updated list of experiences.

**Final Output Requirements**:
*   The output must be a single JSON object.
*   The object should contain a key `updated_experiences`, which is a list of strings.
*   Each string in the list is a complete, refined experience.

<problem>
{problem}
</problem>

<existing_experiences>
{experiences}
</existing_experiences>

<trajectories>
{trajectories}
</trajectories>

Now, provide the final, updated set of experiences for this problem in the following JSON format:
```json
{{
    "updated_experiences": [
        "Updated experience 1...",
        "A newly added experience...",
        "A merged and refined experience...",
        ...
    ]
}}
```
"""

PER_PROBLEM_EXPERIENCE_UPDATE_TEMPLATE_0 = """You are an expert assistant skilled at distilling generalizable principles from concrete examples.
For a single problem, you will be given the ground-truth answer, a set of existing experiences, and several new solution attempts (trajectories).

Your task is to **refine the existing experience set by extracting transferable wisdom** from the new trajectories, using the ground-truth as a reference. The goal is to produce an improved set of **generalizable experiences**.

Follow these steps:

1.  **Analyze the Attempts for General Principles**: Review the `<trajectories>` and compare them against the `<ground_truth_answer>`. Do not just identify what is right or wrong; identify *why*.
    *   **Underlying Correct Strategies**: What generalizable reasoning paths in successful trajectories led to the correct answer? Abstract these into principles.
    *   **Common Errors & Pitfalls**: What general mistakes were made in incorrect trajectories? Formulate experiences that help prevent these general pitfalls in the future.
    *   **Redundancy and Specificity**: Are any existing experiences too specific, redundant, or less effective than the principles learned from the new attempts?

2.  **Plan the Update for Generalization**: Based on your analysis, decide how to update the experience set to make it more powerful and transferable.
    *   `add`: Introduce a new, broadly applicable principle learned from the attempts.
    *   `modify`: Rewrite an existing experience to be more abstract, clearer, and more general.
    *   `merge`: Combine multiple specific experiences into a single, more powerful and universal one.

3.  **Generate the Final Experience Set**: Apply your update plan to produce the final, updated list of generalizable experiences.

**Final Output Requirements**:
*   The output must be a single JSON object.
*   The object should contain a key `updated_experiences`, which is a list of strings.
*   Each string in the list must be a **generalizable and transferable experience**, not just a specific instruction for the given problem.

<problem>
{problem}
</problem>

<ground_truth_answer>
{answer}
</ground_truth_answer>

<existing_experiences>
{experiences}
</existing_experiences>

<trajectories>
{trajectories}
</trajectories>

Now, provide the final, updated set of experiences for this problem in the following JSON format:
```json
{{
    "updated_experiences": [
        "Updated experience 1...",
        "A newly added experience...",
        "A merged and refined experience...",
        ...
    ]
}}
```
"""

PER_PROBLEM_EXPERIENCE_UPDATE_TEMPLATE2_0 = """You are an expert assistant skilled at distilling generalizable principles from concrete examples.
For a single problem, you will be given existing experiences, several new solution attempts (trajectories), and a high-quality unified response that synthesizes these attempts.

Your task is to **refine the existing experience set by extracting transferable wisdom** from the new trajectories and the unified response. The goal is to produce an improved set of **generalizable experiences**.

Follow these steps:

1.  **Analyze for General Principles**: Review the `<trajectories>` and the `<unified_response>`. The unified response often represents a better general strategy.
    *   **Superior Strategies**: What generalizable reasoning paths in the unified response or successful trajectories should be preserved or added as new principles?
    *   **Common Errors & Pitfalls**: What general mistakes were made in the trajectories that a new or modified experience could prevent?
    *   **Redundancy and Specificity**: Are any existing experiences too specific, redundant, or less effective than the principles demonstrated in the new attempts?

2.  **Plan the Update for Generalization**: Based on your analysis, decide how to update the experience set to make it more powerful and transferable.
    *   `add`: Introduce a new, broadly applicable principle learned from the new attempts.
    *   `modify`: Rewrite an existing experience to be more abstract, clearer, and more general.
    *   `merge`: Combine multiple specific experiences into a single, more powerful and universal one.

3.  **Generate the Final Experience Set**: Apply your update plan to produce the final, updated list of generalizable experiences.

**Final Output Requirements**:
*   The output must be a single JSON object.
*   The object should contain a key `updated_experiences`, which is a list of strings.
*   Each string in the list must be a **generalizable and transferable experience**, not just a specific instruction for the given problem.

<problem>
{problem}
</problem>

<existing_experiences>
{experiences}
</existing_experiences>

<trajectories>
{trajectories}
</trajectories>

<unified_response>
{unified_response}
</unified_response>

Now, provide the final, updated set of experiences for this problem in the following JSON format:
```json
{{
    "updated_experiences": [
        "Updated experience 1...",
        "A newly added experience from the unified response...",
        "A merged and refined experience...",
        ...
    ]
}}
```
"""

PER_PROBLEM_EXPERIENCE_UPDATE_NO_UNIFIED_TEMPLATE_0 = """You are an expert assistant skilled at distilling generalizable principles from concrete examples.
For a single problem, you will be given existing experiences and several new solution attempts (trajectories) without a definitive answer.

Your task is to **refine the existing experience set by extracting transferable wisdom** from the patterns observed across the new trajectories. The goal is to produce an improved set of **generalizable experiences**.

Follow these steps:

1.  **Analyze for General Patterns**: Review the `<trajectories>`. Since there is no ground truth, look for recurring patterns.
    *   **Common Strategies**: What are the common reasoning paths or approaches across multiple trajectories? Abstract these into general strategies.
    *   **Potential Errors**: What are common mistakes or divergent paths that could indicate a general pitfall? Formulate experiences to warn against these.
    *   **Redundancy and Specificity**: Are any existing experiences too specific, redundant, or less effective than the principles inferred from the new attempts?

2.  **Plan the Update for Generalization**: Based on your analysis, decide how to update the experience set to make it more powerful and transferable.
    *   `add`: Introduce a new, broadly applicable principle learned from the observed patterns.
    *   `modify`: Rewrite an existing experience to be more abstract, clearer, and more general.
    *   `merge`: Combine multiple specific experiences into a single, more powerful and universal one.

3.  **Generate the Final Experience Set**: Apply your update plan to produce the final, updated list of generalizable experiences.

**Final Output Requirements**:
*   The output must be a single JSON object.
*   The object should contain a key `updated_experiences`, which is a list of strings.
*   Each string in the list must be a **generalizable and transferable experience**, not just a specific instruction for the given problem.

<problem>
{problem}
</problem>

<existing_experiences>
{experiences}
</existing_experiences>

<trajectories>
{trajectories}
</trajectories>

Now, provide the final, updated set of experiences for this problem in the following JSON format:
```json
{{
    "updated_experiences": [
        "Updated experience 1...",
        "A newly added experience...",
        "A merged and refined experience...",
        ...
    ]
}}
```
"""