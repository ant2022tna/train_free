# python main.py \
#     --mode prompt \
#     --domain medical \
#     --experiment_name MP_health_prompt2_ds67_ollama \
#     --dataset MP_health \
#     --rollout_concurrency 128 \
#     --pass_k 1 
    #--experience_file /home/ubuntu/yuhang2/youtu-agent/training_free_grpo/data/medical/train/MP_health_ds2/step_4/experiences.json
    #/home/ubuntu/yuhang2/youtu-agent/training_free_grpo/data/medical/train/MP_health_ds/step_1/experiences.json
    #/home/ubuntu/yuhang2/youtu-agent/training_free_grpo/data/medical/train/MP_health/step_4/experiences.json

#     --experience_file /home/ubuntu/yuhang2/youtu-agent/training_free_grpo/data/medical/train/MP_health_ori_mmlu-clinical_knowledge_nogt_1/step_3/experiences.json  \
#    --experience_file data/web/train/AFM_web_RL_100/step_3/experiences.json \

# python main.py \
#     --mode agent \
#     --domain medical \
#     --experiment_name medqa_16_eval_ds_step2 \
#     --experience_file /home/ubuntu/yuhang2/youtu-agent/training_free_grpo/data/medical/train/medqa_ds/step_2/experiences.json \
#     --dataset medqa \
#     --dataset_truncate 100 \
#     --rollout_concurrency 128 \
#     --pass_k 16 \
#     --rollout_max_tokens 8000



python main.py \
    --mode prompt \
    --domain medical \
    --experiment_name MP_health_qwen72b \
    --dataset MP_health \
    --rollout_concurrency 128 \
    --pass_k 1 

    # --experience_file /home/ubuntu/yuhang2/youtu-agent/training_free_grpo/data/medical/train/MP_health_ds2_test/step_3/experiences.json \