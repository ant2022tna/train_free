python main.py \
    --mode agent \
    --domain medical \
    --experiment_name MP_health_ori_mmlu-clinical_knowledge_nogt_eval-5-2 \
    --dataset MP_health_ori_mmlu-clinical_knowledge \
    --rollout_concurrency 128 \
    --pass_k 5

#     --experience_file data/web/train/AFM_web_RL_100/step_3/experiences.json \