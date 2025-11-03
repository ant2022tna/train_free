python main_sc.py \
    --mode agent \
    --domain medical \
    --experiment_name medqa400_eval_ds_step6_agent_sc \
    --experience_file /home/ubuntu/yuhang2/youtu-agent/training_free_grpo/data/medical/train/medqa_ds_test/step_6/experiences.json \
    --dataset medqa \
    --dataset_truncate 400 \
    --rollout_concurrency 128 \
    --pass_k 5

#     --experience_file data/web/train/AFM_web_RL_100/step_3/experiences.json \


# python main.py \
#      --mode agent \
#      --domain math \
#      --experiment_name AIME25_eval_ds \
#      --dataset AIME25 \
#      --rollout_concurrency 128 \
#      --pass_k 32