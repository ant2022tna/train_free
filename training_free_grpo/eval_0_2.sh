python main.py \
    --mode agent \
    --domain medical \
    --experiment_name medqa400_eval_ds_agent \
    --dataset medqa \
    --dataset_truncate 500 \
    --rollout_concurrency 128 \
    --pass_k 1

#     --experience_file data/web/train/AFM_web_RL_100/step_3/experiences.json \


# python main.py \
#      --mode agent \
#      --domain math \
#      --experiment_name AIME25_eval_ds \
#      --dataset AIME25 \
#      --rollout_concurrency 128 \
#      --pass_k 32