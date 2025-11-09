# python main_sc.py \
#     --mode agent \
#     --domain medical \
#     --experiment_name medqa200_eval_ds_agent_sc_0_5_3 \
#     --dataset medqa \
#     --dataset_truncate 200 \
#     --rollout_concurrency 128 \
#     --pass_k 5

#     --experience_file data/web/train/AFM_web_RL_100/step_3/experiences.json \


python main_sc2.py \
    --mode agent \
    --domain medical \
    --experiment_name MP_health200_eval_7 \
    --dataset MP_health \
    --rollout_concurrency 128 \
    --dataset_truncate 200 \
    --pass_k 5 \
    --rollout_temperature 0

#v     --dataset_truncate 250 \
# python main.py \
#      --mode agent \
#      --domain math \
#      --experiment_name AIME25_eval_ds \
#      --dataset AIME25 \
#      --rollout_concurrency 128 \
#      --pass_k 32