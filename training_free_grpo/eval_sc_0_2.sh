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
    --experiment_name MP_health_eval_medqa100-ds \
    --dataset MP_health \
    --rollout_concurrency 128 \
    --pass_k 1 \
    --rollout_temperature 0 \
    --experience_file /home/ubuntu/yuhang2/youtu-agent/training_free_grpo/data/medical/train/MP_health100_ds_sc_1_tt_scref2-pel_all/step_2/experiences.json

#v     --dataset_truncate 250 \
# python main.py \
#      --mode agent \
#      --domain math \
#      --experiment_name AIME25_eval_ds \
#      --dataset AIME25 \
#      --rollout_concurrency 128 \
#      --pass_k 32