# python train.py \
#     --mode agent \
#     --domain medical \
#     --experiment_name medqa_ds   \
#     --dataset medqa \
#     --epochs 5 \
# 	--dataset_truncate 100 \
#     --batchsize 100 \
#     --grpo_n 5 \
#     --rollout_concurrency 128 \
#     --rollout_temperature 0.7 \
#     --task_timeout 1800  \

python train_sc.py \
    --mode agent \
    --domain medical \
    --experiment_name MP_health100_ds_sc_5   \
    --dataset MP_health \
    --epochs 6 \
	--dataset_truncate 100 \
    --batchsize 100 \
    --grpo_n 5 \
    --rollout_concurrency 128 \
    --rollout_temperature 0.7 \
    --task_timeout 1800  \
    --rollout_max_tokens 2048 #\
    #--given_ground_truth False


# python train.py \
#     --mode agent \
#     --domain math \
#     --experiment_name DAPO100-ds   \
#     --dataset DAPO-Math-17k \
#     --epochs 3 \
# 	--dataset_truncate 100 \
#     --batchsize 100 \
#     --grpo_n 5 \
#     --rollout_concurrency 128 \
#     --rollout_temperature 0.7 \
#     --task_timeout 1800  \
#     --rollout_max_tokens 8000