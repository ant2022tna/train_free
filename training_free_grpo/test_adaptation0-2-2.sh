
python train_sc2.py \
    --mode agent \
    --domain medical \
    --experiment_name MP_health800_200_ds_sc_1_tt_all_scref2-pel   \
    --dataset MP_health \
    --epochs 2 \
	--dataset_truncate 800 \
    --batchsize 200 \
    --grpo_n 5 \
    --rollout_concurrency 128 \
    --rollout_temperature 0.7 \
    --task_timeout 1800  \
    --rollout_max_tokens 2048 \
     --given_ground_truth False \
# --use_trajectory_summary_for_synthesis

