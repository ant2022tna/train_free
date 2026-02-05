
python train_sc2.py \
    --mode agent \
    --domain medical \
    --experiment_name MP_health100_ds_sc_1_tt_all_scref2-pel-7-re   \
    --dataset MP_health \
    --epochs 6 \
	--dataset_truncate 100 \
    --batchsize 100 \
    --grpo_n 5 \
    --rollout_concurrency 128 \
    --rollout_temperature 0.7 \
    --task_timeout 1800  \
    --rollout_max_tokens 2048 \
    --given_ground_truth False \
# --use_trajectory_summary_for_synthesis

