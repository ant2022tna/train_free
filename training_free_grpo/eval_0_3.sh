python main.py \
    --mode agent \
    --domain medical \
    --experiment_name medqa100_eval_ds_step5-re-1 \
    --experience_file /home/nextLabUser/yuhang2/youtu-agent/training_free_grpo/data/medical/train/medqa100_ds_sc_1_tt_all_scref2-pel-7-re/step_5/experiences.json \
    --dataset medqa \
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