python main_sc3.py \
    --mode agent \
    --domain medical \
    --experiment_name eval_MP_health200_ds_sc_1_tt_conf6_scref2-pel_noself_rag_abs_abs_8_all——step3_skip \
    --dataset MP_health \
    --retrieval_dataset MP_health \
    --rollout_concurrency 128 \
    --pass_k 1 \
    --rollout_temperature 0  \
    --experience_file /home/ubuntu/yuhang2/youtu-agent/training_free_grpo/data/medical/train/MP_health200_ds_sc_1_tt_conf6_scref2-pel_noself_rag_abs_abs_8_all/step_3/per_problem_experiences.json \
    --skip_identical_problems_for_retrieval \

#     --dataset_truncate 250 \
#     --experience_file data/web/train/AFM_web_RL_100/step_3/experiences.json \


# python main.py \
#      --mode agent \
#      --domain math \
#      --experiment_name AIME25_eval_ds \
#      --dataset AIME25 \
#      --rollout_concurrency 128 \
#      --pass_k 32

# --experience_file /home/ubuntu/yuhang2/youtu-agent/training_free_grpo/data/medical/train/MP_health_ds2/step_4/experiences.json \

# /home/ubuntu/yuhang2/youtu-agent/training_free_grpo/data/medical/train/MP_health100_ds_sc_1_tt_conf5_scref2-pel_all/step_5/experiences.json