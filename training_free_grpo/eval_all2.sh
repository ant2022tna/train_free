python main.py \
    --mode agent \
    --domain medical \
    --experiment_name NEJMQA_general_surgery_gpt4o_ture \
    --dataset NEJMQA_general_surgery \
    --rollout_concurrency 128 \
    --pass_k 1  

python main.py \
    --mode agent \
    --domain medical \
    --experiment_name NEJMQA_internal_medicine_gpt4o_ture \
    --dataset NEJMQA_internal_medicine \
    --rollout_concurrency 128 \
    --pass_k 1  

python main.py \
    --mode agent \
    --domain medical \
    --experiment_name NEJMQA_obgyn_gpt4o_ture \
    --dataset NEJMQA_obgyn \
    --rollout_concurrency 128 \
    --pass_k 1  

python main.py \
    --mode agent \
    --domain medical \
    --experiment_name NEJMQA_pediatrics_gpt4o_ture \
    --dataset NEJMQA_pediatrics \
    --rollout_concurrency 128 \
    --pass_k 1  

python main.py \
    --mode agent \
    --domain medical \
    --experiment_name NEJMQA_psychiatry_gpt4o_ture \
    --dataset NEJMQA_psychiatry \
    --rollout_concurrency 128 \
    --pass_k 1  
