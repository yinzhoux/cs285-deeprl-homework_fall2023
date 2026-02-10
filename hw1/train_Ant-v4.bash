for batch_size in 100 200 300 400 500 600 700 800 900 1000
do 
    python cs285/scripts/run_hw1.py \
    --expert_policy_file cs285/policies/experts/Hopper.pkl \
    --env_name Hopper-v4 --exp_name bc_Ant --n_iter 1 \
    --train_batch_size ${batch_size} \
    --expert_data cs285/expert_data/expert_data_Hopper-v4.pkl \
    --video_log_freq -1 \
    --n_layers 2 --size 64 \
    --eval_batch_size=100000
done    