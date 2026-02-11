python cs285/scripts/run_hw1.py \
--expert_policy_file cs285/policies/experts/Ant.pkl \
--env_name Ant-v4 --exp_name bc_ant --n_iter 50 \
--expert_data cs285/expert_data/expert_data_Ant-v4.pkl \
--video_log_freq -1 \
--n_layers 2 --size 64 \
--do_dagger \
--scalar_log_freq 1 \
--eval_batch_size=10000