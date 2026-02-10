net_size=$1
n_layers=$2
n_iter=$3
n_video=$4
eval_batch_size=$5

for object in Ant HalfCheetah Hopper Walker2d
do 
    python cs285/scripts/run_hw1.py \
    --expert_policy_file cs285/policies/experts/${object}.pkl \
    --env_name ${object}-v4 --exp_name bc_${object} --n_iter ${n_iter} \
    --expert_data cs285/expert_data/expert_data_${object}-v4.pkl \
    --video_log_freq ${n_video} \
    --n_layers ${n_layers} --size ${net_size} \
    --eval_batch_size=${eval_batch_size}
done    