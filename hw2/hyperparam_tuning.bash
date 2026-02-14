common_args="--env_name InvertedPendulum-v4 -n 100 -rtg --use_baseline -na --baseline_gradient_steps 5 --discount 0.99 --n_layers 2 --layer_size 64"

# only try different batch size and learning rate.
for batch_size in 500 1000 2000
do 
    for learning_rate in 0.01 0.02 0.03
    do 
        gae_lambda=0.98
        
        for seed in $(seq 1 5); do
            exp_name="speed_test_b${batch_size}_lr${learning_rate}_gae${gae_lambda}_s${seed}"
            
            echo "Running: $exp_name"
            
            python cs285/scripts/run_hw2.py $common_args \
            --exp_name $exp_name \
            --batch_size $batch_size \
            --learning_rate $learning_rate \
            --baseline_learning_rate $learning_rate \
            --gae_lambda $gae_lambda \
            --seed $seed
        done
    done
done