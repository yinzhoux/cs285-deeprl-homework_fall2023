#!/bin/bash

log_dir_name=logs-basic-q-learning

common_args=(

)

rm -rf ${log_dir_name}
mkdir ${log_dir_name}

MAX_JOBS=4
counter=0

python cs285/scripts/run_hw3_dqn.py -cfg experiments/dqn/cartpole.yaml \
       >"$log_dir_name/cartpole.txt" 2>&1 &
sleep 5
python cs285/scripts/run_hw3_dqn.py -cfg experiments/dqn/lunarlander.yaml --seed 1 \
       >"$log_dir_name/lunarlander1.txt" 2>&1 &
sleep 5
python cs285/scripts/run_hw3_dqn.py -cfg experiments/dqn/lunarlander.yaml --seed 2 \
       >"$log_dir_name/lunarlander2.txt" 2>&1 &
sleep 5
python cs285/scripts/run_hw3_dqn.py -cfg experiments/dqn/lunarlander.yaml --seed 3 \
       >"$log_dir_name/lunarlander3.txt" 2>&1 &
sleep 5
python cs285/scripts/run_hw3_dqn.py -cfg experiments/dqn/cartpole-lr0.05.yaml \
       >"$log_dir_name/cartpole_high_lr.txt" 2>&1 &

