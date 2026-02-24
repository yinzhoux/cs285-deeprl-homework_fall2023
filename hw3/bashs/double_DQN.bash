#!/bin/bash

log_dir_name=logs-double-DQN

common_args=(

)

rm -rf ${log_dir_name}
mkdir ${log_dir_name}

MAX_JOBS=4
counter=0

python cs285/scripts/run_hw3_dqn.py -cfg experiments/dqn/lunarlander_doubleq.yaml --seed 1 \
       >"${log_dir_name}/lunarlander_doubleq1.txt" 2>&1 &
sleep 2
python cs285/scripts/run_hw3_dqn.py -cfg experiments/dqn/lunarlander_doubleq.yaml --seed 2 \
       >"${log_dir_name}/lunarlander_doubleq2.txt" 2>&1 &
sleep 2
python cs285/scripts/run_hw3_dqn.py -cfg experiments/dqn/lunarlander_doubleq.yaml --seed 3 \
       >"${log_dir_name}/lunarlander_doubleq3.txt" 2>&1 &
sleep 2
python cs285/scripts/run_hw3_dqn.py -cfg experiments/dqn/mspacman.yaml \
       >"${log_dir_name}/mspacman.txt" 2>&1 &