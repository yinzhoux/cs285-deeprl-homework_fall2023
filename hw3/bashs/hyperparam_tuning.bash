#!/bin/bash

log_dir_name=logs-hyperparam-tuning

common_args=(

)

rm -rf ${log_dir_name}
mkdir ${log_dir_name}

MAX_JOBS=4
counter=0

python cs285/scripts/run_hw3_dqn.py -cfg experiments/dqn/hyperparameters/lunarlander_doubleq_lr1e-3_default.yaml \
       >"${log_dir_name}/lunarlander_doubleq1e-3.txt" 2>&1 &
sleep 1
python cs285/scripts/run_hw3_dqn.py -cfg experiments/dqn/hyperparameters/lunarlander_doubleq_lr1e-4.yaml \
       >"${log_dir_name}/lunarlander_doubleq1e-4.txt" 2>&1 &
sleep 1
python cs285/scripts/run_hw3_dqn.py -cfg experiments/dqn/hyperparameters/lunarlander_doubleq_lr1e-2.yaml \
       >"${log_dir_name}/lunarlander_doubleq1e-2.txt" 2>&1 &
sleep 1
python cs285/scripts/run_hw3_dqn.py -cfg experiments/dqn/hyperparameters/lunarlander_doubleq_lr5e-3.yaml\
       >"${log_dir_name}/lunarlander_doubleq5e-3.txt" 2>&1 &
