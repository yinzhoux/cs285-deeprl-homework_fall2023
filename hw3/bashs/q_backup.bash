python cs285/scripts/run_hw3_sac.py -cfg experiments/sac/hopper.yaml \
 > logs-sac/hopper.txt 2>&1 &
sleep 1

 python cs285/scripts/run_hw3_sac.py -cfg experiments/sac/hopper_clipq.yaml \
 > logs-sac/hopper_clipq.txt 2>&1 &

sleep 1
 python cs285/scripts/run_hw3_sac.py -cfg experiments/sac/hopper_doubleq.yaml \
 > logs-sac/hopper_doubleq.txt 2>&1 &