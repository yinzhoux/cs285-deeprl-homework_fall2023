python cs285/scripts/run_hw3_sac.py -cfg experiments/sac/halfcheetah_reinforce1.yaml \
 > logs-sac/actor-reinforce1.txt 2>&1 &
sleep 1
 python cs285/scripts/run_hw3_sac.py -cfg experiments/sac/halfcheetah_reinforce10.yaml \
 > logs-sac/actor-reinforce10.txt 2>&1 &