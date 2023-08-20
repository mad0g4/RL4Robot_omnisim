python rlgames_train.py headless=True device_id=7 rl_device=cuda:7 task=UnitreeA1Stand
python rlgames_train.py headless=True task=UnitreeA1Stand

python rlgames_train.py headless=True task=UnitreeA1Stand

python rlgames_train.py task=UnitreeA1Stand checkpoint=/home/mk/Desktop/code/RL4Robot_omnisim/omniisaacgymenvs/scripts/runs/UnitreeA1Stand/nn/UnitreeA1Stand.pth test=True num_envs=4

# sample prepared state data
python rlgames_train.py task=UnitreeA1Stand checkpoint=/home/mk/Desktop/code/RL4Robot_omnisim/omniisaacgymenvs/scripts/runs/UnitreeA1Stand/nn/UnitreeA1Stand.pth test=True num_envs=4096 headless=True games_num=100000