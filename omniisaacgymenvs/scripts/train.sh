python rlgames_train.py headless=True device_id=7 rl_device=cuda:7 task=UnitreeA1Stand
python rlgames_train.py headless=True task=UnitreeA1Stand

python rlgames_train.py headless=True task=UnitreeA1Stand max_iterations=100000 init_from_prepared_state_data=False push_robots=True
python rlgames_train.py headless=True task=UnitreeA1Stand max_iterations=100000 init_from_prepared_state_data=True push_robots=True

python rlgames_train.py task=UnitreeA1Stand checkpoint=/home/mk/Desktop/code/RL4Robot_omnisim/omniisaacgymenvs/scripts/runs/UnitreeA1Stand/nn/UnitreeA1Stand.pth test=True num_envs=4 init_from_prepared_state_data=False push_robots=True
python rlgames_train.py task=UnitreeA1Stand checkpoint=/home/mk/Desktop/code/RL4Robot_omnisim/omniisaacgymenvs/scripts/runs/UnitreeA1Stand/nn/UnitreeA1Stand.pth test=True num_envs=4 init_from_prepared_state_data=True push_robots=True


python rlgames_train.py task=UnitreeA1Stand checkpoint=/home/mk/Desktop/code/RL4Robot_omnisim/omniisaacgymenvs/scripts/runs/UnitreeA1Stand/nn/UnitreeA1Stand.pth test=True num_envs=4 init_from_prepared_state_data=False push_robots=True dummy_action=True
python rlgames_train.py task=UnitreeA1Stand checkpoint=/home/mk/Desktop/code/RL4Robot_omnisim/omniisaacgymenvs/scripts/runs/UnitreeA1Stand/nn/UnitreeA1Stand.pth test=True num_envs=4 init_from_prepared_state_data=True push_robots=True dummy_action=True


# sample prepared state data
python rlgames_train.py task=UnitreeA1Stand checkpoint=/home/mk/Desktop/code/RL4Robot_omnisim/omniisaacgymenvs/scripts/runs/UnitreeA1Stand/nn/UnitreeA1Stand.pth test=True num_envs=4096 headless=True games_num=100000