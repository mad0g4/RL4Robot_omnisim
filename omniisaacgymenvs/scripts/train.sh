python rlgames_train.py headless=True device_id=7 rl_device=cuda:7 task=UnitreeA1Stand
python rlgames_train.py headless=True task=UnitreeA1Stand

python rlgames_train.py headless=True task=UnitreeA1Stand

python rlgames_train.py task=UnitreeA1Stand checkpoint=runs/Ant/nn/Ant.pth test=True num_envs=64