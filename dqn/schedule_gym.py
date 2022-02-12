import os
import sys

os.chdir('C:/Users/Fabian/Documents/Master/LR - IP/Project/shared/dqn')

os.system(
    'python run_gym.py '\
    '--features sigmoid ' \
    '--n-exp 100 ' \
    '--game CartPole-v1 Acrobot-v1 MountainCar-v0 caronhill pendulum ' \
    '--gamma .99 .99 .99 .95 .95 ' \
    '--horizon 500 1000 1000 100 3000 ' \
    '--sampling uniform' 
)

os.system(
    'python run_gym.py '\
    '--features sigmoid ' \
    '--n-exp 100 ' \
    '--game CartPole-v1 Acrobot-v1 MountainCar-v0 caronhill pendulum ' \
    '--gamma .99 .99 .99 .95 .95 ' \
    '--horizon 500 1000 1000 100 3000 ' \
    '--sampling prism'
)