# Assuming that you already installed openai gym (https://github.com/openai/gym)

# reinmav-gym
openai gym environment for reinforcement quadrotor

# Requirements
python3.6 (or above) and vpython

## for mujoco env 

- conda (strongly recommended)
- gym with every environment
- mujoco 1.5

1. put mjpro150 directory into ~/.mujoco
2. put mjkey.txt into ~/.mujoco
3. install apt dependencies
    - see gym README.md
4. export LD_LIBRARY_PATH
```
$ export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$HOME/.mujoco/mjpro150/bin
$ # check your nvidia driver version 
$ export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/lib/nvidia-410 
```
5. install gym by pip3 install 'gym[all]'

# Installation
(venv) Username@reinmav-gym $ pip install -e .


## Check installation
pip list

pip show gym-reinmav

``` pip show gym-reinmav
Name: gym-reinmav
Version: 0.0.1
Summary: UNKNOWN
Home-page: UNKNOWN
Author: UNKNOWN
Author-email: UNKNOWN
License: UNKNOWN
Location: /Users/YOUR_INSTALLED_PATH/openai/venv/lib/python3.6/site-packages
Requires: gym
Required-by: 
```

# Testing
Executing the following command should generate 4 plots (3D position, 1D position, velocity and yaw plots).

``` $python ./test/test_reinmav.py ```

![3D plot](http://drive.google.com/uc?export=view&id=1tiTP0UBm1NjB1Wpm53m2ThZQsTZ8N9cy)


# openai gym example environments
* Continuous mountain car: https://github.com/openai/gym/blob/master/gym/envs/classic_control/continuous_mountain_car.py
* Cart pole: https://github.com/openai/gym/blob/master/gym/envs/classic_control/cartpole.py
* Pendulum: https://github.com/openai/gym/blob/master/gym/envs/classic_control/pendulum.py

# Integration into openai gym environment
(if you want to use this reinmav environment with Keras-rl(https://github.com/keras-rl/keras-rl)..)
1. copy reinmav-gym/gym_reinmav/envs/reinmav_env.py to gym/gym/envs/classic_control/
2. add ``` from gym.envs.classic_control.reinmav_env import ReinmavEnv ``` to ```gym/gym/envs/classic_control/__init__.py``` @ line 6
3. add the below to ```gym/gym/envs/__init__.py``` @ line 92-95
```
register(
    id='reinmav-v0',
    entry_point='gym.envs.classic_control:ReinmavEnv',
)
```
