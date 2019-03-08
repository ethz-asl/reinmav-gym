# reinmav-gym
`reinmav-gym` is a gym environment for developing mav controllers using the openai gym framework. The environment composes of two environment `native` which has a built in simulator and `mujoco` which uses the mujoco simulator to train your drone.

<img src="gym_reinmav/resources/native_slungload.gif" width="400" /> <img src="gym_reinmav/resources/mujoco_quad.gif" width="400" />

# Installation
## Requirements

- python3.6 (or 3.7) environment by one of the following 
    - system python 
    - conda 
    - virtualenv  
    - venv 
- [gym](https://github.com/openai/gym.git) 
- vpython
- [baselines](https://github.com/openai/baselines.git)
- matplotlib

Note that the code was tested on Ubuntu 16.04, 18.04 and macOS; but matplotlib has some issues in macOS. Please see [this doc](https://matplotlib.org/faq/osx_framework.html) for more details: we strongly recommend to use conda + pythonw (```conda install python.app```) on macOS.

## for mujoco env (optional)

- mujoco 1.5
- mujoco-py

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

## Installing the package
The package can be installed simply by executing the following command. 
```
cd <path-to-reinmav-gym>
pip install -e .
```

## Check installation
You can check your installation using `pip show`
```
pip show gym-reinmav
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