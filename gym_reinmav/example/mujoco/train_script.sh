#!/bin/bash

if [ "$#" -ne 4 ]
then
	echo "================================================================================================"
    echo "Script for training a model using ppo2 (default) given some parameters.                          "
    echo "Usage : source ./train_script.sh <model save path and name> <log dir path> <# steps> <# env>    "
    echo "e.g., source ./train_script.sh ./model/quad-10M ./log/mylog 1e3 2          "
    echo ""
    echo "Note that save model and log are optional but recommended to validate the trained model after   "
    echo "training and debugging purposes.                                                                "
    echo "trained model after training and debugging purposes                                             "
    echo "================================================================================================"
    #exit 1
elif [ "$#" -eq 4 ]
then
	MODEL_SAVE_PATH=$1
	LOG_PATH=$2
	N_STEP=$3
	N_ENV=$4
	python ./train_hovering.py --save_path=$MODEL_SAVE_PATH --num_timesteps=$N_STEP --num_env=$N_ENV --play=False --env=MujocoQuadQuat-v0 --logdir=$LOG_PATH
fi

