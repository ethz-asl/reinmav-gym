#!/bin/bash

if [ "$#" -ne 1 ]
then
	echo "====================================================================="
    echo "This script will test trained model                                  "
    echo "Usage : source test_script.sh <model path>"
    echo "e.g., source test_script.sh ~/workspace/model/quad-10M               "
    echo "====================================================================="
    #exit 1
elif [ "$#" -eq 1 ]
then
	MODEL_PATH=$1
	python ./train_hovering.py --load_path=$MODEL_PATH --num_timesteps=0 --play=True --env=MujocoQuadQuat-v0
fi