#OPENAI_LOGDIR=
export OPENAI_LOG_FORMAT=stdout
python -m baselines.run --alg=ppo2 --env=quadrotor2d-v0 --network=mlp --num_timesteps=2e7