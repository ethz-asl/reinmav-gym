from gym.envs.registration import register

register(
    id='reinmav-v0',
    entry_point='gym_reinmav.envs:ReinmavEnv',
)

register(
	id='quadrotor2d-v0',
	entry_point='gym_reinmav.envs:Quadrotor2D',
)