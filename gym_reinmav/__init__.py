from gym.envs.registration import register

register(
    id='reinmav-v0',
    entry_point='gym_reinmav.envs:ReinmavEnv',
)

register(
	id='quadrotor2d-v0',
	entry_point='gym_reinmav.envs:Quadrotor2D',
)

register(
	id='quadrotor2d-slungload-v0',
	entry_point='gym_reinmav.envs:Quadrotor2DSlungload',
)

register(
	id='quadrotor3d-v0',
	entry_point='gym_reinmav.envs:Quadrotor3D',
)

register(
	id='quadrotor3d-slungload-v0',
	entry_point='gym_reinmav.envs:Quadrotor3DSlungload',
)
