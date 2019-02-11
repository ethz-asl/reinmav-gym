from gym.envs.registration import register

from gym_reinmav.envs.reinmav_env import ReinmavEnv
from gym_reinmav.envs.quadrotor2d import Quadrotor2D
from gym_reinmav.envs.quadrotor2d_slungload import Quadrotor2DSlungload
from gym_reinmav.envs.quadrotor3d import Quadrotor3D
from gym_reinmav.envs.quadrotor3d_slungload import Quadrotor3DSlungload

register(
    id='MujocoQuadForce-v0',
    entry_point='gym_reinmav.envs.mujoco:MujocoQuadForceEnv',
)
