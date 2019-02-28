import argparse
import os
import tensorflow as tf
import numpy as np

from baselines.common.tf_util import get_session
from baselines.common.vec_env.vec_video_recorder import VecVideoRecorder
from baselines.common.vec_env.vec_frame_stack import VecFrameStack
from baselines.common.cmd_util import make_vec_env, make_env
from baselines import logger
from importlib import import_module
from autolab_core import YamlConfig

from gym_reinmav.envs.mujoco import MujocoQuadHoveringEnv

try:
    from mpi4py import MPI
except ImportError:
    MPI = None


DEFAULT_NET = 'mlp'

#=========================
#   Example command lines
#=========================

#Trining
#python ./train_hovering.py --save_path=~/workspace/myGym/models/quadhover-ppo2-20M --num_timesteps=2e7 --num_env=4 --play False

#Playing
#python ./train_hovering.py --load_path=~/workspace/myGym/models/quadhover-ppo2-20M --num_timesteps=0 --play True

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--num_timesteps', type=float, required=False, default=2e7, help='the number of timestep')
    parser.add_argument('--seed', type=int, required=False, help='random seed')
    parser.add_argument('--network', type=str, required=False, default='mlp', help='network type')
    parser.add_argument('--alg', type=str, required=False, default='ppo2', help='algorithm type')
    parser.add_argument('--save_video_interval', type=int, required=False, default=0, help='Save video every x steps (0 = disabled)')
    parser.add_argument('--save_video_length', type=int, required=False, default=10, help='Length of recorded video. Default 10s')
    parser.add_argument('--num_env', type=int, required=False, default=1, help='number of environment')
    parser.add_argument('--reward_scale', type=float, required=False, default=1., help='reward scale')
    parser.add_argument('--save_path', type=str, required=False, default=None, help='Path to save trained model to')
    parser.add_argument('--load_path', type=str, required=False, default=None, help='Path to load trained model from')
    parser.add_argument('--play', required=False, default=False, help='testing trained model')

    parser.add_argument('--nminibatches', type=int, required=False, default=4, help='number of training minibatches per update')
    parser.add_argument('--noptepochs', type=int, required=False, default=4, help='number of training epochs per update')
    parser.add_argument('--num_hidden', type=int, required=False, default=64, help='number of hidden units')
    parser.add_argument('--num_layers', type=int, required=False, default=2, help='number of layers')
    #There are more parameters that we can set (e.g., activation and learning rate). Please have a look baselines/ppo2/ppo2.py and baselines/common/models.py
    return parser.parse_args()


def get_alg_module(alg, submodule=None):
    submodule = submodule or alg
    try:
        # first try to import the alg module from baselines
        alg_module = import_module('.'.join(['baselines', alg, submodule]))
    except ImportError:
        # then from rl_algs
        alg_module = import_module('.'.join(['rl_' + 'algs', alg, submodule]))

    return alg_module


def get_learn_function(alg):
    return get_alg_module(alg).learn


def get_learn_function_defaults(alg, env_type='mujoco'):
    # todo change
    try:
        alg_defaults = get_alg_module(alg, 'defaults')
        kwargs = getattr(alg_defaults, env_type)()
    except (ImportError, AttributeError):
        kwargs = {}
    return kwargs


def train(args):
    total_timesteps = int(args.num_timesteps)
    seed = args.seed
    if args.load_path is not None:
        load_path = os.path.expanduser(args.load_path)
    else:
        load_path = None

    learn = get_learn_function(args.alg)
    alg_kwargs = get_learn_function_defaults(args.alg)
    # alg_kwargs.update(extra_args)

    env = build_env(args)
    if args.save_video_interval != 0:
        env = VecVideoRecorder(env, os.path.join(logger.Logger.CURRENT.dir, "videos"),
                               record_video_trigger=lambda x: x % args.save_video_interval == 0,
                               video_length=args.save_video_length)

    if args.network:
        alg_kwargs['network'] = args.network
        alg_kwargs['num_hidden'] = args.num_hidden
        alg_kwargs['num_layers'] = args.num_layers
    else:
        if alg_kwargs.get('network') is None:
            alg_kwargs['network'] = args.network

    print('Training {} on {} with arguments \n{}'.format(args.alg, 'Anymal', alg_kwargs))

    model = learn(
        env=env,
        seed=seed,
        total_timesteps=total_timesteps,
        **alg_kwargs,
        load_path=load_path
    )

    return model, env


def build_env(args):

    alg = args.alg
    seed = args.seed

    # tf config
    config = tf.ConfigProto(allow_soft_placement=True,
                            intra_op_parallelism_threads=1,
                            inter_op_parallelism_threads=1)
    config.gpu_options.allow_growth = True
    get_session(config=config)

    flatten_dict_observations = alg not in {'her'}

    env = make_vec_env('MujocoQuadForce-v1',
                       'mujoco',
                       args.num_env or 1,
                       seed,
                       reward_scale=args.reward_scale,
                       flatten_dict_observations=flatten_dict_observations)
    # env = ActionClipWrapper(env)

    # if env_type == 'mujoco':
    #     env = VecNormalize(env)

    return env


def main():
    # tensorboard log
    logdir = os.path.join(os.path.curdir, 'log', 'mujoco')
    logformat = ['tensorboard', 'stdout', 'log']

    args = parse_args()

    if MPI is None or MPI.COMM_WORLD.Get_rank() == 0:
        rank = 0
        logger.configure(dir=logdir, format_strs=logformat)
    else:
        logger.configure(dir=logdir, format_strs=[])
        rank = MPI.COMM_WORLD.Get_rank()
    model, env = train(args)
    env.close()

    if args.save_path is not None and rank == 0:
        save_path = os.path.expanduser(args.save_path)
        model.save(save_path)

    if args.play:
        logger.log("Running trained model")
        env = build_env(args)
        obs = env.reset()

        state = model.initial_state if hasattr(model, 'initial_state') else None
        dones = np.zeros((1,))

        while True:
            if state is not None:
                actions, _, state, _ = model.step(obs, S=state, M=dones)
            else:
                actions, _, _, _ = model.step(obs)

            obs, _, done, _ = env.step(actions)
            env.render()
            done = done.any() if isinstance(done, np.ndarray) else done

            if done:
                obs = env.reset()

        env.close()

    return model


if __name__ == '__main__':
    main()

