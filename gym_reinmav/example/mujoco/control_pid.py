import gym
import numpy as np

from gym_reinmav.envs.mujoco import MujocoQuadEnv


class Trajectory:
    R = 0.5     # trajectory radius
    w = 1.0     # trajectory angular speed (rad/s)


class CtrlParam:

    # attitude
    kpz = 2.
    kpphi = 0.1
    kptheta = 0.1
    kppsi = 0.3

    Kx_p = np.array([
        [kpz, 0, 0, 0],
        [0, kpphi, 0, 0],
        [0, 0, kptheta, 0],
        [0, 0, 0, kppsi],
    ])

    kdz = 0.5
    kdphi = 0.1
    kdtheta = 0.1
    kdpsi = 0.1

    Kx_d = np.array([
        [kdz, 0, 0, 0],
        [0, kdphi, 0, 0],
        [0, 0, kdtheta, 0],
        [0, 0, 0, kdpsi],
    ])

    kiz = 0.01
    kiphi = 0.01
    kitheta = 0.01
    kipsi = 0.01

    Kx_i = np.array([
        [kiz, 0, 0, 0],
        [0, kiphi, 0, 0],
        [0, 0, kitheta, 0],
        [0, 0, 0, kipsi],
    ])

    # position control matrix
    kpx = 0.6
    kpy = 0.6

    Ks_p = np.array([
        [kpx, 0],
        [0, kpy],
    ])

    kdx = 0.2
    kdy = 0.2

    Ks_d = np.array([
        [kdx, 0],
        [0, kdy],
    ])


class MotorParam:
    C = 0.1     # constant factor
    L = 0.1     # moment arm (L_arm cos 45)

    a = 0.25
    b = 1 / (4*L)
    c = 1 / (4*C)

    C_R = np.array([
        [a, b, -b, -c],
        [a, -b, -b, c],
        [a, -b, b, -c],
        [a, b, b, c],
    ])


def main():
    dt = 0.01
    mass = 0.3
    gravity = 9.81

    ex = 0
    es = 0
    ex_int = 0

    env = gym.make('MujocoQuadForce-v0')

    # [x, y, z, q0, q1, q2, q3]
    observation = env.reset()
    for t in range(1000):

        env.render()
        #################################
        # desired position state (x, y, z)
        # circle trajectory on 1 m height
        s_d = np.array([
            Trajectory.R * np.cos(Trajectory.w * dt * t),
            Trajectory.R * np.sin(Trajectory.w * dt * t),
            1.0
        ])

        ################################
        # quat -> rpy
        quat = observation[3:]
        rotmat_WB = np.array([
            [1 - 2*(quat[2]**2 + quat[3]**2), 2*(quat[1]*quat[2] - quat[3]*quat[0]), 2*(quat[1]*quat[3] + quat[2]*quat[0])],
            [2 * (quat[1]*quat[2] + quat[3]*quat[0]), 1 - 2*(quat[1]**2 + quat[3]**2), 2*(quat[2]*quat[3] - quat[1]*quat[0])],
            [2 * (quat[1]*quat[3] - quat[2]*quat[0]), 2*(quat[2]*quat[3] + quat[1]*quat[0]), 1 - 2*(quat[1]**2 + quat[2]**2)],
        ])

        roll = np.arctan2(2*(quat[0] * quat[1] + quat[2] * quat[3]), 1 - 2*(quat[1]**2 + quat[2]**2))
        pitch = np.arcsin(2*(quat[0] * quat[2] - quat[3] * quat[1]))
        yaw = np.arctan2(2*(quat[0] * quat[3] + quat[1] * quat[2]), 1 - 2*(quat[2]**2 + quat[3]**2))

        ################################
        # state

        # position
        s = np.array([
            observation[0],
            observation[1],
        ])

        # attitude
        x = np.array([
            observation[2],
            roll,
            pitch,
            yaw,
        ])

        ################################
        # error

        # position
        es_last = es
        es = s_d[0:2] - s
        es_dot = (es - es_last) / dt  # differentiation

        # position input
        us = np.matmul(CtrlParam.Ks_p, es) \
             + np.matmul(CtrlParam.Ks_d, es_dot)
        us = np.append(us, 0)

        # attitude
        rotmat_BW = np.linalg.inv(rotmat_WB)

        x_d = np.array([
            s_d[2],  # +z
            -np.matmul(rotmat_BW, us)[1],  # -y -> roll,
            np.matmul(rotmat_BW, us)[0],  # +x -> pitch,
            (Trajectory.w * dt * t + np.pi) % (2 * np.pi) - np.pi,
        ])

        ex_last = ex
        ex = x_d - x
        ex_dot = (ex - ex_last) / dt  # differentiation
        ex_int += ex * dt  # integration

        # attitude input
        u = np.matmul(CtrlParam.Kx_p, ex) \
            + np.matmul(CtrlParam.Kx_d, ex_dot) \
            + np.matmul(CtrlParam.Kx_i, ex_int)
        u[0] += mass * gravity / (np.cos(pitch) * np.cos(roll))

        # actuator input
        # +,+
        # +,-
        # -,-
        # -,+
        F = np.matmul(MotorParam.C_R, u)

        observation, reward, done, info = env.step(F)

        if done:
            break

if __name__ == "__main__":
    main()
