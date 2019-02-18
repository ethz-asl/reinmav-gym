""" Quadrotor PID position controller for state space [x, y, z, roll, pitch yaw]. 
Implementation is based on Jeong, Seungho, and Seul Jung, "Position Control of a Quad-Rotor System." Robot Intelligence Technology and Applications 2012. Springer, Berlin, Heidelberg, 2013. 971-981.

Note 
    Attitude state is defined as [z, roll, pitch, yaw] 
    Position state is defined as [x, y] 
"""

import numpy as np


class AttitudeControlGain:
    """Attitude and height PID gain"""
    
    # p gain for z, phi, theta, psi
    kpz = 2.            
    kpphi = 0.1
    kptheta = 0.1
    kppsi = 0.3

    # d gain for z, phi, theta, psi
    kdz = 0.5
    kdphi = 0.1
    kdtheta = 0.1
    kdpsi = 0.1

    # i gain for z, phi, theta, psi
    kiz = 0.01
    kiphi = 0.01
    kitheta = 0.01
    kipsi = 0.01

    @property
    def Kx_p(self):
        """p gain matrix for attitude"""
        return np.diag([self.kpz, self.kpphi, self.kptheta, self.kppsi])

    @property
    def Kx_d(self):
        """d gain matrix for attitude"""
        return np.diag([self.kdz, self.kdphi, self.kdtheta, self.kdpsi])

    @property
    def Kx_i(self):
        """i gain matrix for attitude"""
        return np.diag([self.kiz, self.kiphi, self.kitheta, self.kipsi])


class PositionControlGain:
    """PD control gain for x, y"""
    
    # p gain for x, y
    kpx = 0.6      
    kpy = 0.6

    # d gain for x, y
    kdx = 0.2
    kdy = 0.2

    @property
    def Ks_p(self):
        """p gain matrix for position"""
        return np.diag([self.kpx, self.kpy])

    @property
    def Ks_d(self):
        """d gain matrix for position"""
        return np.diag([self.kdx, self.kdy])

class MotorParam:
    C = 0.1     # constant factor
    L = 0.1     # moment arm (L_arm cos 45)

    a = 0.25
    b = 1 / (4*L)
    c = 1 / (4*C)

    @property
    def C_R(self):
        return np.array([
            [self.a, self.b, -self.b, -self.c],
            [self.a, -self.b, -self.b, self.c],
            [self.a, -self.b, self.b, -self.c],
            [self.a, self.b, self.b, self.c],
        ])


class RpyController:
    def __init__(
        self, dt, mass, 
        gravity=-9.81,
        attitude_control_gain=AttitudeControlGain(), 
        position_control_gain=PositionControlGain(),
        motor_params=MotorParam(),
        ):

        self.dt = dt
        self.mass = mass
        self.gravity = gravity

        # check if the gain matrix is diagonal
        assert self._is_diagonal_matrix(attitude_control_gain.Kx_p)
        assert self._is_diagonal_matrix(attitude_control_gain.Kx_d)
        assert self._is_diagonal_matrix(attitude_control_gain.Kx_i)
        assert self._is_diagonal_matrix(position_control_gain.Ks_d)
        assert self._is_diagonal_matrix(position_control_gain.Ks_p)
        # assert self._is_diagonal_matrix(position_control_gain.Ks_i)

        self.attitude_control_gain = attitude_control_gain
        self.position_control_gain = position_control_gain
        self.motor_params = motor_params

        # buffer for last state
        self.position_error_buff = np.zeros(2)
        self.zrpy_error_buff = np.zeros(4)
        self.zrpy_error_int = np.zeros(4)

    def control(self, position, quat, position_d=[0, 0, 0], yaw_d=0):
        """Controller 
        Args:
            position: current position [x, y, z]
            quat: current attittude quaternion w.r.t reference frame
            position_d: position desired [x, y, z]
            yaw_d: yaw desired scalar
        Return: 
            F: motor input in order of (++, +-, --, -+) 
        """

        assert np.shape(position) == (3,)
        assert np.shape(quat) == (4,)

        # position state
        s = np.array([
            position[0],
            position[1],
        ])

        # attitude state
        rpy = self._quat_to_rpy(quat)

        x = np.array([
            position[2],
            rpy[0],
            rpy[1],
            rpy[2],
        ])

        ################################
        # error

        # position
        es_last = self.position_error_buff
        es = position_d[0:2] - s
        es_dot = (es - es_last) / self.dt  # differentiation
        self.position_error_buff = es

        # position input
        us = np.matmul(self.position_control_gain.Ks_p, es) \
             + np.matmul(self.position_control_gain.Ks_d, es_dot)
        us = np.append(us, 0)

        # attitude
        rotmat_BW = np.linalg.inv(self._quat_to_rotmat(quat))

        x_d = np.array([
            position_d[2],                      # +z
            -np.matmul(rotmat_BW, us)[1],       # -y -> roll,
            np.matmul(rotmat_BW, us)[0],        # +x -> pitch,
            yaw_d,
        ])

        ex_last = self.zrpy_error_buff
        ex = x_d - x
        ex_dot = (ex - ex_last) / self.dt  # differentiation
        self.zrpy_error_int += ex * self.dt  # integration
        self.zrpy_error_buff = ex

        # attitude input
        u = np.matmul(self.attitude_control_gain.Kx_p, ex) \
            + np.matmul(self.attitude_control_gain.Kx_d, ex_dot) \
            + np.matmul(self.attitude_control_gain.Kx_i, self.zrpy_error_int)
        u[0] += - self.mass * self.gravity / (np.cos(rpy[1]) * np.cos(rpy[0]))

        # actuator input
        # +,+
        # +,-
        # -,-
        # -,+
        F = np.matmul(self.motor_params.C_R, u)

        return F

    @staticmethod
    def _is_diagonal_matrix(M):
        return np.all(M == np.diag(np.diagonal(M)))

    @staticmethod
    def _quat_to_rotmat(quat):
        assert np.shape(quat) == (4,)

        rotmat_WB = np.array([
            [1 - 2*(quat[2]**2 + quat[3]**2), 2*(quat[1]*quat[2] - quat[3]*quat[0]), 2*(quat[1]*quat[3] + quat[2]*quat[0])],
            [2 * (quat[1]*quat[2] + quat[3]*quat[0]), 1 - 2*(quat[1]**2 + quat[3]**2), 2*(quat[2]*quat[3] - quat[1]*quat[0])],
            [2 * (quat[1]*quat[3] - quat[2]*quat[0]), 2*(quat[2]*quat[3] + quat[1]*quat[0]), 1 - 2*(quat[1]**2 + quat[2]**2)],
        ])

        return rotmat_WB

    @staticmethod
    def _quat_to_rpy(quat):
        assert np.shape(quat) == (4,)

        roll = np.arctan2(2*(quat[0] * quat[1] + quat[2] * quat[3]), 1 - 2*(quat[1]**2 + quat[2]**2))
        pitch = np.arcsin(2*(quat[0] * quat[2] - quat[3] * quat[1]))
        yaw = np.arctan2(2*(quat[0] * quat[3] + quat[1] * quat[2]), 1 - 2*(quat[2]**2 + quat[3]**2))
        
        return np.array([roll, pitch, yaw])

