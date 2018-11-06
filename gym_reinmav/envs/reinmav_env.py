#Copyright (C) 2018, by Inkyu Sa, enddl22@gmail.com
# Adaptation of the MountainCar Environment (continuous control)

#This is free software: you can redistribute it and/or modify
#it under the terms of the GNU Lesser General Public License as published by
#the Free Software Foundation, either version 3 of the License, or
#(at your option) any later version.
 
#This software package is distributed in the hope that it will be useful,
#but WITHOUT ANY WARRANTY; without even the implied warranty of
#MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#GNU Lesser General Public License for more details.

#You should have received a copy of the GNU Leser General Public License.
#If not, see <http://www.gnu.org/licenses/>.


import gym
from gym import error, spaces, utils
from gym.utils import seeding
import math
import numpy as np
from timeit import default_timer as timer
from scipy.integrate import odeint
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('TkAgg')

# For testing whether a number is close to zero
_FLOAT_EPS = np.finfo(np.float64).eps
_EPS4 = _FLOAT_EPS * 4.0

class ReinmavEnv(gym.Env):
	metadata = {'render.modes': ['human']}
	def __init__(self):
		print("__init__ called")
		self.arm_length = 0.0860 #in meter 
		self.mass = 0.1800 # in kg
		self.gravity=9.8100 # in kg/
		self.min_force=0
		self.max_force=3.5316 # in N
		self.Inertia= np.matrix([[0.00025, 0 , 2.55e-06],
							 [0,0.000232,0],
							 [2.55e-06,0,0.0003738]])
		self.invInertia=self.Inertia.getI()
		#print("Inertia=",self.Inertia)
		#print("invInertia=",self.invInertia)

		self.min_action = -100.0
		self.max_action = 100.0 #in N/m, Force
		self.min_position = -100 #in meter
		self.max_position = 100
		self.max_speed = 100 #in m/s
		self.goal_position = 0.45 # was 0.5 in gym, 0.45 in Arnaud de Broissia's version

		self.low_state = np.array([self.min_position, -self.max_speed])
		self.high_state = np.array([self.max_position, self.max_speed])

		self.viewer = None

		self.action_space = spaces.Box(low=self.min_action, high=self.max_action, shape=(1,),dtype=np.float32)
		self.observation_space = spaces.Box(low=self.low_state, high=self.high_state, dtype=np.float32)

		self.m=10 #kg
		self.k=50 #N/m
		self.c=1/2*np.sqrt(self.k*self.m) #Critical damping
		self.c1=self.c/self.m
		self.c2=self.k/self.m
		self.max_force=3.5316 #N
		self.min_force=0

		self.force=[1.765857676074257] #1x1
		self.moment=[-0.00788085922907771,
			-2.52982080208154e-08,
			-0.000112674126155975] #the Inertial moment along x,y,z axis.
		self.init_state=[5.0,0.0,0.0,0.0,0.0,0.0,1.0,0.0,0.0,0.0,0.0,0.0,0.0] #init value for [x, y, z, dx, dy, dz, qw, qx, qy, qz, p, q, r]
		self.t=0
		self.dt=1.0/100.0 #10ms
		self.action=0
		self.state=self.init_state
		self.desired_state=[]

		self.seed()
		self.cum_state=self.state
		#self.reset()
		#print("state1=",self.state)
		#self.quad_eq_of_motion()
		#self.trj_gen()
		#self.stateToQd(self.state)
		self.trj_gen(0.012)
		self.controller()

	def seed(self, seed=None):
		print("seed")
		self.np_random, seed = seeding.np_random(seed)
		return [seed]

	def step(self, action):
		print("step")
		self.action=action
		start = timer()
		state = odeint(self.quad_eq_of_motion, self.state, [0,self.dt]) #takes about 1.6097ms
		end = timer()
		print("step odeint duration={:0.4f}ms".format((end - start)*1e3)) #in us

		#print("ode return state=",state)
		#print("state.shape=",state.shape)
		self.state = state[-1] # We only care about the state at the ''final timestep'', self.dt
		#print("self.state.shape=",self.state.shape)

		self.cum_state = np.vstack([self.cum_state,self.state])
		#position = self.state[0] # We only care about the state at the ''final timestep'', self.dt
		#velocity = self.state[1]
		done = bool(self.state[0] >= self.goal_position)
		reward = 0
		if done:
			reward = 100.0
		reward-= math.pow(action,2)*0.1
		#self.state = np.array([position, velocity])
		#self.state = state
		
		return self.state, reward, done, {}
	def trj_gen(self,t):
		t_max=4
		t = np.maximum(0,np.minimum(t,t_max))
		pos = 10*t**3 - 15*t**4 + 6*t**5;
		vel = (30/t_max)*t**2 - (60/t_max)*t**3 + (30/t_max)*t**4;
		acc = (60/t_max**2)*t - (180/t_max**2)*t**2 + (120/t_max**2)*t**3;
		self.desired_state.append(pos)
		self.desired_state.append(pos)
		self.desired_state.append(pos)
		self.desired_state.append(vel)
		self.desired_state.append(vel)
		self.desired_state.append(vel)
		self.desired_state.append(acc)
		self.desired_state.append(acc)
		self.desired_state.append(acc)
		self.desired_state.append(pos) #yaw
		self.desired_state.append(vel) #yaw rate
		#print("s_desired",s_desired)
		#return s_desired

	def plot_state(self):
		print("plot_state")
		t=np.arange(0.0,len(self.cum_state)*self.dt,self.dt)
		plt.plot(t, self.cum_state)
		plt.title("title")
		plt.xlabel("Time(s)")
		plt.ylabel("y-label")
		plt.legend(["position","velocity"])
		plt.show()
		
	def mass_spring_damping(self,state,t):
		print("mass_spring_damping")
		# 1st order ODE
		ret0 = state[1] #vel
		ret1 = self.action/self.m-self.c1*state[1]-self.c2*state[0] #acc, input is force not accelation
		# return the two state derivatives
		return [ret0, ret1]

	def quad_eq_of_motion(self,state,t):
			"""output the derivative of the state vector"""
			A = np.matrix([ [0.25,0, -0.5/self.arm_length],
				[0.25,0.5/self.arm_length,0.],
				[0.25,0,0.5/self.arm_length],
				[0.25,-0.5/self.arm_length,0]])
			a=np.hstack((self.force,self.moment[:2]))
			T=A*np.asmatrix(np.hstack((self.force,self.moment[:2]))).transpose()
			T_clamped=np.maximum(np.minimum(T,self.max_force/4.0),self.min_force/4.0)
			B = np.matrix([[1.0,1.0,1.0,1.0],
							[0.0,self.arm_length,0.0,-self.arm_length],
							[-self.arm_length,0.0,self.arm_length,0.]])
			self.force = B[[0],:]*T_clamped;
			self.force = np.array(self.force).reshape(-1,).tolist()
			self.moment = np.vstack(  (B[[1,2],:]*np.asmatrix(T_clamped),  self.moment[2]));
			self.moment = np.array(self.moment).reshape(-1,).tolist()
			
			#Assign 13 states
			x = self.state[0]
			y = self.state[1]
			z = self.state[2]
			xdot = self.state[3];
			ydot = self.state[4];
			zdot = self.state[5];
			qW = self.state[6];
			qX = self.state[7];
			qY = self.state[8];
			qZ = self.state[9];
			p = self.state[10];
			q = self.state[11];
			r = self.state[12];

			quat = np.vstack((qW,qX,qY,qZ)); #!! Attention to the order!!
			bRw=self.quat2mat(quat.transpose())
			bRw=bRw.reshape(3,3) #to remove the last dimension i.e., 3,3,1
			wRb = bRw.transpose()
			
			# Acceleration
			accel = 1.0 / self.mass * (wRb * np.matrix([[0],[0],self.force]) - np.matrix([[0],[0],[self.mass * self.gravity]]))
			accel = np.array(accel).reshape(-1,).tolist()
			# Angular velocity
			K_quat = 2.0; #%this enforces the magnitude 1 constraint for the quaternion
			quaterror = 1 - (qW**2 + qX**2 + qY**2 + qZ**2);
			qdot = -1/2*np.matrix([ [0,-p,-q,-r],[p,0,-r,q],[q,r,0,-p],[r,-q,p,0]])*quat + K_quat*quaterror * quat
			qdot = np.array(qdot).reshape(-1,).tolist()
			# % Angular acceleration
			omega = [p,q,r] #np.vstack((p,q,r))
			a_temp =np.array(self.Inertia*np.asmatrix(omega).reshape(3,1)).reshape(-1,).tolist() 
			b_temp =np.matrix(np.cross(omega,a_temp)).reshape(3,1)
			pqrdot = self.invInertia * (self.moment - b_temp);
			pqrdot = np.array(pqrdot).reshape(-1,).tolist()
			sdot=[]
			sdot.append(xdot)
			sdot.append(ydot)
			sdot.append(zdot)
			sdot.append(accel[0])
			sdot.append(accel[1])
			sdot.append(accel[2])
			sdot.append(qdot[0])
			sdot.append(qdot[1])
			sdot.append(qdot[2])
			sdot.append(qdot[3])
			sdot.append(pqrdot[0])
			sdot.append(pqrdot[1])
			sdot.append(pqrdot[2])
			return sdot
			
	#stealed from rotations.py
	def quat2mat(self,quat):
	    """ Convert Quaternion to Rotation matrix.  See rotation.py for notes """
	    quat = np.asarray(quat, dtype=np.float64)
	    assert quat.shape[-1] == 4, "Invalid shape quat {}".format(quat)

	    w, x, y, z = quat[..., 0], quat[..., 1], quat[..., 2], quat[..., 3]
	    Nq = np.sum(quat * quat, axis=-1)
	    s = 2.0 / Nq
	    X, Y, Z = x * s, y * s, z * s
	    wX, wY, wZ = w * X, w * Y, w * Z
	    xX, xY, xZ = x * X, x * Y, x * Z
	    yY, yZ, zZ = y * Y, y * Z, z * Z

	    mat = np.empty(quat.shape[:-1] + (3, 3), dtype=np.float64)
	    mat[..., 0, 0] = 1.0 - (yY + zZ)
	    mat[..., 0, 1] = xY - wZ
	    mat[..., 0, 2] = xZ + wY
	    mat[..., 1, 0] = xY + wZ
	    mat[..., 1, 1] = 1.0 - (xX + zZ)
	    mat[..., 1, 2] = yZ - wX
	    mat[..., 2, 0] = xZ - wY
	    mat[..., 2, 1] = yZ + wX
	    mat[..., 2, 2] = 1.0 - (xX + yY)
	    return np.where((Nq > _FLOAT_EPS)[..., np.newaxis, np.newaxis], mat, np.eye(3))
	def stateToQd(self,s):
		# return is 1x12
		# This function converts the quaternion to ZXY euler angle, no more or less
		qd=[]
		for i in range(6): qd.append(s[i]) #pos,vel
		quat = np.vstack((s[6],s[7],s[8],s[9])); #!! Attention to the order!!, w,x,y,z
		R=self.quat2mat(quat.transpose())
		phi, theta, yaw = self.RotToRPY(R)
		qd.append(phi)
		qd.append(theta)
		qd.append(yaw)
		for i in range(3): qd.append(s[10+i]) #omega
#		qd.append(s[10:13])
		#print("qd=",qd)
		return qd
	def controller(self):
		state=np.asmatrix(self.stateToQd(self.state)) #1x12 vector, x,y,z,dx,dy,dz,phi,theta,yaw,p,q,r
		desired_state=np.asmatrix(self.desired_state) #1x11 vector, x,y,z,dz,dy,dz,ddx,ddy,ddz,yaw,dyaw
		print("desired_state=",desired_state)
		print("desired_state.shape=",desired_state.shape)
		print("state=",state)
		print("state.shape=",state.shape)
		print("state[0:2]=",state[[0],[0,1,2]])
		error_p=desired_state[[0],[0,1,2]]-state[[0],[0,1,2]]
		print("error_p",error_p)
		error_v=desired_state[[0],[3,4,5]]-state[[0],[3,4,5]]
		#error_p=desired_state(0:2)-state[0:2]

		#error_p=np.matrix(self.desired_state[0:2]) - np.matrix(self.state[0:2]) #x,y,z, position
		#error_v=np.matrix(self.desired_state[3:5]) - np.matrix(self.state[3:5]) #dx,dy,dz, velocity
		#print ("error_p",error_p)
		#print ("error_v",error_v)
		kp=np.array([10,10,35]);
		kd=np.array([5,5,22]);
		kp_rot=np.array([100,100,100]);
		kd_rot=np.array([.1,.1,.1]);
		# m=params.mass;
		# g=params.gravity;
		psi_des=desired_state[[0],[9]] #desired yaw
		phi=state[[0],[6]]
		theta=state[[0],[7]]
		psi=state[[0],[8]]
		p=state[[0],[9]]
		q=state[[0],[10]]
		r=state[[0],[11]]

		# #Lecture week3, Control 0002.pdf 3D quadrotor.
		dpsi_des=desired_state[[0],[10]]
		ddr=desired_state[[0],[6,7,8]].transpose()+np.diag(kd)*error_v.transpose()+np.diag(kp)*error_p.transpose()
		print("ddr=",ddr)
		# ddr=des_state.acc+diag(kd)*err_v+diag(kp)*err_p;
		u1=self.mass*(self.gravity+ddr[2])

		phi_des=1/self.gravity*(ddr[0]*math.sin(psi_des)-ddr[1]*math.cos(psi_des))
		theta_des=1/self.gravity*(ddr[0]*math.cos(psi_des)+ddr[1]*math.sin(psi_des))
		print("psi_des=",psi_des)
		print("psi=",psi)
		print("dpsi_des=",dpsi_des)
		print("r=",r)
		mx=kp_rot[0]*(phi_des-phi)-kd_rot[0]*p
		#mx=np.squeeze(mx.reshape(-1))
		#print("mxxx=",np.asarray(mx))

		#print("mx.shape=",mx.shape)
		#print("mx.type=",type(mx))
		my=(kp_rot[1]*(theta_des-theta)-kd_rot[1]*q)
		mz=(kp_rot[2]*(psi_des-psi)+kd_rot[2]*(dpsi_des-r))
		# Moment
		moment= np.concatenate((mx,my,mz))
		#self.moment=self.moment.reshape()
		#print("u2=",u2)
		# # Thrust
		self.force = np.array(u1).reshape(-1,).tolist()
		#M = u2;
		print("force=",self.force)
		print("moment=",moment)
		self.moment=[moment[0,0],moment[1,0],moment[2,0]]



	def RotToRPY(self,R):
		#print("R=",R)
		#print("R.shape=",R.shape)
		R=R.reshape(3,3) #to remove the last dimension i.e., 3,3,1
		phi = math.asin(R[1,2])
		psi = math.atan2(-R[1,0]/math.cos(phi),R[1,1]/math.cos(phi))
		theta = math.atan2(-R[0,2]/math.cos(phi),R[2,2]/math.cos(phi))
		return phi,theta,psi

	def reset(self):
		print("reset")
		#self.state = np.array([self.np_random.uniform(low=-0.6, high=-0.4), 0])
		return np.array(self.state)

	def render(self, mode='human', close=False):
		print("render() called")
