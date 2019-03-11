# **********************************************************************
#
# Copyright (c) 2018, Autonomous Systems Lab
# Author: Inkyu Sa <enddl22@gmail.com>
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions
# are met:
#
# 1. Redistributions of source code must retain the above copyright
#    notice, this list of conditions and the following disclaimer.
# 2. Redistributions in binary form must reproduce the above copyright
#    notice, this list of conditions and the following disclaimer in
#    the documentation and/or other materials provided with the
#    distribution.
# 3. Neither the name PX4 nor the names of its contributors may be
#    used to endorse or promote products derived from this software
#    without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
# "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
# LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
# FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
# COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
# INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
# BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS
# OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED
# AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
# LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
# ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.
#
# *************************************************************************
import gym
from gym import error, spaces, utils
from gym.utils import seeding
import math
import numpy as np
from timeit import default_timer as timer
#from scipy.integrate import odeint
#from scikits.odes.odeint import odeint

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
		self.gravity=9.8100 # in m/s^2=N/kg
		self.min_force=0.0
		self.max_force=3.5316 # in N
		self.Inertia= np.matrix([[0.00025, 0 , 2.55e-06],
							 [0,0.000232,0],
							 [2.55e-06,0,0.0003738]]) #moment of inertial, kg m^2
		self.invInertia=self.Inertia.getI()

		# This will be used when implementing reinforcement controllers.
		# self.low_state = np.array([self.min_position, -self.max_speed])
		# self.high_state = np.array([self.max_position, self.max_speed])
		# self.viewer = None
		# self.action_space = spaces.Box(low=self.min_action, high=self.max_action, shape=(1,),dtype=np.float32)
		# self.observation_space = spaces.Box(low=self.low_state, high=self.high_state, dtype=np.float32)

		self.t=0.0
		self.dt=1/100 #1.0/5000 #10ms
		#self.dt=1.0/5000 #10ms
		
		# self.action=0
		#self.init_state=[0.01, 0.01, 0.01, 0.05, 0.05, 0.05, 0.8, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0] #init value for [x, y, z, dx, dy, dz, qw, qx, qy, qz, p, q, r]
		self.init_state=[0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0] #init value for [x, y, z, dx, dy, dz, qw, qx, qy, qz, p, q, r]
		
		self.state=self.init_state
		self.cum_state=self.stateToQd(self.state)
		#self.seed()
		self.cum_desired_state=[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0] #[x,y,z,dx,dy,dz,ddx,ddy,ddz,yaw,dyaw]
		self.cum_t=[0.0]

	def seed(self, seed=None):
		print("seed")
		self.np_random, seed = seeding.np_random(seed)
		return [seed]
	def myODE(self):
		ds=1/5000
		timeint = np.arange(self.t, self.t+self.dt,ds)
		for t in timeint:
			s_t=timer()
			xdot = self.quad_eq_of_motion1(self.state,t)
			e_t=timer()
			#print("dura={}ms".format((e_t-s_t)*1e3))
			self.state = self.state+ds*xdot
	def step(self):
		start_t=timer()
		#Simple Euler integration, note that chooseing dt is important due to linearizing the nonlinearity of EOM. 1/1000 and 1/100 didn't work (i.e., numerical unstable by generating NaNs) but 1/10000 looks good.
		#xdot = self.quad_eq_of_motion1(self.state,self.t)
		#self.state = self.state+self.dt*xdot
		self.myODE()
		end_t = timer()
		#I can't control the behaviors of odeint (e.g., step-size) and it seems I didn't fully understand how this work under the hood.. so change odeint to a simple Euler integration by sacrificing accuracy...

		#state = odeint(self.quad_eq_of_motion1, self.state, [self.t,self.t+self.dt])#,atol=1.0e-5, rtol=1.0e-5) #takes about 1.6097ms

		desired_state=self.trj_gen(self.t+self.dt)
		done = True #bool(self.state[0] >= self.goal_position)
		reward = 0
		if done:
			reward = 100.0
		#reward-= math.pow(action,2)*0.1
		reward-= 10.0

		#Update time
		self.t = self.t+self.dt
		
		#Store time, state, and desired for plot
		self.cum_desired_state = np.vstack([self.cum_desired_state,desired_state])
		self.cum_state = np.vstack([self.cum_state,self.stateToQd(self.state)])
		self.cum_t.append(self.t)
		#print("step duration= {}ms".format((end_t-start_t)*1e3)) #average 1.2ms
		return self.state, reward, done, {}

	def trj_gen(self,t):
		t_max=4.0
		t = np.maximum(0,np.minimum(t,t_max))
		t = t/t_max
		pos = 10.0*t**3 - 15.0*t**4 + 6.0*t**5;
		vel = (30/t_max)*t**2 - (60/t_max)*t**3 + (30/t_max)*t**4;
		acc = (60/t_max**2)*t - (180/t_max**2)*t**2 + (120/t_max**2)*t**3;
		#self.desired_state=[pos,pos,pos,vel,vel,vel,acc,acc,acc,pos,vel]
		return [pos,pos,pos,vel,vel,vel,acc,acc,acc,pos,vel]

	def plot_state(self):
		from mpl_toolkits.mplot3d import axes3d, Axes3D

		fig1=plt.figure(1)
		print("plot_state")
		#t=np.arange(0.0,len(self.cum_state)*self.dt,self.dt)
		plt.plot(self.cum_t, self.cum_state[:,0],'b',self.cum_t, self.cum_desired_state[:,0],'r-.')
		#plt.plot(self.cum_t, self.cum_desired_state[:,0])
		plt.title("title")
		plt.xlabel("Time(s)")
		plt.ylabel("m")
		plt.legend(["position x","desired x"])
		plt.grid(True)
		fig1.savefig("position_plot.pdf",format='pdf')

		fig2=plt.figure(2)
		plt.plot(self.cum_t, self.cum_state[:,3],'b',self.cum_t, self.cum_desired_state[:,3],'r-.')
		#plt.plot(self.cum_t, self.cum_desired_state[:,0])
		plt.title("title")
		plt.xlabel("Time(s)")
		plt.ylabel("m/s")
		plt.legend(["velocity x","desired vel x"])
		plt.grid(True)
		fig2.savefig("velocity_plot.pdf",format='pdf')

		fig3=plt.figure(3)
		plt.plot(self.cum_t, self.cum_state[:,8],'b',self.cum_t, self.cum_desired_state[:,9],'r-.')
		#plt.plot(self.cum_t, self.cum_desired_state[:,0])
		plt.title("title")
		plt.xlabel("Time(s)")
		plt.ylabel("rad")
		plt.legend(["yaw x","desired yaw"])
		plt.grid(True)
		fig3.savefig("yaw_plot.pdf",format='pdf')

		fig1=plt.figure(4)
		ax = Axes3D(fig1)
		print("plot_state")
		#t=np.arange(0.0,len(self.cum_state)*self.dt,self.dt)
		plt.plot(self.cum_t, self.cum_state[:,0],'b',self.cum_t, self.cum_desired_state[:,0],'r-.')
		#plt.plot(self.cum_t, self.cum_desired_state[:,0])
		plt.title("title")
		plt.xlabel("Time(s)")
		plt.ylabel("m")
		plt.legend(["position x","desired x"])
		plt.grid(True)
		fig1.savefig("3dposition_plot.pdf",format='pdf')
		plt.show()


	def quad_eq_of_motion1(self,state,time):
		cur_state=self.stateToQd(state)
		desired_state=self.trj_gen(time)
		F,M=self.controller(time,cur_state,desired_state)
		sdot=self.quad_eq_of_motion2(state,time,F,M)
		# Debug state
		# print("cur_state=",cur_state)
		# print("desired_state=",desired_state)
		# print("F=",F)
		# print("M=",M)
		# print("sdot=",sdot)
		return sdot

	def quad_eq_of_motion2(self,state,time,force,moment):
			"""output the derivative of the state vector"""

			A = np.matrix([ [0.25,0, -0.5/self.arm_length],
				[0.25,0.5/self.arm_length,0.],
				[0.25,0,0.5/self.arm_length],
				[0.25,-0.5/self.arm_length,0]])
			T=A*np.asmatrix(np.hstack((force,moment[:2]))).transpose()
			T_clamped=np.maximum(np.minimum(T,self.max_force/4.0),self.min_force/4.0)
			B = np.matrix([[1.0,1.0,1.0,1.0],
							[0.0,self.arm_length,0.0,-self.arm_length],
							[-self.arm_length,0.0,self.arm_length,0.]])
			force = B[[0],:]*T_clamped;
			force = np.array(force).reshape(-1,).tolist()
			moment = np.vstack(  (B[[1,2],:]*np.asmatrix(T_clamped),  moment[2]));
			moment = np.array(moment).reshape(-1,).tolist()
			
			#Assign 13 states
			#x = state[0]
			#y = state[1]
			#z = state[2]
			xdot = state[3];
			ydot = state[4];
			zdot = state[5];
			qW = state[6];
			qX = state[7];
			qY = state[8];
			qZ = state[9];
			p = state[10];
			q = state[11];
			r = state[12];

			quat = np.vstack((qW,qX,qY,qZ)); #!! Attention to the order!!
			bRw=self.quat2mat(quat.transpose())
			bRw=bRw.reshape(3,3) #to remove the last dimension i.e., 3,3,1
			wRb = bRw.transpose()
			
			# Acceleration
			accel = 1.0 / self.mass * (wRb * np.matrix([[0],[0],force]) - np.matrix([[0],[0],[self.mass * self.gravity]]))
			accel = np.array(accel).reshape(-1,).tolist()
			# Angular velocity
			K_quat = 2.0; #%this enforces the magnitude 1 constraint for the quaternion
			quaterror = 1 - (qW**2 + qX**2 + qY**2 + qZ**2);
			qdot = -1/2*np.matrix([ [0,-p,-q,-r],[p,0,-r,q],[q,r,0,-p],[r,-q,p,0]])*quat + K_quat*quaterror * quat
			qdot = np.array(qdot).reshape(-1,).tolist()
			# % Angular acceleration
			omega = np.matrix([[p],[q],[r]])
			temp = np.squeeze(np.cross(omega.transpose(),(self.Inertia*omega).transpose()))
			pqrdot  = self.invInertia * (moment - temp).reshape(-1,1)
			sdot=np.zeros(13) #default=float64
			sdot[0]=xdot#[]
			sdot[1]=ydot
			sdot[2]=zdot
			sdot[3]=accel[0]
			sdot[4]=accel[1]
			sdot[5]=accel[2]
			sdot[6]=qdot[0]
			sdot[7]=qdot[1]
			sdot[8]=qdot[2]
			sdot[9]=qdot[3]
			sdot[10]=pqrdot[0]
			sdot[11]=pqrdot[1]
			sdot[12]=pqrdot[2]
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
		qd=np.zeros(12)
		for i in range(6): qd[i]=s[i] #pos,vel
		quat = np.vstack((s[6],s[7],s[8],s[9])); #!! Attention to the order!!, w,x,y,z
		R=self.quat2mat(quat.transpose())
		phi, theta, yaw = self.RotToRPY(R)
		qd[6]=phi
		qd[7]=theta
		qd[8]=yaw
		for i in range(3): qd[9+i]=s[10+i] #omega
		return qd

	def controller(self,time,cur_state,desired_state):
		state=np.asmatrix(cur_state) #1x12 vector, x,y,z,dx,dy,dz,phi,theta,yaw,p,q,r
		desired_state=np.asmatrix(desired_state) #1x11 vector, x,y,z,dz,dy,dz,ddx,ddy,ddz,yaw,dyaw
		
		error_p=desired_state[[0],[0,1,2]]-state[[0],[0,1,2]]
		error_v=desired_state[[0],[3,4,5]]-state[[0],[3,4,5]]
		kp=np.array([10,10,35]);
		kd=np.array([5,5,22]);
		kp_rot=np.array([100,100,100]);
		kd_rot=np.array([.1,.1,.1]);
		psi_des=desired_state[[0],[9]] #desired yaw
		phi=state[[0],[6]]
		theta=state[[0],[7]]
		psi=state[[0],[8]]
		p=state[[0],[9]]
		q=state[[0],[10]]
		r=state[[0],[11]]
		dpsi_des=desired_state[[0],[10]]
		ddr=desired_state[[0],[6,7,8]].transpose()+np.diag(kd)*error_v.transpose()+np.diag(kp)*error_p.transpose()
		u1=self.mass*(self.gravity+ddr[2])

		phi_des=1/self.gravity*(ddr[0]*math.sin(psi_des)-ddr[1]*math.cos(psi_des))
		theta_des=1/self.gravity*(ddr[0]*math.cos(psi_des)+ddr[1]*math.sin(psi_des))
		mx=kp_rot[0]*(phi_des-phi)-kd_rot[0]*p
		my=(kp_rot[1]*(theta_des-theta)-kd_rot[1]*q)
		mz=(kp_rot[2]*(psi_des-psi)+kd_rot[2]*(dpsi_des-r))
		# Moment
		moment= np.concatenate((mx,my,mz))
		# # Thrust
		force = np.array(u1).reshape(-1,).tolist()
		moment=[moment[0,0],moment[1,0],moment[2,0]]
		return force,moment



	def RotToRPY(self,R):
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
