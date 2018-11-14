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

import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('TkAgg')

# For testing whether a number is close to zero

_FLOAT_EPS = np.finfo(np.float64).eps
_EPS4 = _FLOAT_EPS * 4.0

class ReinmavEnv(gym.Env):
	"""
    Description:
        A quadrotor environment. The quad is located at the initial pose (i.e., position and orientation), and the goal is to track a pre-defined trajectory.

    Source:
        openai environment template, and cartpole.py example.

    Observation: 
        Type: Box(12) [x, y, z, dx, dy, dz, phi, theta, psi, p, q, r]
        Num	Observation Min    Max  Unit
        0	x position  -10    10   m
        1	y position  -10    10   m
        2	z position  -10    10   m
        3	x velocity  -5      5   m/s
        4	y velocity  -5      5   m/s
        5	z velocity  -5      5   m/s
        6	roll 		-0.7854(-45°) 0.7854(45°) rad
        7	pitch 		-0.7854(-45°) 0.7854(45°) rad
        8	yaw 		-1.5708(-90°) 1.5708(90°) rad
        9	roll-rate	-0.8726(-50°) 0.8726(50°) rad/s
        10	pitch-rate	-0.8726(-50°) 0.8726(50°) rad/s
        11	yaw-rate	-0.5236(-30°) 0.5236(30°) rad/s
    Actions:
        Type: Box(4)
        Num	Action          min     max       unit
        0	force in         0      3.5316     N
        1	moment in x, rolling, -kg m^2
        2	moment in y, pitching, in kg m^2
        3	moment in z, yawing, in kg m^2

    Reward:
        - Reward is 1 for every step taken, including the termination step
		- to do (reward weighting with tracjing error)

    Starting State:
        All observations are assigned a uniform random value between ±0.05

    Episode Termination:
        - at time(t), the Euclidient distance between the current quad position (t,x,y,z) and planned trajectoriy (t,x*,y*,z*), (i.e., tracking error) is more than ±0.2m.
        - some attitude error.
        - Episode length is greater than 200 (update numbers)
        - Solved Requirements
        - Considered solved when the average reward is greater than or equal to 195.0 over 100 consecutive trials (update numbers)
    """

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

		# criteria at which to fail the episode
		self.tracking_error = 0.2 #in meter

		# x,y,z,dx,dy,dz,phi,theta,psi limits
		#limit_state_rpy=np.array([10,10,10,5,5,5,0.7854,0.7854,0.7854*2,0.8726,0.8726,0.5236])
		limit_state_rpy=np.array([1,1,1,0.5,0.5,0.5,0.7854,0.7854,0.7854*2,0.8726,0.8726,0.5236])
		
		self.limit_state_quat=self.QdToState(limit_state_rpy)
		#self.limit_state = np.array([10,10,10,5,5,5,0.7854,0.7854,0.7854*2,0.8726,0.8726,0.5236])

		# force, mx, my,mz
		self.min_action = np.array([0.0,-3*1e-5,-3*1e-5,-6*1e-7]) #Obtained from MATLAB simulation with additional small margins
		self.max_action = np.array([2.0,3*1e-5,3*1e-5,6*1e-7]) #Obtained from MATLAB simulation with additional small margins


		self.action_space = spaces.Box(low=self.min_action,high=self.max_action,dtype=np.float32)
		self.observation_space = spaces.Box(low=-self.limit_state_quat,high=self.limit_state_quat,dtype=np.float32)
		self.seed()

	def myODE(self,action):
		ds=1/3000
		timeint = np.arange(self.t, self.t+self.dt,ds)
		for t in timeint:
			s_t=timer()
			xdot = self.quad_eq_of_motion1(self.state,action,t) #xdot is 1x13 (quaternion form)
			e_t=timer()
			#print("dura={}ms".format((e_t-s_t)*1e3))
			#print("self.state=",self.state)
			self.state = self.state+ds*xdot

	def step(self,action):
		start_t=timer()
		#self.force=action[0]
		#self.moment=action[1:4]
		#print("self.force=",self.force)
		#print("self.moment=",self.moment)
		#Simple Euler integration, note that chooseing dt is important due to linearizing the nonlinearity of EOM. 1/1000 and 1/100 didn't work (i.e., numerical unstable by generating NaNs) but 1/10000 looks good.
		#xdot = self.quad_eq_of_motion1(self.state,self.t)
		#self.state = self.state+self.dt*xdot
		desired_state=self.trj_gen(self.t)
		trj_err = self.calc_trj_err(self.state, desired_state)
		self.myODE(action) #!!!! In this function, we update self.state by integrating (i.e., self.state is the next state). 
		end_t = timer()
		#I can't control the behaviors of odeint (e.g., step-size) and it seems I didn't fully understand how this work under the hood.. so change odeint to a simple Euler integration by sacrificing accuracy...

		#state = odeint(self.quad_eq_of_motion1, self.state, [self.t,self.t+self.dt])#,atol=1.0e-5, rtol=1.0e-5) #takes about 1.6097ms

		#desired_state=self.trj_gen(self.t+self.dt)
		#done = True #bool(self.state[0] >= self.goal_position)
		#reward = 0
		#if done:
		#	reward = 100.0
		#reward-= math.pow(action,2)*0.1
		#reward-= 10.0

		#episode terminal condition
		#print("trj_err=",trj_err)
		done =  trj_err > self.tracking_error
		done = bool(done)

		if not done:
			reward = 1.0
		else:
			reward = 0.0
		

		#Update time
		self.t = self.t+self.dt
		#Store time, state, and desired for plot
		#self.cum_desired_state = np.vstack([self.cum_desired_state,desired_state])
		#self.cum_state = np.vstack([self.cum_state,self.stateToQd(self.state)])
		#self.cum_t.append(self.t)
		#print("step duration= {}ms".format((end_t-start_t)*1e3)) #average 1.2ms
		#return self.state, reward, done, {}
		return np.array(self.state), reward, done, {}

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
		plt.show()

	def quad_eq_of_motion1(self,state,action,time):
		#print("state=",state_rpy)
		cur_state=self.stateToQd(state)
		desired_state=self.trj_gen(time)
		#F,M=self.controller(time,cur_state,desired_state)
		#print("action in quad_eq_of_motion1=",action)
		F=action[0]
		M=action[1:4]
		#print("F=",F)
		#print("M=",M)
		#state_quat=self.QdToState(state_rpy)
		sdot=self.quad_eq_of_motion2(state,time,F,M)
		# Debug state
		# print("cur_state=",cur_state)
		# print("desired_state=",desired_state)
		# print("F=",F)
		# print("M=",M)
		# print("sdot=",sdot)
		return sdot

	def calc_trj_err(self,state,desired):
		state=np.asmatrix(state) #1x12 vector, x,y,z,dx,dy,dz,phi,theta,yaw,p,q,r
		desired_state=np.asmatrix(desired) #1x11 vector, x,y,z,dz,dy,dz,ddx,ddy,ddz,yaw,dyaw
		error_p=desired_state[[0],[0,1,2]]-state[[0],[0,1,2]]
		#print("error_p=",error_p)
		#print("errorp",error_p[[0],[0]])
		ret=math.sqrt(error_p[[0],[0]]**2+error_p[[0],[1]]**2+error_p[[0],[2]]**2)
		#print("ret=",ret)
		return ret

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

	def mat2quat(self,mat):
	    """ Convert Rotation Matrix to Quaternion.  See rotation.py for notes """
	    mat = np.asarray(mat, dtype=np.float64)
	    assert mat.shape[-2:] == (3, 3), "Invalid shape matrix {}".format(mat)

	    Qxx, Qyx, Qzx = mat[..., 0, 0], mat[..., 0, 1], mat[..., 0, 2]
	    Qxy, Qyy, Qzy = mat[..., 1, 0], mat[..., 1, 1], mat[..., 1, 2]
	    Qxz, Qyz, Qzz = mat[..., 2, 0], mat[..., 2, 1], mat[..., 2, 2]
	    # Fill only lower half of symmetric matrix
	    K = np.zeros(mat.shape[:-2] + (4, 4), dtype=np.float64)
	    K[..., 0, 0] = Qxx - Qyy - Qzz
	    K[..., 1, 0] = Qyx + Qxy
	    K[..., 1, 1] = Qyy - Qxx - Qzz
	    K[..., 2, 0] = Qzx + Qxz
	    K[..., 2, 1] = Qzy + Qyz
	    K[..., 2, 2] = Qzz - Qxx - Qyy
	    K[..., 3, 0] = Qyz - Qzy
	    K[..., 3, 1] = Qzx - Qxz
	    K[..., 3, 2] = Qxy - Qyx
	    K[..., 3, 3] = Qxx + Qyy + Qzz
	    K /= 3.0
	    # TODO: vectorize this -- probably could be made faster
	    q = np.empty(K.shape[:-2] + (4,))
	    it = np.nditer(q[..., 0], flags=['multi_index'])
	    while not it.finished:
	        # Use Hermitian eigenvectors, values for speed
	        vals, vecs = np.linalg.eigh(K[it.multi_index])
	        # Select largest eigenvector, reorder to w,x,y,z quaternion
	        q[it.multi_index] = vecs[[3, 0, 1, 2], np.argmax(vals)]
	        # Prefer quaternion with positive w
	        # (q * -1 corresponds to same rotation as q)
	        if q[it.multi_index][0] < 0:
	            q[it.multi_index] *= -1
	        it.iternext()
	    return q

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

	def QdToState(self,Qd):
		# return is 1x13
		# This function converts state (1x13) to quadrotor state (1x12, ZXY euler angle)
		s=np.zeros(13)
		quad=self.mat2quat(self.RPYToRot(Qd[6],Qd[7],Qd[8])) #phi,theta,psi
		s[0]=Qd[0] #x
		s[1]=Qd[1] #y
		s[2]=Qd[2] #z
		s[3]=Qd[3] #dx
		s[4]=Qd[4] #dy
		s[5]=Qd[5] #dz
		s[6]=quad[0] #qw
		s[7]=quad[1] #qx
		s[8]=quad[2] #qy
		s[9]=quad[3] #qz
		s[10]=Qd[9] #p
		s[11]=Qd[10] #q
		s[12]=Qd[11] #r
		return s 

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

	def RotToRPY(self,R): #ZXY order!!!
		R=R.reshape(3,3) #to remove the last dimension i.e., 3,3,1
		phi = math.asin(R[1,2])
		psi = math.atan2(-R[1,0]/math.cos(phi),R[1,1]/math.cos(phi))
		theta = math.atan2(-R[0,2]/math.cos(phi),R[2,2]/math.cos(phi))
		return phi,theta,psi

	def RPYToRot(self,phi,theta,psi):
		#working confirmed by calling stateToQd, QdTostate, and stateToQd
		R = np.matrix( [[math.cos(psi)*math.cos(theta) - math.sin(phi)*math.sin(psi)*math.sin(theta),math.cos(theta)*math.sin(psi) + math.cos(psi)*math.sin(phi)*math.sin(theta),-math.cos(phi)*math.sin(theta)],[-math.cos(phi)*math.sin(psi),math.cos(phi)*math.cos(psi),math.sin(phi)],[math.cos(psi)*math.sin(theta) + math.cos(theta)*math.sin(phi)*math.sin(psi),math.sin(psi)*math.sin(theta) - math.cos(psi)*math.cos(theta)*math.sin(phi),math.cos(phi)*math.cos(theta)]])
		return R

	def reset(self):
		#print("reset called")
		self.state = self.np_random.uniform(low=-self.limit_state_quat,high=self.limit_state_quat, size=(13,)) #12 state, Qd format
		#state_quad=self.QdToState(state)
		return np.array(self.state)

	def render(self, mode='human', close=False):
		print("render() called")

	def seed(self, seed=None):
		self.np_random, seed = seeding.np_random(seed)
		return [seed]
