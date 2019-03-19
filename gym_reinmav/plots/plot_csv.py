from baselines.common import plot_util as pu
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

log_path='~/workspace/reinmav-gym/gym_reinmav/example/mujoco/log/mujoco/0.0.monitor.csv'
headers = ['r','l','t','rp','rlv','rav','ra','rlive']
data = pd.read_csv(log_path,names=headers,skiprows=1,header=2)
x_axis_unit="step" #"step" or "time"
#print(data.head())
if x_axis_unit=="time":
	x_axis=data.t
else:
	x_axis=np.cumsum(data.l)

plt.figure('Main params')
#plt.plot(np.cumsum(data.l),pu.smooth(data.r, radius=100))
plt.plot(x_axis,pu.smooth(data.r, radius=100))
plt.plot(x_axis,pu.smooth(data.l, radius=100))

plt.legend(['episode mean reward','episode mean length'])
if x_axis_unit=="time":
	plt.xlabel('Time elapsed(s)')
else:
	plt.xlabel('Step')
plt.ylabel('reward')
plt.grid()


plt.figure('Aux params')
plt.plot(x_axis,pu.smooth(data.rp, radius=100))
plt.plot(x_axis,pu.smooth(data.rlv, radius=100))
plt.plot(x_axis,pu.smooth(data.rav, radius=100))
plt.plot(x_axis,pu.smooth(data.ra, radius=100))
plt.plot(x_axis,pu.smooth(data.rlive, radius=100))
plt.legend(['pos','lvel','avel','act','live'])
plt.ylabel('reward')
if x_axis_unit=="time":
	plt.xlabel('Time elapsed(s)')
else:
	plt.xlabel('Step')
plt.grid()
plt.show()

#print(data.shape())
#print(data[0:])
# import matplotlib.pyplot as plt
# import numpy as np

# t = np.arange(0.0, 2.0, 0.01)
# s = 1 + np.sin(2*np.pi*t)
# plt.plot(t, s)

# plt.xlabel('time (s)')
# plt.ylabel('voltage (mV)')
# plt.title('About as simple as it gets, folks')
# plt.grid(True)
# plt.savefig("test.png")
# plt.show()