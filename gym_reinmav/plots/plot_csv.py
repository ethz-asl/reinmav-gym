import matplotlib
matplotlib.use('Qt4Agg')
from baselines.common import plot_util as pu
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import sys,os

class plotCsv():
	def __init__(self,log_path):
		self.log_path=log_path
		self.headers = ['r','l','t','rp','rlv','rav','ra','rlive']
		self.x_axis_unit="step" #"step" or "time"
		self.radius=100
	def loadData(self):
		self.data = pd.read_csv(self.log_path,names=self.headers,skiprows=1,header=2)
		if self.x_axis_unit=="time":
			self.x_axis=self.data.t
		else:
			self.x_axis=np.cumsum(self.data.l)
	def plotData(self):
		plt.figure('All params')
		plt.plot(self.x_axis,pu.smooth(self.data.r, radius=self.radius))
		plt.plot(self.x_axis,pu.smooth(self.data.l, radius=self.radius))
		plt.plot(self.x_axis,pu.smooth(self.data.rp, radius=self.radius))
		plt.plot(self.x_axis,pu.smooth(self.data.rlv, radius=self.radius))
		plt.plot(self.x_axis,pu.smooth(self.data.rav, radius=self.radius))
		plt.plot(self.x_axis,pu.smooth(self.data.ra, radius=self.radius))
		plt.plot(self.x_axis,pu.smooth(self.data.rlive, radius=self.radius))
		plt.legend(['episode mean reward','episode mean length','pos','lvel','avel','act','live'])
		plt.ylabel('reward')
		if self.x_axis_unit=="time":
			plt.xlabel('Time elapsed(s)')
		else:
			plt.xlabel('Step')
		plt.grid()
		plt.show()

if __name__=="__main__":
	arg_length = len(sys.argv)
	print ("With # args={}".format(arg_length))
	if (arg_length != 2):
		print ("""
		usage: python ./plot_csv.py arg0
		arg0: csv log file path
		""")
	else:
		print("Loading file from {}".format(sys.argv[1]))
		myPlot=plotCsv(sys.argv[1])
		myPlot.loadData()
		myPlot.plotData()