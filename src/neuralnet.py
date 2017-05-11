#!/usr/bin/python
import numpy as np
import math
from sklearn.preprocessing import StandardScaler
from scipy.special import expit
import pandas as pd
from sklearn.metrics import log_loss

class NeuralNet(object):

	def __init__(self, filein, fileout, batch_size, learning_rate, num_neurons, ephocs):
		super(NeuralNet, self).__init__()
		self.filein = filein
		self.fileout = fileout
		self.batch_size = batch_size
		self.lr = learning_rate
		self.numNeurons = num_neurons
		self.ephocs = ephocs

		self.readTrainSet()
		self.createWeightMatrix()
		self.BackpropagationAlgorithm()

	def BackpropagationAlgorithm(self):

		output_file = open(self.fileout, 'w')
		cont = 0
		for epoch in xrange(self.ephocs):

			for batch in range(0,self.input.shape[0], self.batch_size):
				input_slice = self.input[batch:batch+self.batch_size]

				self.calcForwardPass(input_slice)
				self.calculateError(batch)
				self.updateWeights(input_slice)
			self.calcEpochErr(epoch)	

	def calcEpochErr(self, epoch):
		l1_test = expit(np.dot(self.input,self.weights))
		l2_test = expit(np.dot(l1_test,self.weightsOut))
		print("Cross Entropy empirical error at epoch "+str(epoch)+": "+str(log_loss(self.labels,l2_test)))

	def calculateError(self, batch):
		self.derror = self.r_expected[batch:batch+self.batch_size] - self.l2
		# self.l2_delta = self.derror 		
		self.hidder_error = self.derror.dot(self.weightsOut.T) * self.sigmoid_prime(self.l1)

	def updateWeights(self, input_slice):
		self.weightsOut+= self.lr * (self.l1.T.dot(self.derror))
		self.weights+= self.lr * (input_slice.T.dot(self.hidder_error))

	def createWeightMatrix(self):
		self.weights = np.random.uniform(-1,1,[self.input.shape[1],self.numNeurons])
		self.weightsOut = np.random.uniform(-1,1,[self.numNeurons,10])
	
	### self.y corresponds to the results of the neurons of the hidden layer
	### self.z corresponds to the the prediction of the forward pass, it will result in a 10 positions vector with the respectives probabilities estimated for each digit
	
	def calcForwardPass(self, X_batch):				
		self.l1 = expit(np.dot(X_batch,self.weights))
		self.l2 = expit(np.dot(self.l1,self.weightsOut))

	def sigmoid_prime(self,z):
		return z * (1-z)

	def readTrainSet(self):
		MNIST_dataset = pd.read_csv(self.filein,header=None,sep=",").rename(columns={0: "label"})
		cont = 0
		self.labels = MNIST_dataset["label"]
		self.r_expected = np.zeros((self.labels.shape[0],10), dtype=np.int)
		for lab in self.labels:
			self.r_expected[cont][int(lab)] = 1
			cont+=1
		self.input = MNIST_dataset[[col for col in MNIST_dataset.columns if col !="label"]]
		self.input = StandardScaler().fit_transform(self.input) #scaling features
		self.input = np.c_[np.ones(self.input.shape[0]), self.input] #adding biases