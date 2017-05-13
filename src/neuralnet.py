#!/usr/bin/python
import numpy as np
from sklearn import preprocessing
import math
from scipy.special import expit
import pandas as pd
from sklearn.metrics import log_loss
import time
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
		self.createNet()
		self.BackpropagationAlgorithm()

	def BackpropagationAlgorithm(self):
		self.initial_time = time.time()
		for epoch in xrange(self.ephocs):

			for batch in range(0,len(self.input), self.batch_size):

				### Depending of the batch_size, get a slice of the training set as a Batch
				input_slice = self.input[batch:batch+self.batch_size]

				self.calcForwardPass(input_slice)
				self.calculateError(batch)
				self.updateWeights(input_slice)
			self.calcEpochErr(epoch)	

	### Just for tests..
	def calcEpochErr(self, epoch):
		diff = time.time() - self.initial_time	
		l1_test = expit(np.dot(self.input,self.weights))
		l2_test = expit(np.dot(l1_test,self.weightsOut))
		self.output_file.write(str(diff) +"	"+str(epoch)+"	"+str(log_loss(self.labels,l2_test))+"\n")

	### Calculates the output error and the error of the hidden layer
	def calculateError(self, batch):
		self.derror = self.r_expected[batch:batch+self.batch_size] - self.z
		self.hidder_error = np.dot(self.derror, np.transpose(self.weightsOut)) * self.sigmoid_prime(self.y)

	### Update the weights of the neural network..	
	def updateWeights(self, input_slice):
		self.weightsOut+= self.lr * np.dot(np.transpose(self.y),self.derror)
		self.weights+= self.lr * np.dot(np.transpose(input_slice), self.hidder_error)

	### Create the neural network, process the input-data, add the bias, create the random weight matrix. 
	def createNet(self):
		self.input = preprocessing.scale(self.input)
		bias = np.ones((len(self.input)))
		self.input = np.c_[bias, self.input]
		self.weights = np.random.uniform(-1,1,[len(self.input[0]),self.numNeurons])
		self.weightsOut = np.random.uniform(-1,1,[self.numNeurons,10])
		self.output_file = open(self.fileout, 'w')
		

	### self.y corresponds to the results of the neurons of the hidden layer
	### self.z corresponds to the the prediction of the forward pass, it will result in a 10 positions vector with the respectives probabilities estimated for each digit
	def calcForwardPass(self, X_batch):				
		self.y = expit(np.dot(X_batch,self.weights))
		self.z = expit(np.dot(self.y,self.weightsOut))

	### Calculates the derivate of the sigmoid
	def sigmoid_prime(self,z):
		return z * (1-z)

	### Reads the traning data_set, creates the input matrix, labels_set and r_expected which contains the expected value of
	### the label in form of 10 positions vector..
	def readTrainSet(self):
		MNIST_dataset = pd.read_csv(self.filein,header=None,sep=",").rename(columns={0: "lab"})
		self.labels = MNIST_dataset["lab"]
		self.r_expected = np.zeros((self.labels.shape[0],10), dtype=np.int)
		cont = 0
		for lab in self.labels:
			self.r_expected[cont][int(lab)] = 1
			cont+=1
		self.input = MNIST_dataset[[col for col in MNIST_dataset.columns if col !="lab"]]
