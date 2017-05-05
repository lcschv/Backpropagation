#!/usr/bin/python
import numpy as np
# from input import Input
import math

class NeuralNet(object):
	"""docstring for NeuralNet"""
	
	labels = np.zeros((5000,10), dtype=np.int)

 
	def __init__(self, filein):
		super(NeuralNet, self).__init__()
		self.setOutputUnits()
		self.readTrainSet(filein)
		self.createOutWeightsMatrix(25)
		self.createWeightMatrix(25)
		self.createInputUnits(filein,1)
		
				
		# self.arg = arg
	def setOutputUnits(self):
		self.outUnit = np.identity(10)

	def createInputUnits(self,filein, batchsize):
		# self.input = np.zeros((5,1), dtype=np.float)
		self.lr = 0.5
		# self.input = np.random.randint(9, size=(batchsize, 784))
		self.input = np.empty([1, 784])
		cont = 0
		for x in xrange(1,60):
			input_file = open(filein,'r')
			for i in input_file:
				i = i.rstrip()
				vet = i.split(",")
				self.label = int(vet[0])
				self.input = map(int, vet[1:])		
				# print self.label
				print cont
				cont +=1
				self.calcForwardPass()
				print "GroundTruth" ,self.outUnit[self.label]
				print "Probabilities of Z", self.z
				self.derror = np.subtract(self.outUnit[self.label],self.z)
				print "Error", self.derror
				self.hidden_neurons_error()
				self.updateWeights()
				self.updateOutWeightsMatrix()

	def hidden_neurons_error(self):
		self.neurons_err = np.dot(self.weightOut, self.derror)
		# print "hidden errors", self.neurons_err

	def updateWeights(self):
		# print self.input[0]
		# print "\n"
		for i in range(0,784):
			for j in range(0,self.numNeurons):
				# print "parameters: ",self.weights[i][j], self.lr, self.neurons_err[j], self.dsigmoid(self.input[i])
				self.weights[i][j] = self.weights[i][j] + (self.lr * self.neurons_err[j]) * self.input[j]
				# print "WeightAfter: ",self.weights[i][j]

	def updateOutWeightsMatrix(self):
		for i in range(0,self.numNeurons):
			for j in range(0,10):
				self.weightOut[i][j] = self.weightOut[i][j] + (self.lr * self.derror[j]) * self.y[j]

	def createWeightMatrix(self, numNeurons):
		self.numNeurons = numNeurons
		self.derror = np.empty([10, 1])

		self.weights = np.random.uniform(-4,4,[784,numNeurons])

	def createOutWeightsMatrix(self, numNeurons):
		self.weightOut = np.random.uniform(-4,4,[numNeurons,10])
	
	
	### self.y corresponds to the results of the neurons of the hidden layer
	### self.z corresponds to the the prediction of the forward pass, it will result in a 10 positions vector with the respectives probabilities estimated for each digit
	def calcForwardPass(self):
		self.y = np.dot(self.input, self.weights)
		# print self.y
		self.y = self.sigmoid(self.y)
		# print self.y
		self.z = np.dot(self.y, self.weightOut)
		self.z = self.sigmoid(self.z)
		# print self.z

	def sigmoid(self,x):
		return 1 / (1 + np.exp(-x))	

	# def sigmoid(self, signal):
	#     # Prevent overflow.
	#     signal = np.clip( signal, -500, 500 )
	#     # Calculate activation signal
	#     signal = 1.0/( 1 + np.exp( -signal ))
	#     return signal

	
	
	def sigmoid_prime(z):
		"""Derivative of the sigmoid function."""
		return sigmoid(z)*(1-sigmoid(z))

	def readTrainSet(self, filein):
		input_file = open(filein, 'r')
		cont = 0
		for i in input_file:
			i = i.rstrip()
			vet = i.split(",")
			self.labels[cont][int(vet[0])] = 1
			cont += 1