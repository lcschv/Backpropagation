#!/usr/bin/python
import numpy as np
# from input import Input

class NeuralNet(object):
	"""docstring for NeuralNet"""
	outUnit = dict()
	inputUnit = dict()
	hiddenNeurons = dict()
	labels = np.zeros((5000,10), dtype=np.int)
 
	def __init__(self, filein):
		super(NeuralNet, self).__init__()
		self.setOutputUnits()
		self.createInputUnits(filein,1)
		self.createHiddenLayerUnits(25)
		self.readTrainSet(filein)
		self.createOutWeightsMatrix(25)
		self.createWeightMatrix(25)		
		self.calcForwardPass()
		# self.arg = arg
	def setOutputUnits(self):
		self.outUnit = np.identity(10)

	def createInputUnits(self,filein, batchsize):
		# self.input = np.zeros((5,1), dtype=np.float)
		input_file = open(filein,'r')

		# self.input = np.random.randint(9, size=(batchsize, 784))
		self.input = np.empty([1, 784])
		for i in input_file:
			i = i.rstrip()
			vet = i.split(",")
			self.input = map(int, vet[1:])		
			# print self.input

	def createHiddenLayerUnits(self, numNeurons):
		np.zeros((numNeurons,), dtype=np.float)

	def createWeightMatrix(self, numNeurons):
		self.weights = np.random.uniform(-1,1,[784,numNeurons])
		# print self.weights, "\n"

	def createOutWeightsMatrix(self, numNeurons):
		self.z = np.random.uniform(-1,1,[numNeurons,10])
	
	
	### self.y corresponds to the results of the neurons of the hidden layer
	### self.z corresponds to the the prediction of the forward pass, it will result in a 10 positions vector with the respectives probabilities estimated for each digit

	def calcForwardPass(self):
		self.y = np.dot(self.input, self.weights)
		print self.y
		self.y = self.sigmoid(self.y)
		self.z = np.dot(self.y, self.z)
		self.z = self.sigmoid(self.z)
	
	# def sigmoid(self, x):
	# 	print x
	# 	return 1 / (1 + np.exp(-x))

	def sigmoid(self, signal):
	    # Prevent overflow.
	    signal = np.clip( signal, -500, 500 )

	    # Calculate activation signal
	    signal = 1.0/( 1 + np.exp( -signal ))

	    return signal

	
	def dsigmoid(y):
		return y * (1.0 - y)

	def readTrainSet(self, filein):
		input_file = open(filein, 'r')
		cont = 0
		for i in input_file:
			i = i.rstrip()
			vet = i.split(",")
			self.labels[cont][int(vet[0])] = 1
			cont += 1