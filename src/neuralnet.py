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
		self.createInputUnits()
		self.createHiddenLayerUnits(25)
		self.readTrainSet(filein)
		
		# self.arg = arg
	def setOutputUnits(self):
		self.outUnit = np.identity(10)

	def createInputUnits(self):
		self.input = np.zeros((784,), dtype=np.float)			

	def createHiddenLayerUnits(self, n):
		np.zeros((n,), dtype=np.float)

	def readTrainSet(self, filein):
		input_file = open(filein, 'r')
		cont = 0
		for i in input_file:
			i = i.rstrip()
			vet = i.split(",")
			print vet[0]
			self.labels[cont][int(vet[0])] = 1
			cont += 1
		
		# print self.labels	