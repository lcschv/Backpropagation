#!/usr/bin/python
import numpy as np

class Input(object):
	"""docstring for Input"""
	def __init__(self):
		super(Input, self).__init__()
		# self.arg = arg
		
	def readTrainSet(self, filein):
		input_file = open(filein, 'r')
		for i in input_file:
			i = i.rstrip()
			vet = i.split(",")
		return vet[0],vet[1:]
