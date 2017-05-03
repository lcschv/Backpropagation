#!/usr/bin/python

class Input(object):
	"""docstring for Input"""
	def __init__(self):
		super(Input, self).__init__()
		# self.arg = arg
		
	def readfile(self, filein):
		input_file = open(filein, 'r')
		for i in input_file:
			i = i.rstrip()
			vet = i.split(",")
			print vet[0]

