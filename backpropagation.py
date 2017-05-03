#!/usr/bin/python
import argparse
import sys
# from src.input import Input
from src.neuralnet import NeuralNet
__author__ = 'Lucas Chaves'


def get_args():
	#Description of the program
	parser = argparse.ArgumentParser(description='Realize the handwriting recognition.')
	#Add arguments to the program
	parser.add_argument(
		"-i", "--input", help="Directs the input to the dataset of your choice", required=True)
	parser.add_argument(
		"-o", "--output", help="Directs the output to a name of your choice", required=True)
	parser.add_argument(
		"-a", "--algorithm", help="Select the algorithm strategy that will be used to calculate the gradient", required=True)
	#Array containing all the prguments passed to the program
	args = parser.parse_args()
	#Assign args to variables
	input_file = args.input
	output_file = args.output
	algorithm = args.algorithm
	
	#return variables
	return input_file, output_file, algorithm


def main():
	filein, fileout, algorithm = get_args()
	
	neuralnet = NeuralNet(filein)

	neuralnet.setOutputUnits()
	output_file = open(fileout, 'w')


if __name__ == "__main__":
	main()