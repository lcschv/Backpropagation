#!/usr/bin/python
import argparse
import sys
# from src.input import Input
from src.neuralnet import NeuralNet

__author__ = 'Lucas Chaves'


def get_args():
	#Description of the program
	parser = argparse.ArgumentParser(description='Realize the hand-written digits recognition.')
	#Add arguments to the program
	parser.add_argument(
		"-i", "--input", help="Directs the input to the dataset of your choice.", required=True)
	parser.add_argument(
		"-o", "--output", help="Directs the output to a name of your choice.", required=True)
	parser.add_argument(
		"-b", "--batch_size", type=int, help="Select the algorithm strategy by chosing the batch_size. Ps.: set 1 to SGD or 5000 to GD.", required=True)
	parser.add_argument(
		"-lr", "--learning_rate",type=float, help="Select the learning_rate that you want to use.", required=True)
	parser.add_argument(
		"-n", "--num_neurons", type=int, help="Select the number of neurons in the hidden layer.", required=True)
	parser.add_argument(
		"-e", "--ephocs", type=int, help="Number of epochs that you want to compute.", required=True)
	#Array containing all the prguments passed to the program
	args = parser.parse_args()
	
	#Assign args to variables
	input_file = args.input
	output_file = args.output
	batch_size = args.batch_size
	learning_rate = args.learning_rate
	num_neurons = args.num_neurons
	ephocs = args.ephocs

	#return variables
	return input_file, output_file, batch_size, learning_rate, num_neurons, ephocs


def main():
	filein, fileout, batch_size, learning_rate, num_neurons, ephocs = get_args()
	
	neuralnet = NeuralNet(filein, fileout, batch_size, learning_rate, num_neurons, ephocs)

if __name__ == "__main__":
	main()