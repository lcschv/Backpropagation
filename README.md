Neural network to Handwriting Recognition
========

## Purpose
This is a implementation of three-layer neural network designed to recognize hand-written digits using the MNIST dataset as training.
We implement the backpropagation algorithm using three different gradient.

## How to use
You can run the backpropagation algorithm by executing:

```
 python backpropagation.py -i "input_path" -o "outputfile_path" -b "Batch_size" -lr "learning_rate" -n "NumOfNeuronsHiddenLayer" -e "NumberOfEpochs"
```
You can also execute the code using the bashscript in the folder. This bashscript is for analysis. It will run the backpropagation algorithm vayring different parameters as learning_rate (0.02, 0.5, 1, 10), the batch_size (5000, 50, 10, 1) and number of neurons in the hidden layer (25, 50, 100), then it will plot the results in the Results folder:

```
sh runner.sh

```
The `output_` paths will contain the log for the 1000 epochs running each of the algorithms. It will have .txt files containing 
`time_spent, epoch, crossentropy`.
