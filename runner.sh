#!/bin/sh
mkdir output_gd
mkdir output_sgd
mkdir output_mb10
mkdir output_mb50gd

echo "Starting processing GD.."
python backpropagation.py -i input/data_tp1 -o output_gd/neuros_log_GD_002_25.txt -b 5000 -lr 0.02 -n 25 -e 1000
python backpropagation.py -i input/data_tp1 -o output_gd/neuros_log_GD_002_50.txt -b 5000 -lr 0.02 -n 50 -e 1000
python backpropagation.py -i input/data_tp1 -o output_gd/neuros_log_GD_002_100.txt -b 5000 -lr 0.02 -n 100 -e 1000
python backpropagation.py -i input/data_tp1 -o output_gd/neuros_log_GD_005_25.txt -b 5000 -lr 0.5 -n 25 -e 1000
python backpropagation.py -i input/data_tp1 -o output_gd/neuros_log_GD_005_50.txt -b 5000 -lr 0.5 -n 50 -e 1000
python backpropagation.py -i input/data_tp1 -o output_gd/neuros_log_GD_005_100.txt -b 5000 -lr 0.5 -n 100 -e 1000
python backpropagation.py -i input/data_tp1 -o output_gd/neuros_log_GD_1_25.txt -b 5000 -lr 1 -n 25 -e 1000
python backpropagation.py -i input/data_tp1 -o output_gd/neuros_log_GD_1_50.txt -b 5000 -lr 1 -n 50 -e 1000
python backpropagation.py -i input/data_tp1 -o output_gd/neuros_log_GD_1_100.txt -b 5000 -lr 1 -n 100 -e 1000
python backpropagation.py -i input/data_tp1 -o output_gd/neuros_log_GD_10_25.txt -b 5000 -lr 10 -n 25 -e 1000
python backpropagation.py -i input/data_tp1 -o output_gd/neuros_log_GD_10_50.txt -b 5000 -lr 10 -n 50 -e 1000
python backpropagation.py -i input/data_tp1 -o output_gd/neuros_log_GD_10_100.txt -b 5000 -lr 10 -n 100 -e 1000
echo "Done processing GD.."


echo "Starting MB_10 processing.."
python backpropagation.py -i input/data_tp1 -o output_mb10/neuros_log_mb10_002_25.txt -b 10 -lr 0.02 -n 25 -e 1000
python backpropagation.py -i input/data_tp1 -o output_mb10/neuros_log_mb10_002_50.txt -b 10 -lr 0.02 -n 50 -e 1000
python backpropagation.py -i input/data_tp1 -o output_mb10/neuros_log_mb10_002_100.txt -b 10 -lr 0.02 -n 100 -e 1000
python backpropagation.py -i input/data_tp1 -o output_mb10/neuros_log_mb10_005_25.txt -b 10 -lr 0.5 -n 25 -e 1000
python backpropagation.py -i input/data_tp1 -o output_mb10/neuros_log_mb10_005_50.txt -b 10 -lr 0.5 -n 50 -e 1000
python backpropagation.py -i input/data_tp1 -o output_mb10/neuros_log_mb10_005_100.txt -b 10 -lr 0.5 -n 100 -e 1000
python backpropagation.py -i input/data_tp1 -o output_mb10/neuros_log_mb10_1_25.txt -b 10 -lr 1 -n 25 -e 1000
python backpropagation.py -i input/data_tp1 -o output_mb10/neuros_log_mb10_1_50.txt -b 10 -lr 1 -n 50 -e 1000
python backpropagation.py -i input/data_tp1 -o output_mb10/neuros_log_mb10_1_100.txt -b 10 -lr 1 -n 100 -e 1000
python backpropagation.py -i input/data_tp1 -o output_mb10/neuros_log_mb10_10_25.txt -b 10 -lr 10 -n 25 -e 1000
python backpropagation.py -i input/data_tp1 -o output_mb10/neuros_log_mb10_10_50.txt -b 10 -lr 10 -n 50 -e 1000
python backpropagation.py -i input/data_tp1 -o output_mb10/neuros_log_mb10_10_100.txt -b 10 -lr 10 -n 100 -e 1000
echo "Done processing MB_10.."


echo "Start processing MB_50.."
python backpropagation.py -i input/data_tp1 -o output_mb50gd/neuros_log_mb50_002_25.txt -b 50 -lr 0.02 -n 25 -e 1000
python backpropagation.py -i input/data_tp1 -o output_mb50/neuros_log_mb50_002_50.txt -b 50 -lr 0.02 -n 50 -e 1000
python backpropagation.py -i input/data_tp1 -o output_mb50/neuros_log_mb50_002_100.txt -b 50 -lr 0.02 -n 100 -e 1000
python backpropagation.py -i input/data_tp1 -o output_mb50/neuros_log_mb50_005_25.txt -b 50 -lr 0.5 -n 25 -e 1000
python backpropagation.py -i input/data_tp1 -o output_mb50/neuros_log_mb50_005_50.txt -b 50 -lr 0.5 -n 50 -e 1000
python backpropagation.py -i input/data_tp1 -o output_mb50/neuros_log_mb50_005_100.txt -b 50 -lr 0.5 -n 100 -e 1000
python backpropagation.py -i input/data_tp1 -o output_mb50/neuros_log_mb50_1_25.txt -b 50 -lr 1 -n 25 -e 1000
python backpropagation.py -i input/data_tp1 -o output_mb50/neuros_log_mb50_1_50.txt -b 50 -lr 1 -n 50 -e 1000
python backpropagation.py -i input/data_tp1 -o output_mb50/neuros_log_mb50_1_100.txt -b 50 -lr 1 -n 100 -e 1000
python backpropagation.py -i input/data_tp1 -o output_mb50/neuros_log_mb50_10_25.txt -b 50 -lr 10 -n 25 -e 1000
python backpropagation.py -i input/data_tp1 -o output_mb50/neuros_log_mb50_10_50.txt -b 50 -lr 10 -n 50 -e 1000
python backpropagation.py -i input/data_tp1 -o output_mb50/neuros_log_mb50_10_100.txt -b 50 -lr 10 -n 100 -e 1000
echo "Done processing MB.."


echo "Starting SGD processing.."
python backpropagation.py -i input/data_tp1 -o output_sgd/neuros_log_SGD_002_25.txt -b 1 -lr 0.02 -n 25 -e 1000
python backpropagation.py -i input/data_tp1 -o output_sgd/neuros_log_SGD_002_50.txt -b 1 -lr 0.02 -n 50 -e 1000
python backpropagation.py -i input/data_tp1 -o output_sgd/neuros_log_SGD_002_100.txt -b 1 -lr 0.02 -n 100 -e 1000
python backpropagation.py -i input/data_tp1 -o output_sgd/neuros_log_SGD_005_25.txt -b 1 -lr 0.5 -n 25 -e 1000
python backpropagation.py -i input/data_tp1 -o output_sgd/neuros_log_SGD_005_50.txt -b 1 -lr 0.5 -n 50 -e 1000
python backpropagation.py -i input/data_tp1 -o output_sgd/neuros_log_SGD_005_100.txt -b 1 -lr 0.5 -n 100 -e 1000
python backpropagation.py -i input/data_tp1 -o output_sgd/neuros_log_SGD_1_25.txt -b 1 -lr 1 -n 25 -e 1000
python backpropagation.py -i input/data_tp1 -o output_sgd/neuros_log_SGD_1_50.txt -b 1 -lr 1 -n 50 -e 1000
python backpropagation.py -i input/data_tp1 -o output_sgd/neuros_log_SGD_1_100.txt -b 1 -lr 1 -n 100 -e 1000
python backpropagation.py -i input/data_tp1 -o output_sgd/neuros_log_SGD_10_25.txt -b 1 -lr 10 -n 25 -e 1000
python backpropagation.py -i input/data_tp1 -o output_sgd/neuros_log_SGD_10_50.txt -b 1 -lr 10 -n 50 -e 1000
python backpropagation.py -i input/data_tp1 -o output_sgd/neuros_log_SGD_10_100.txt -b 1 -lr 10 -n 100 -e 1000
echo "Done processing SGD.."

# etc.