#!/bin/bash
mkdir saidas_gd
mkdir saidas_sgd
mkdir saidas_mb10
mkdir saidas_mb50

folders=("gd" "sgd" "mb10" "mb50")
lr=(0.02 0.5 1 10)
names=("002" "005" "1" "10")
batchs=(5000 1 10 50)
neurons=(25 50 100)

#Script to run all the different parameters ...
for ((i=0;i<4;i++));
do 	
	for ((j=0;j<4;j++))
	do
		for n in "${neurons[@]}"
		do
			echo "python backpropagation.py -i input/data_tp1 -o output_${folders[$i]}/neuros_log_${folders[$i]}_${names[$j]}_$n.txt -b ${batchs[$i]} -lr ${lr[j]} -n $n -e 10"
			python backpropagation.py -i input/data_tp1 -o saidas_${folders[$i]}/neuros_log_${folders[$i]}_${names[$j]}_$n.txt -b ${batchs[$i]} -lr ${lr[j]} -n $n -e 10

		done
	done
done

# Calling the script to create plots ..
mkdir Results2
python plot_results.py