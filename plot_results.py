#!/usr/bin/python
#!/usr/bin/python

import numpy as np
import matplotlib.pyplot as plt

folders = ["gd","mb10", "mb50","sgd"]
neurons = [25,50,100]

for i in folders:
    for j in neurons:
        with open("output_"+str(i)+"/neuros_log_"+str(i)+"_002_"+str(j)+".txt") as f:
            lines = f.readlines()
            time = [line.split("\t")[0] for line in lines]
            x = [line.split("\t")[1] for line in lines]
            a = [line.split("\t")[2] for line in lines]

        with open("output_"+str(i)+"/neuros_log_"+str(i)+"_005_"+str(j)+".txt") as f:
            lines = f.readlines()
            # time = [line.split("\t")[0] for line in lines]
            # x = [line.split("\t")[1] for line in lines]
            b = [line.split("\t")[2] for line in lines]

        with open("output_"+str(i)+"/neuros_log_"+str(i)+"_1_"+str(j)+".txt") as f:
            lines = f.readlines()
            # time = [line.split("\t")[0] for line in lines]
            # x = [line.split("\t")[1] for line in lines]
            c = [line.split("\t")[2] for line in lines]

        with open("output_"+str(i)+"/neuros_log_"+str(i)+"_10_"+str(j)+".txt") as f:
            lines = f.readlines()
            # time = [line.split("\t")[0] for line in lines]
            # x = [line.split("\t")[1] for line in lines]
            d = [line.split("\t")[2] for line in lines]

        fig = plt.figure()
        ax1 = fig.add_subplot(111)
        # ax1.set_title("Gradient Descent")   
        ax1.set_xlabel('Epochs')
        ax1.set_ylabel('Cross Entropy')

        ax1.plot(x,a, c='r', label='lr 0.01')
        ax1.plot(x,b, c='b', label='lr 0.5')
        ax1.plot(x,c, c='g', label='lr 1')
        ax1.plot(x,d, c='m', label='lr 10')

        leg = ax1.legend()
        plt.savefig("Results/"+str(i)+"_"+str(j)+".eps", dpi=1000, format='eps')

# savefig('foo.png', bbox_inches='tight')
# plt.show()
