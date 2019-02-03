import numpy as np
import matplotlib.pyplot as plt
import json

plt.ion()

while(True):
    plt.clf()
    plt.ylabel('Overlap Error = 1-F')
    plt.xlabel('Iteration #')

    iters=[]
    log_overlap=[]
    mse=[]
    mse_log=[]

    data=json.load(open('output.log'))
    for iteration in data["Output"]:
        iters.append(iteration["Iteration"])
        log_overlap.append(iteration["log_overlap"])
        mse.append(iteration["mse"])
        mse_log.append(iteration["mse_log"])

    overlap = np.exp(-np.array(log_overlap))

    plt.semilogy(iters, 1.-overlap)
    plt.axhline(y=1, xmin=0, xmax=iters[-1], linewidth=2, color='k',label='max error = 1')

    plt.legend(frameon=False)

    J2 = 0.4
    plt.title(r'$J_1 J_2$ model, $J_2=' + str(J2) + '$')

    plt.pause(1)


    plt.show()
