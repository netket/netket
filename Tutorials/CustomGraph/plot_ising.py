import matplotlib.pyplot as plt
import json

plt.ion()

# In this example we plot the energy variance

while(True):
    plt.clf()
    plt.ylabel('EnergyVariance')
    plt.xlabel('Iteration #')

    iters = []
    energy = []
    sigma = []
    evar = []
    evarsig = []

    data = json.load(open('test.log'))
    for iteration in data["Output"]:
        iters.append(iteration["Iteration"])
        energy.append(iteration["Energy"]["Mean"])
        sigma.append(iteration["Energy"]["Sigma"])
        evar.append(iteration["EnergyVariance"]["Mean"])
        evarsig.append(iteration["EnergyVariance"]["Sigma"])

    plt.semilogy()
    plt.errorbar(iters, evar, yerr=evarsig, color='red')

    plt.legend(frameon=False)
    plt.pause(1)

plt.ioff()
plt.show()
