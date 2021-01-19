import numpy as np
import matplotlib.pyplot as plt
import json

plt.ion()

# N=14
exact = -17.86280854

while True:
    plt.clf()
    plt.ylabel("Energy")
    plt.xlabel("Time $i\\tau$")

    iters = []
    times = []
    energy = []
    sigma = []
    evar = []

    data = json.load(open("ising_imag.log"))
    for iteration in data["Output"]:
        iters.append(iteration["Iteration"])
        times.append(iteration["Time"])
        energy.append(iteration["Energy"]["Mean"])
        sigma.append(iteration["Energy"]["Sigma"])
        evar.append(iteration["Energy"]["Variance"])

    nres = len(iters)
    cut = 100
    if nres > cut:

        fitx = iters[-cut:-1]
        fity = energy[-cut:-1]
        z = np.polyfit(fitx, fity, deg=0)
        p = np.poly1d(z)

        plt.xlim([nres - cut, nres])
        maxval = np.max(energy[-cut:-1])
        plt.ylim([exact - (np.abs(exact) * 0.01), maxval + np.abs(maxval) * 0.01])
        error = (z[0] - exact) / -exact
        plt.gca().text(
            0.95,
            0.8,
            "Relative Error : " + "{:.2e}".format(error),
            verticalalignment="bottom",
            horizontalalignment="right",
            color="green",
            fontsize=15,
            transform=plt.gca().transAxes,
        )

        plt.plot(fitx, p(fitx))

    plt.plot(times, energy, color="red")
    plt.axhline(y=exact, xmin=0, xmax=iters[-1], linewidth=2, color="k", label="Exact")

    plt.legend(frameon=False)
    plt.pause(1)
    # plt.draw()

plt.ioff()
plt.show()
