import numpy as np
import matplotlib.pyplot as plt
import json

plt.ion()


exact = -35.6175461195


while True:
    plt.clf()
    plt.ylabel("Energy")
    plt.xlabel("Iteration #")

    iters = []
    energy = []
    sigma = []
    evar = []

    data = json.load(open("test.log"))
    for iteration in data["Output"]:
        iters.append(iteration["Iteration"])
        energy.append(iteration["Energy"]["Mean"])
        sigma.append(iteration["Energy"]["Sigma"])
        evar.append(iteration["Energy"]["Variance"])

    plt.errorbar(iters, energy, yerr=sigma, color="red")
    plt.axhline(y=exact, xmin=0, xmax=iters[-1], linewidth=2, color="k", label="Exact")
    nres = len(iters)
    if nres > 100:
        fitx = iters[-30:-1]
        fity = energy[-30:-1]
        z = np.polyfit(fitx, fity, deg=0)
        p = np.poly1d(z)

        plt.plot(fitx, p(fitx))
        if nres > 100:
            plt.xlim([nres - 100, nres])
            maxval = np.max(energy[-100:-1])
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

    plt.legend(frameon=False)
    plt.pause(1)


plt.show()
