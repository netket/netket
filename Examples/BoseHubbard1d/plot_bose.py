import numpy as np
import matplotlib.pyplot as plt
import json

plt.ion()


# U=1 N=10
# exact=-16.17945609

# U=4 N=10
# exact=-9.23881058

# U=10 N=10
# exact=-3.95910804

# U=4 N=12
exact = -11.03872267

while True:
    plt.clf()
    plt.ylabel("Energy")
    plt.xlabel("Iteration #")

    iters = []
    energy = []
    sigma = []
    evar = []
    evarsig = []

    data = json.load(open("test.log"))
    for iteration in data["Output"]:
        iters.append(iteration["Iteration"])
        energy.append(iteration["Energy"]["Mean"])
        sigma.append(iteration["Energy"]["Sigma"])
        evar.append(iteration["EnergyVariance"]["Mean"])
        evarsig.append(iteration["EnergyVariance"]["Sigma"])

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

    plt.errorbar(iters, energy, yerr=sigma, color="red")
    plt.axhline(y=exact, xmin=0, xmax=iters[-1], linewidth=2, color="k", label="Exact")

    plt.legend(frameon=False)
    plt.pause(1)
    # plt.draw()

plt.ioff()
plt.show()
