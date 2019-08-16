import numpy as np
import matplotlib.pyplot as plt
import json

plt.ion()

# N=20
exact = 0.637275 * 20

# N=80
# exact=-1.273321360724e+00*80

while True:
    plt.clf()
    plt.ylabel("Sigmax")
    plt.xlabel("Iteration #")

    iters = []
    sx = []
    sigma = []

    data = json.load(open("test.log"))
    for iteration in data["Output"]:
        iters.append(iteration["Iteration"])
        sx.append(iteration["SigmaX"]["Mean"])
        sigma.append(iteration["SigmaX"]["Sigma"])

    nres = len(iters)
    cut = 100
    if nres > cut:

        fitx = iters[-cut:-1]
        fity = sx[-cut:-1]
        z = np.polyfit(fitx, fity, deg=0)
        p = np.poly1d(z)

        maxval = np.max(sx[-cut:-1])
        error = np.abs((z[0] - exact) / exact)
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

    plt.errorbar(iters, sx, yerr=sigma, color="red")
    plt.axhline(y=exact, xmin=0, xmax=iters[-1], linewidth=2, color="k", label="Exact")

    plt.legend(frameon=False)
    plt.pause(1)
    # plt.draw()

plt.ioff()
plt.show()
