#!/usr/bin/env python
# Copyright 2018 The Simons Foundation, Inc. - All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
'''
Netket python utilities.

'''
import json
import numpy as np
import matplotlib.pyplot as plt


def Message(type, message):
    if type == "Info":
        print("# " + message)

    elif type == "Warning":
        print("# WARNING: " + message)

    elif type == "Error":
        print("# ERROR: " + message)

    elif type == "Debug":
        print("# DEBUG: " + message)

    else:
        raise ValueError("%s Message not supported" % type)


def plot_output(exact, outputfile):
    plt.ion()
    plt.pause(10)
    while (True):
        plt.clf()
        plt.ylabel('Energy')
        plt.xlabel('Iteration #')

        iters = []
        energy = []
        sigma = []
        evar = []
        evarsig = []

        data = json.load(open(outputfile + ".log"))
        for iteration in data["Output"]:
            iters.append(iteration["Iteration"])
            energy.append(iteration["Energy"]["Mean"])
            sigma.append(iteration["Energy"]["Sigma"])
            evar.append(iteration["EnergyVariance"]["Mean"])
            evarsig.append(iteration["EnergyVariance"]["Sigma"])

        nres = len(iters)
        cut = 60
        if (nres > cut):

            fitx = iters[-cut:-1]
            fity = energy[-cut:-1]
            z = np.polyfit(fitx, fity, deg=0)
            p = np.poly1d(z)

            plt.xlim([nres - cut, nres])
            maxval = np.max(energy[-cut:-1])
            plt.ylim([
                exact - (np.abs(exact) * 0.01), maxval + np.abs(maxval) * 0.01
            ])
            error = (z[0] - exact) / -exact
            plt.gca().text(
                0.95,
                0.8,
                'Relative Error : ' + "{:.2e}".format(error),
                verticalalignment='bottom',
                horizontalalignment='right',
                color='green',
                fontsize=15,
                transform=plt.gca().transAxes)

            plt.plot(fitx, p(fitx))

        plt.errorbar(iters, energy, yerr=sigma, color='red')
        plt.axhline(
            y=exact,
            xmin=0,
            xmax=iters[-1],
            linewidth=2,
            color='k',
            label='Exact')

        plt.legend(frameon=False)
        plt.pause(1)
        # plt.draw()

    plt.ioff()
    plt.show()
