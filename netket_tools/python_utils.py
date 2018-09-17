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
"""
Netket python utilities.
"""
import json
import numpy as np
import matplotlib.pyplot as plt


def encode_complex(z):
    if isinstance(z, complex):
        return (z.real, z.imag)
    else:
        type_name = z.__class__.__name__
        raise TypeError(
            f"Object of type '{type_name}' is not JSON serializable")


def get_obsv_from_json(outputfile):
    """
    Reads observables from the NetKet's json output file.

    Arguments
    ---------

        outputfile : string
            File name (not including json file extension).

    Returns
    -------

        data : dict
            Contains the data from the json output file stored as np.ndarrays.
    """

    # Read in data
    raw_data = json.load(open(outputfile + ".log"))

    # Set up dictionary to return
    data = {}
    for k, v in raw_data["Output"][0].items():
        # For everything other than iteration, initialize as a dict
        if isinstance(v, dict):
            data[k] = {}
            for ki, _ in v.items():
                data[k][ki] = []
        else:
            data[k] = []

    # Read in all the data
    for iteration in raw_data["Output"]:
        for k, v in iteration.items():
            # For everything other than iteration, initialize as a dict
            if isinstance(v, dict):
                for ki, _ in v.items():
                    data[k][ki].append(iteration[k][ki])
            else:
                data[k].append(iteration[k])

    # Convert all arrays to ndarrays
    for k, v in data.items():
        # For everything other than iteration, initialize as a dict
        if isinstance(v, dict):
            for ki, _ in v.items():
                data[k][ki] = np.array(data[k][ki])
        else:
            data[k] = np.array(data[k])

    return data


def plot_observable(outputfile, observable, exact=None):
    """
    Arguments
    ---------

        outputfile : string
            Path to NetKet output.

        observable : string
            Name of observable as it's given in output. The standard observables
            dumped to the output are Energy and EnergyVariance. Other custom
            observables can also be used. See the tutorial
            Input_Driver/sigmax.py.

        exact : None (or float)
            Exact answer for observable. If given, used to calculate the error.
    """

    plt.ion()
    plt.pause(2)  # Necessary to give NetKet time to write to outputfile
    while (True):
        plt.clf()
        plt.ylabel(observable)
        plt.xlabel('Iteration #')

        data = get_obsv_from_json(outputfile)
        iters = data["Iteration"]
        obsv = data[observable]["Mean"]
        sigma = data[observable]["Sigma"]

        nres = len(iters)
        cut = 60
        if (nres > cut):

            fitx = iters[-cut:-1]
            fity = obsv[-cut:-1]
            z = np.polyfit(fitx, fity, deg=0)
            p = np.poly1d(z)

            plt.xlim([nres - cut, nres])
            maxval = np.max(obsv[-cut:-1])
            minval = np.min(obsv[-cut:-1])

            if exact != None:
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

                plt.axhline(
                    y=exact,
                    xmin=0,
                    xmax=iters[-1],
                    linewidth=2,
                    color='k',
                    label='Exact')

                plt.legend(frameon=False)

                plt.ylim([
                    exact - (np.abs(exact) * 0.01),
                    maxval + (np.abs(maxval) * 0.01)
                ])
            else:
                plt.ylim([
                    minval - (np.abs(minval) * 0.01),
                    maxval + (np.abs(maxval) * 0.01)
                ])

            plt.plot(fitx, p(fitx))
            plt.errorbar(iters, obsv, yerr=sigma, color='red')
        plt.pause(1)
        # plt.draw()

    plt.ioff()
    plt.show()
