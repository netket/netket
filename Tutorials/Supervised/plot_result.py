import numpy as np
import matplotlib.pyplot as plt
import json

plt.ion()


exact=0.


while(True):
    plt.clf()
    plt.ylabel('Ratio Variance')
    plt.xlabel('Iteration #')

    iters=[]
    energy=[]
    sigma=[]
    evar=[]
    evarsig=[]

    data=json.load(open('test.log'))
    for iteration in data["Output"]:
        iters.append(iteration["Iteration"])
        energy.append(iteration["Ratio"]["Mean"])
        sigma.append(iteration["Ratio"]["Sigma"])
        evar.append(iteration["RatioVariance"]["Mean"])
        evarsig.append(iteration["RatioVariance"]["Sigma"])

    plt.errorbar(iters,evar,yerr=evarsig,color='red')
    plt.axhline(y=exact, xmin=0, xmax=iters[-1], linewidth=2, color = 'k',label='Exact')
    nres=len(iters)
    if(nres>100):
        fitx=iters[-100:-1]
        fity=evar[-100:-1]
        z=np.polyfit(fitx,fity,deg=0)
        p = np.poly1d(z)

        plt.plot(fitx,p(fitx))
        # if(nres>100):
        #     plt.xlim([nres-100,nres])
        #     maxval=np.max(energy[-100:-1])
        #     plt.ylim([exact-(np.abs(exact)*0.01),maxval+np.abs(maxval)*0.01])
        error=(z[0]-exact)/-(z[0]+exact+1e-16)
        plt.gca().text(0.95, 0.8, 'Relative Error : '+"{:.2e}".format(error),
        verticalalignment='bottom', horizontalalignment='right',
        color='green', fontsize=15,transform=plt.gca().transAxes)

    plt.legend(frameon=False)
    plt.pause(1)


plt.show()
