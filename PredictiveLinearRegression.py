#Samuel Makarovskiy, Bayesian ML HW2 (Figure 3.8 Simulation)

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import multivariate_normal
from scipy.stats import norm
from math import sqrt


def main():

    #Known Parameters
    noiseSD = .2            #guess on gaussian noise SD
    Beta = (1/noiseSD)**2   #guess on Beta
    alpha = 2               #guess on prior
    density = 100           #how dense plotting is (arbitrary)
    pi = np.pi              #pi re-definition

    #Plot Configuration - Bunch of Beautification
    fig = plt.figure(figsize=(10,8))
    ax = fig.subplots(2,2)
    plt.setp(ax, adjustable = 'box', xlim = (-.05,1.05), ylim = (-1.5,1.5), 
                 xticks = [0,1],  yticks = [-1,0,1],
                 xlabel = 'x', ylabel = 't')
    fig.tight_layout()          #Make sure different subplots don't overlap
    
    
    xtrue = np.linspace(0,1,density)    #True function x values
    ytrue = np.sin(2*pi*xtrue)          #True function y values
    #plot True function on each plot for reference 
    ax[0,0].plot(xtrue,ytrue,'lawngreen')   
    ax[1,0].plot(xtrue,ytrue,'lawngreen')
    ax[0,1].plot(xtrue,ytrue,'lawngreen')
    ax[1,1].plot(xtrue,ytrue,'lawngreen')
    
    #Generation of 25 data points
    #generate random gaussian noise on sample targets (y):
    noise_sample = np.random.normal(0,noiseSD,size = 25) 
    x_sample = np.random.uniform(low = 0, high = 1.0, size = 25) #generate random x-samples
    #generate y samples from x-samples plugged into true function with gaussian noise
    y_sample = np.sin(2*pi*x_sample) + noise_sample  
    ax[0,0].plot(x_sample[:1],y_sample[:1],'bo', mfc = 'none') #plot samples as appropriate
    ax[0,1].plot(x_sample[:2],y_sample[:2],'bo', mfc = 'none')
    ax[1,0].plot(x_sample[:4],y_sample[:4],'bo', mfc = 'none')
    ax[1,1].plot(x_sample,y_sample,'bo', mfc = 'none')


    #Creation of basis functions
    s = .15         #visual guess: SD of gaussian basis functions 
    basisFunctions = [None]*9 #empty 9 gaussian function array
    for i in range(9):
        #9 equally distributed gaussian basis functions with mean in [0,1]
        basisFunctions[i] = norm(i/8,s) 
    phi = np.empty([25,10]) #phi array for 25 samples and 9 gaussian + 1 constant basis function
    for i in range(25):
        phi[i][0] = 1       #First basis function is constant for all samples
        for j in range(9):
            #Remaining 9 basis functions are gaussians evaluated at that sample#:
            phi[i][j+1] = basisFunctions[j].pdf(x_sample[i])
  
    #End goal is to use eqn 3.58 and 3.59 to make plots but we need 3.53 and 3.54 to get there
    #eq 3.54 for each sample:
    Sn_sample1 = np.linalg.inv(alpha*np.eye(10)+Beta*np.dot(phi[:1].T,phi[:1])) 
    Sn_sample2 = np.linalg.inv(alpha*np.eye(10)+Beta*np.dot(phi[:2].T,phi[:2]))
    Sn_sample4 = np.linalg.inv(alpha*np.eye(10)+Beta*np.dot(phi[:4].T,phi[:4]))
    Sn_sample25 = np.linalg.inv(alpha*np.eye(10)+Beta*np.dot(phi.T,phi))

    #eq 3.53 for each sample:
    mN_sample1 = Beta*np.dot(np.dot(Sn_sample1,phi[:1].T),y_sample[:1].reshape(-1,1)) 
    mN_sample2 = Beta*np.dot(np.dot(Sn_sample2,phi[:2].T),y_sample[:2].reshape(-1,1))
    mN_sample4 = Beta*np.dot(np.dot(Sn_sample4,phi[:4].T),y_sample[:4].reshape(-1,1))
    mN_sample25 = Beta*np.dot(np.dot(Sn_sample25,phi.T),y_sample.reshape(-1,1))

    stDev = np.empty([4,density]) #empty stdev matrix for each x-coord for each sample (of 4)
    means = np.empty([4,density]) #empty means matrix for each x-coord for each sample (of 4) 
    phitemp = np.empty([10])    #empty phi to be evaluated at every x-true test point
    phitemp[0] = 1              #first phi is always constant 1 for every x
    
    for i in range(density):
        for j in range(9):
            #recalculate basis function pdfs in phi for every xtrue point
            phitemp[j+1] = basisFunctions[j].pdf(xtrue[i])
        #calculate stDev for each xtrue point for each sample - eq 3.58:
        stDev[0][i] = sqrt(1/Beta + np.dot(np.dot(phitemp,Sn_sample1),phitemp.reshape(-1,1)))
        stDev[1][i] = sqrt(1/Beta + np.dot(np.dot(phitemp,Sn_sample2),phitemp.reshape(-1,1)))
        stDev[2][i] = sqrt(1/Beta + np.dot(np.dot(phitemp,Sn_sample4),phitemp.reshape(-1,1)))
        stDev[3][i] = sqrt(1/Beta + np.dot(np.dot(phitemp,Sn_sample25),phitemp.reshape(-1,1)))
        #calculate mean for each xtrue point for each sample - eq 3.59:
        means[0][i] = np.dot(mN_sample1.T,phitemp.reshape(-1,1))
        means[1][i] = np.dot(mN_sample2.T,phitemp.reshape(-1,1))
        means[2][i] = np.dot(mN_sample4.T,phitemp.reshape(-1,1))
        means[3][i] = np.dot(mN_sample25.T,phitemp.reshape(-1,1))
    #plot the means for each sample as a line and fill between mean-SD and mean+SD:
    ax[0,0].plot(xtrue,means[0],'r-',)
    ax[0,0].fill_between(xtrue,np.subtract(means[0],stDev[0]),np.add(means[0],stDev[0]),color = 'pink')
    ax[0,1].plot(xtrue,means[1],'r-',)
    ax[0,1].fill_between(xtrue,np.subtract(means[1],stDev[1]),np.add(means[1],stDev[1]),color = 'pink')
    ax[1,0].plot(xtrue,means[2],'r-',)
    ax[1,0].fill_between(xtrue,np.subtract(means[2],stDev[2]),np.add(means[2],stDev[2]),color = 'pink')
    ax[1,1].plot(xtrue,means[3],'r-',)
    ax[1,1].fill_between(xtrue,np.subtract(means[3],stDev[3]),np.add(means[3],stDev[3]),color = 'pink')
    #Show Plot
    plt.show()




if __name__ == '__main__':
    main()
