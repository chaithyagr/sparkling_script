#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 14 14:07:30 2019

@author: nc258476
"""

import numpy as np
import matplotlib.pyplot as plt

def dispSampling(shot, decim, k):
    
#    print("size of shot = " + str(shot.shape))
#
#    print("nb of zero elements in kvec x :")
#    print(len(np.argwhere(shot[:,0] == 0)))
#
#    print("nb ofzero elements in kvec y :")
#    print(len(np.argwhere(shot[:,1] == 0)))
    
    plt.figure()
    plt.scatter(shot[:,0], shot[:,1], marker = '.', s = 0.2)

    plt.grid()
    
    #Set labels
    unit = 1/4 ; tick = np.arange(-0.5, 0.5 + unit, unit)

    label_pi = [r"$-\frac{1}{2\pi}$", r"$-\frac{1}{4\pi}$", r"$0$", r"$+\frac{1}{4\pi}$",  r"$+\frac{1}{2\pi}$"]

    plt.xticks(tick/np.pi,labels = label_pi, fontsize = 16) ; plt.yticks(tick/np.pi,labels = label_pi, fontsize = 16)
    plt.title("K-space sampling : decim = " +str(decim) + " and k = " + str(k), fontsize = 18)
    plt.xlabel(r"$k_x$", fontsize = 22) ; plt.ylabel(r"$k_y$", fontsize = 20)
    #plt.legend(fontsize = 16)
    plt.show()