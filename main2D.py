#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os.path as op
import os
import time
import numpy as np
import math
import matplotlib.pyplot as plt
import sys
from scipy import interpolate
from collections import namedtuple
import copy
import datetime

#Modules needed for SPARKLING
import algo2D_Init as init_2D
import constraints_sets as Cstr
import projectionOp as projOp
import projectedGradientDescent as pgd
import plotTrajectories

#%%
#get current working dir
cwd = os.getcwd()
img_size = 512   #assuming a square image
FOV = 0.2 #field of view in meters
pixelSize = FOV/img_size

#%%
samplingOptions = namedtuple("samplingOptions", "nc ns OS_factor decim tau \
                             decay mask iter init Kmax Smax Gmax gamma dTime n_revol n_epi")

# nc is the number of shots in the k-space
# ns is the number of gradient samples per shot
# Oversampling factor = (gradient dwell time)/(gradient raster time)
# Kmax depends of FOV and image resolution (NS criterion)
# Gmax and Smax : maximum gradient (T/m) and slew rate (T/m/ms)
# tau and decay parametrize the variable sampling density
# mask=1 for circular isotrope kspace - 0 otherwise
# gradit = number of iterations in projected gradient descent
# init = initialization for SPARKLING (radial in-out, radial center-out, spiral in-out...)
# gamma = gyromagnetic constant (Hz/T)

samplingOptions.nc = 49
samplingOptions.ns = 2048
samplingOptions.OS_factor = 1 #gradient dwell time / gradient raster time
samplingOptions.decim = 64

#See C.Lazarus' PhD, 3.3.3 for details. Samples at the center are separated by tau*deltak, deltaK the pixel size given by Shannon
samplingOptions.decay = 2
samplingOptions.tau = 0.8

samplingOptions.init = "radialCO"
samplingOptions.mask = 1 #boolean

samplingOptions.Kmax = img_size/(2*FOV)
samplingOptions.Gmax = 40e-3 #(Tesla/meter)
samplingOptions.Smax = 200e-3 #(Tesla/meter/millisecond)

samplingOptions.gamma = 42.576e3 #gyromagnetic ratio (Hz/Tesla)
samplingOptions.dTime = 0.010     #gradient raster time in ms

samplingOptions.n_revol = 2 #number of revolutions for spirals only
samplingOptions.n_epi = 10 #number of EPI lines in k-space
samplingOptions.Shaking = 1

samplingOptions.gradit = 200 #number of iterations for the projected gradient descent

#KERNEL FOR ALGORITHM
kernelName = "dist_L2" #L2 norm

#Results file name
currentDT = datetime.datetime.now()

folderpath = '../Trajectories/'
fileName = samplingOptions.init + "_nc" + str(samplingOptions.nc) + "_ns" + str(samplingOptions.ns) \
+ "_decim" + str(samplingOptions.decim) + "_res" + str(img_size)+ "_tau" + str(samplingOptions.tau) \
+ "_decay" + str(samplingOptions.decay)+ "_ker" + kernelName
if not os.path.isdir(folderpath):
    os.mkdir(folderpath)
saveName = os.path.join(folderpath, fileName)
#%%
#CALCULATED VALUES

alpha = samplingOptions.gamma*samplingOptions.Gmax
beta = samplingOptions.gamma*samplingOptions.Smax

#Normalization of the kinetic constraints
alpha_R = alpha/(samplingOptions.Kmax)*1/(2*np.pi)
beta_R = beta/(samplingOptions.Kmax)*1/(2*np.pi)

if samplingOptions.init == "radialIO" or samplingOptions.init == "spiralIO":
     ns_shot  = 2*math.floor(samplingOptions.ns/2)+1 #odd number of samples for symmetric shots
else:
     ns_shot = samplingOptions.ns

#Number of shots at first decimation step
ns_shot_decim = math.ceil(ns_shot/samplingOptions.decim)

Dk0_R = 2*pixelSize/FOV*1/(2*np.pi) # kspace pixel size given by Shannon, divided by Kmax*2pi

eps = sys.float_info.epsilon

#Criterium (iii), eq (3.4) C.Lazarus' PhD thesis
alpha_R = min(Dk0_R/(samplingOptions.dTime*samplingOptions.OS_factor),alpha_R)

#UNDERSAMPLING FACTOR
UF = np.power(img_size,2)/(samplingOptions.nc*samplingOptions.ns) #R in C.Lazarus' PhD 3.3.2

#ACCELERATION FACTOR
AF = img_size/samplingOptions.nc

#%% PLOT INITIALIZATION
shot, k_TE, k_TE_decim = init_2D.init_Trajectory(samplingOptions, ns_shot_decim, img_size, eps)

k_vec = np.zeros((len(shot),2))
k_vec[:,0] = shot.real
k_vec[:,1] = shot.imag

# If constraint on echo time crossing, calcultation will be done for one center point in projectedGradientDescent
if samplingOptions.init != "Cartesian":
    samplingOptions.kTE_crossing = 1
else:
    samplingOptions.kTE_crossing = 0

## Debug section##########################################################################

#print("nb of zero elements in kvec x :")
#print(len(np.argwhere(k_vec[:,0] == 0)))
#
#print("nb ofzero elements in kvec y :")
#print(len(np.argwhere(k_vec[:,1] == 0)))

######################################################################################################
#Full initialization plot

kspace = plt.figure(figsize = (10,10))

if samplingOptions.init == 'EPI':
    plt.plot(k_vec[:,0],k_vec[:,1], marker = 'x', label = "Full Initialization")    
else:    
    plt.scatter(k_vec[:,0],k_vec[:,1], marker = '.', label = "Full Initialization")

plt.grid() ;

####################################################################################################################
#First shot

if samplingOptions.init != 'EPI' and samplingOptions.init !="Full_Cartesian":
    
    plt.plot(k_vec[0:ns_shot_decim,0],k_vec[0:ns_shot_decim,1], color='r', marker = 'x', label = "first shot")

###################################################################################################################
#Sample at echo time at the center of the k-space

if samplingOptions.init == "spiralIO" or samplingOptions.init == "radialIO":
    plt.scatter(k_vec[k_TE_decim-1,0], k_vec[k_TE_decim-1,1], marker = 'o', color='r', s = 150)

plt.title("K-space sampling : " + samplingOptions.init + " initialization, decim = " +
          str(samplingOptions.decim), fontsize = 18)

###################################################################################################################
#Set labels
unit = 1/4 ; tick = np.arange(-0.5, 0.5 + unit, unit)

label_pi = [r"$-\frac{1}{2\pi}$", r"$-\frac{1}{4\pi}$", r"$0$", r"$+\frac{1}{4\pi}$",  r"$+\frac{1}{2\pi}$"]

plt.xticks(tick/np.pi,labels = label_pi, fontsize = 16)
plt.yticks(tick/np.pi,labels = label_pi, fontsize = 16)
plt.xlabel(r"$k_x$", fontsize = 20)
plt.ylabel(r"$k_y$", fontsize = 20)
plt.legend(fontsize = 16)
plt.show()

#%%
#GENERATE Plateau distribution

#x = np.linspace(-1/(2*np.pi), 1/(2*np.pi),img_size)
x = np.linspace(-img_size/2+0.5, img_size/2-0.5,img_size)

#Algorithm described in C.Lazarus' PhD, section 3.3.3
X,Y = np.meshgrid(x,x) 
norm = np.sqrt(np.power(X,2)+np.power(Y,2))
p_decay = np.power(norm, -samplingOptions.decay)
p_decay = p_decay/np.sum(p_decay)

maxDens = 1/(ns_shot*samplingOptions.nc*np.power(samplingOptions.tau,2)) #eq (3.12) in C.Lazarus' PhD
print("VDS maxDens = " + str(maxDens))

p_peak = init_2D.Generate_Peak(samplingOptions.nc, samplingOptions.nc*ns_shot,img_size, 
                       alpha_R, samplingOptions.dTime, Dk0_R, samplingOptions.tau)

x = np.linspace(-1/(2*np.pi), 1/(2*np.pi),img_size)
X, Y = np.meshgrid(x,x)

density = init_2D.generate_density(p_peak, p_decay, maxDens)
density = density/np.sum(density)

#In the case an isotropic mask is applied (ie k-samples in a disc)
if samplingOptions.mask == 1:
    density[np.where(norm >= img_size/2-1/2)] = 0
    density = density/(np.sum(density)) #normalization

densPlot = plt.figure()

plt.pcolormesh(X, Y, density, shading="gouraud")
plt.colorbar()
plt.xticks(tick/np.pi,labels = label_pi, fontsize = 12)
plt.yticks(tick/np.pi,labels = label_pi, fontsize = 12)
plt.title(" Variable Density Sampling in k-space")
print("total density =  %1.6f" %np.sum(density))
plt.show()

#%%
# SET KINETIC CONSTRAINTS
func = Cstr.bound_normL2()

C_speed = Cstr.kinetic_Constraint(func, lambda s : Cstr.Prime(s,1), 
                         lambda s : Cstr.PrimeT(s,1), alpha_R*samplingOptions.dTime)

def Second(s,Dt):
    spp = -Cstr.PrimeT(Cstr.Prime(s,Dt),Dt)
    return spp

C_accel = Cstr.kinetic_Constraint(func, lambda s : Second(s,1), 
                         lambda s : Second(s,1), beta_R*np.power(samplingOptions.dTime,2))

print("C_speed.bound is %1.6f" %C_speed.bound)
print("C_accel.bound is %1.6f" %C_accel.bound)

C_kin = np.array([C_speed, C_accel])

#%%
# MAIN ALGORITHM OPTIONS
algoParams = namedtuple("algoParams", 
                        "nit L discr_step nCheb treeDepth target_acc disp_results show_prog mode kernelID")

algoParams.nit = 150 #number of iteration for projection on constraints space
algoParams.L = 16 #Lipschitz constant of the gradient
algoParams.discr_step = 1

#Parameters for Black-Box FMM ; tree depth and target accuracy are not controlled by user in this 2D implementation
algoParams.nCheb = 5 #number of Chebyshev nodes for interpolation
algoParams.mode = 1 #1 for multiple, 0 for single

#Because I am lazy, I will pass integers instead of string or char* to pybind11
if kernelName == "dist_L2":
    algoParams.kernelID = 1
elif kernelName == "log_L2":
    algoParams.kernelID = 2
elif kernelName == "gaussian":
    algoParams.kernelID = 3

saveName = saveName + "_nCheb" + str(algoParams.nCheb) + "_gradit" + str(samplingOptions.gradit) \
 +  "_D" + str(currentDT.day) + "M"+ str(currentDT.month)+ "Y" + str(currentDT.year)
 

#set to false, not available in current version
algoParams.disp_results = False
algoParams.show_prog = False

nc = samplingOptions.nc

print("ns_shot_decim = %d" %ns_shot_decim)
#print("k_TE_decim = %d" %k_TE_decim)

decim = samplingOptions.decim
stepDef = 1/(2*np.pi)*1/16 #in sparkling2D_standalone, Create_SPARKLING.m

#Structure containing shot as well as the dual variables (projection algorithm)
algoProj = namedtuple("algoProj", "s")

kshot = np.zeros((len(shot),2))
kshot[:,0] = shot.real
kshot[:,1] = shot.imag

algoProj.s = np.copy(kshot)

start = time.time()

#%%
#PROJECTED GRADIENT DESCENT SPARKLING

#Attraction field depends only on the kernel and the VDS
Att_field, x_att, y_att = pgd.Precompute_Attraction_Field(density, kernelName)

while decim >= 1 :
    
    tol = 1e-6 * decim
    
    print("decimation factor is %d" %decim)
    
    #Kinetic bounds change with decimation
    C_kin_decim = copy.deepcopy(C_kin) #Copy is mandatory in this case, otherwise C_kin will be modified as well!
    C_kin_decim[0].bound  = C_kin_decim[0].bound * decim
    C_kin_decim[1].bound = C_kin_decim[1].bound*np.power(decim,2)
    
    #Linear Constraints
    C_lin = Cstr.linear_Constraint() #"Projection au bout de la nuit"

    if samplingOptions.kTE_crossing ==1:
        Cstr.set_lin_constraints_kTE(C_lin, ns_shot_decim, 2, "TE_point", k_TE_decim)
    else:
        Cstr.set_lin_constraints_kTE(C_lin, ns_shot_decim, 2)

    
    #First Projection of Initial trajectories
    if decim == samplingOptions.decim and decim > 1:
        algoProj = projOp.project_fast_fista(C_kin_decim, C_lin, algoParams, algoProj, nc)
        #algoProj = proj_Cstr.Projector_On_Multiple_Curves_Center(C_kin_decim, C_lin, algoParams, algoProj, nc)
        kshot = np.copy(algoProj.s)
        

    #Projected Gradient Descent
    kshot = pgd.gradDescent(density, decim, kshot, C_kin_decim, C_lin, x_att, y_att, Att_field, kernelName, samplingOptions, tol,
                            int(k_TE_decim), stepDef, algoParams)
    
    # Plot distribution at the end of each decimation step
    plotTrajectories.dispSampling(kshot, decim, samplingOptions.gradit)
    
    decim = int(decim/2)
    
    ns_shot_decim_prev = ns_shot_decim
    
    if k_TE != 1: #ie if echo time is not at first sample
        ns_shot_decim = 2*(ns_shot_decim - 1) + 1
        k_TE_decim = 2*(k_TE_decim - 1) + 1
    else:
        ns_shot_decim = 2*ns_shot_decim
    
    #At next decimation, initialization with more samples is done by interpolation with current distribution on each curve
    if decim >= 1:
        
        shotx = np.array([])
        shoty = np.array([])
        
        for k in np.arange(0, nc):
            
            segment = kshot[ns_shot_decim_prev*k:(k+1)*ns_shot_decim_prev, :]  
            fx = interpolate.interp1d(np.linspace(-1/(2*np.pi),1/(2*np.pi), ns_shot_decim_prev), 
                                      segment[:,0], kind = 'linear')
            fy = interpolate.interp1d(np.linspace(-1/(2*np.pi),1/(2*np.pi), ns_shot_decim_prev), 
                                      segment[:,1], kind = 'linear')
            sx = fx(np.linspace(-1/(2*np.pi), 1/(2*np.pi), ns_shot_decim))
            sy = fy(np.linspace(-1/(2*np.pi), 1/(2*np.pi), ns_shot_decim))
            shotx = np.append(shotx, sx)
            shoty = np.append(shoty, sy)

        kshot = np.zeros((shotx.shape[0],2))
        kshot[:,0] = shotx
        kshot[:,1] = shoty
        
    algoProj.s = kshot

#Measure of elapsed time and results file name update
end = time.time()
wallClockM = (end - start)/60 #elapsed wall-clock time in minutes
scl = "%1.0f" %wallClockM
saveName = saveName + "_T" + scl + "Min" + ".npy"
print("%1.2f spent on Projected Gradient Descent" %wallClockM)

#save gradient samples (raster time sampling period)
np.save(saveName, kshot)
