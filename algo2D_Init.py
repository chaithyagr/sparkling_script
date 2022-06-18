#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May  9 12:57:47 2019

@author: nc258476
"""
#%%
import numpy as np
import math

#%%
#Trajectory Initialization

def init_Trajectory(samplingOptions, ns_shot_decim,img_size, eps):
    
    if samplingOptions.init == "radialIO":
        
        print("Initalization for Radial In-Out")
        
        k_TE = math.ceil(samplingOptions.ns/2) #echo time
        k_TE_decim = math.ceil(k_TE/samplingOptions.decim)
        
        shot_c = np.arange(1,ns_shot_decim+1)
        shot_c = (-shot_c/(k_TE_decim-1) + k_TE_decim/(k_TE_decim-1))*(1/(2*np.pi))*(1-eps)
        shot_c = np.array(shot_c, dtype = np.complex_) ; 
        
        vecPhase = np.arange(0,samplingOptions.nc-1+1)
        
        shot = np.array([], dtype = np.complex_)    
        
        for k in vecPhase:
            shot = np.append(shot,shot_c*np.exp(2*np.pi*1j*k/(2*samplingOptions.nc)))
            
    elif samplingOptions.init == "radialCO":
        
        print("Initialization for Radial Center-Out")
        
        k_TE = 1
        k_TE_decim = 1
        
        shot_c = np.arange(0,ns_shot_decim)/(ns_shot_decim-1)*1/(2*np.pi)*(1-eps)
        shot = np.array([], dtype = np.complex_)
        
        vecPhase = np.arange(0,samplingOptions.nc)
        
        for k in vecPhase:
            shot = np.append(shot, shot_c*np.exp(2*np.pi*1j*k/samplingOptions.nc))
            
    
    elif samplingOptions.init == "spiralIO":
        
        print("Initialization for Spiral In-Out")
        
        ns_shot_decim_sym = (ns_shot_decim - 1)/2
        k_TE = math.ceil(samplingOptions.ns/2) #echo time
        k_TE_decim = math.ceil(k_TE/samplingOptions.decim)
        
        ic1 = np.arange(0,ns_shot_decim_sym+1, dtype = np.complex_)/(ns_shot_decim_sym)*1/(2*np.pi)*(1-eps)
        ic2 = np.exp(2*1j*np.pi*np.arange(0,ns_shot_decim_sym+1, dtype=np.complex_)/ns_shot_decim_sym*samplingOptions.n_revol)
        
        s00_c1 = np.multiply(ic1,ic2) ; 
        s00_c2 = -s00_c1[1:] 
        s00_c= np.append(np.flip(s00_c1, axis = 0), s00_c2)
        
        shot = np.array([], dtype = np.complex_) 
        
        for l in np.arange(0,samplingOptions.nc):
            shot = np.append(shot, s00_c*np.exp(1j*2*np.pi*l/(samplingOptions.nc*2)))
            
            
    elif samplingOptions.init == "spiralCO":
        
        print("Initialization for Spiral Center-Out")
        k_TE = 1
        k_TE_decim = 1
        
        vec = np.arange(0,ns_shot_decim, dtype = np.complex_)
        
        shot_c = vec/(ns_shot_decim-1)*1/(2*np.pi)*(1-eps)*np.exp(2*np.pi*1j*vec/(ns_shot_decim - 1)*samplingOptions.n_revol)
        shot = np.array([], dtype = np.complex_)
        vecPhase = np.arange(0,samplingOptions.nc)
        
        for k in vecPhase:
            shot = np.append(shot, shot_c*np.exp(2*np.pi*1j*k/samplingOptions.nc))
          
    elif samplingOptions.init == 'Cartesian':
        
        print("Initialization for Cartesian")
        
        k_TE = 0; k_TE_decim = 0 #to be passed as an output, but not used as no TE constraint in cartesian
        
        nc = samplingOptions.nc
        
        shot_c = np.arange(1,ns_shot_decim+1, dtype = np.complex_)
        shot_c = (-shot_c/(math.ceil(ns_shot_decim/2)-1))
        shot_c += math.ceil(ns_shot_decim/2)/(math.ceil(ns_shot_decim/2)-1)
        shot_c = shot_c *1/(2*np.pi)
        shot = np.array([],dtype = np.complex_)
        
        vecPhase = np.arange(1,nc+1, dtype = np.complex_)
        vecPhase = (-vecPhase/(math.ceil(nc/2)-1))
        vecPhase+= math.ceil(nc/2)/(math.ceil(nc/2)-1)
        vecPhase = vecPhase*1/(2*np.pi)
        
        for k in vecPhase:
            shotPhase =  shot_c + 1j*k*np.ones(len(shot_c))
            shot = np.append(shot, shotPhase)
            
    
    elif samplingOptions.init == 'Full_Cartesian':
        
        print("Initialization for Cartesian")
        
        k_TE = []; k_TE_decim = []
        
        shot_c = np.arange(1,img_size, dtype = np.complex_)
       
        shot_c= (-shot_c/(math.ceil(img_size/2)-1))
        shot_c +=  math.ceil(img_size/2)/(math.ceil(img_size/2)-1)
        shot_c = shot_c*1/(2*np.pi)
        shot = np.array([],dtype = np.complex_)
        
        vecPhase = shot_c*2*np.pi
        
        for k in vecPhase:
            shotPhase =  shot_c + 1j*1/(2*np.pi)*k*np.ones(len(shot_c))
            shot = np.append(shot, shotPhase)
        
            
    return shot, k_TE, k_TE_decim

#%%
#Function to mitigate eccessively high sampling rate at the center of the kspace

def Generate_Peak(nc,m,N,alpha,Dt,Dk0,tau):

    if alpha*Dt > Dk0:
        print("Decrease alpha or Dt so that alpha*Dt<=Dk0")
        alpha = Dk0*tau/Dt ; 
        
    x = np.linspace(-1/(2*np.pi), 1/(2*np.pi),N)
    X,Y = np.meshgrid(x,x)
    r = np.sqrt(np.power(X,2) + np.power(Y,2))
    p_peak = np.power(r,-1)
    
    omega = np.sqrt(2-2*np.cos(np.pi/nc)) # = 2sin(pi/(2*nc))
    jb = math.ceil((tau*Dk0)/(alpha*Dt*omega)) #max number of samples from the center of the disk
    print("jb = " + str(jb))
    rmin = jb*alpha*Dt #alpha*Dt is the distance covered between two samples at maximum speed
    print("rmin = " + str(rmin))
    
    p_peak[np.where(p_peak < 1/rmin)] = 0
    #p_peak = p_peak*(jb+1)/np.sum(p_peak)*nc/m #eq (3.10) and (3.11) in C.Lazarus' PhD (2018)
    p_peak = p_peak/np.sum(p_peak)*nc/m #eq (3.10) and (3.11) in C.Lazarus' PhD (2018) : missing jb+1 factor in Matlab?
    
    print("max p_peak is " + str(np.max(p_peak)))
    
    return p_peak

#%%
def generate_density(peak, decay, maxDens):
    
    #I = np.where(peak != 0)
    #Ic = np.where(peak == 0)
    I = np.where(peak >= maxDens)
    Ic = np.where(peak < maxDens)
    
    density = np.zeros(decay.shape)
    density[I] = peak[I]
    nor = 1 - np.sum(peak[I]) #normalization factor outside the peak disk

    n_elements = sum(len(row) for row in decay) #or p_decay.shape[0]*p_decay.shape[1]

    if maxDens*n_elements < 1:
        print("Threashold is too small, result is uniform")
        Qc = np.ones(len(peak[Ic])) #; print("dim of Qc is " +str(Qc.shape))
        Qc = Qc*nor/np.sum(Qc)
        density[Ic] = Qc

    decay_c = decay[Ic]/np.sum(decay[Ic])*nor

    ind = np.where(decay_c > maxDens) # ; print(type(ind[0]))
    indc = np.where(decay_c < maxDens)
    ind0 = np.where(decay_c == maxDens)

    Qc = decay_c

    while len(ind[0]) != 0:
    
        fact = (nor - maxDens*(len(ind0[0])+len(ind[0])))/np.sum(decay_c[indc])
        Qc[indc] = fact * decay_c[indc]
        Qc[ind] = maxDens
        Qc[ind0] = maxDens
        ind = np.where(decay_c > maxDens)
        indc = np.where(decay_c < maxDens)
        ind0 = np.where(decay_c == maxDens)

    density[Ic] = Qc
    
    return density