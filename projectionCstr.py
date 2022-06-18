#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May  9 13:25:00 2019

@author: nc258476
"""
#%%
import numpy as np
import numpy.matlib
import matplotlib.pyplot as plt
import scipy.sparse.linalg as sciLinAlg

#%%

def Projector_On_Multiple_Curves_Center(C_kin, C_lin, algoParams, algoProj, nc):

    n_constraints = len(C_kin)
    s = algoProj.s
    algoProj.s = np.zeros(s.shape)
    n_pts = s.shape[0]
    n_segs = int(n_pts/nc) #needs to be understood as an int for what follows
    
    #First point at zero
    if not(hasattr(algoProj,"R")):
        R = np.array([])
        si = s[0:n_segs,:]
        C_lin.v[0,0] = 0
        C_lin.v[0,1] = 0
        si, ri = Project_Curve_Affine_Constraints(si, C_kin, C_lin, algoParams, R) #FIRE IN THE HOLE, ri different
        algoProj.s[0:n_segs,:] = si
        R_new = ri
    else:
        R = algoProj.R
        si = s[0:n_segs,:]
        ri = R[0:n_segs,:,:]
        C_lin.v[0,0] = 0
        C_lin.v[0,1] = 0
        si, ri = Project_Curve_Affine_Constraints(si, C_kin, C_lin, algoParams, ri)
        algoProj.s[0:n_segs,:] = si
        #R_new = np.append(R_new, ri, axis = 0)
        R_new = ri
    
    #Then, for each shot
    for i in np.arange(2,nc+1):
        
        if not(hasattr(algoProj,"R")):
            si = s[(i-1)*n_segs:i*n_segs,:]
            C_lin.v[0,0] = 0
            C_lin.v[0,1] = 0
            si, ri = Project_Curve_Affine_Constraints(si, C_kin, C_lin, algoParams, R)
        else:
            R = algoProj.R
            si = s[(i-1)*n_segs:i*n_segs,:]
            ri = R[(i-1)*n_segs:i*n_segs,:,:]
            si, ri = Project_Curve_Affine_Constraints(si, C_kin, C_lin, algoParams,ri)
        
        algoProj.s[(i-1)*n_segs:i*n_segs,:] = si
        R_new = np.append(R_new, ri, axis = 0)
    
    #end of loop for each shot
    algoProj.R = R_new
    return algoProj
    
#%%
def compute_Lipschitz_constant(C_kin, n_constraints, n, d):
    
    y = lambda X : C_kin[0].opT(C_kin[0].op(np.reshape(X,(n,d), order = 'F')))
    
    for k in np.arange(1,n_constraints+1):
        y += lambda X : C_kin[k].opT(C_kin[k].op(np.reshape(X,(n,d), order = 'F')))
    
    yMat = np.fromfunction(y, (n,d), dtype = float)
    L = sciLinAlg.eigs(yMat,k=1,which= "LM")
    L = abs(L)
    
    return L

#%%
def Project_Curve_Affine_Constraints(s, C_kin, C_lin, algoParams, R):
    
    n = s.shape[0]
    d = s.shape[1]
    n_constraints = len(C_kin)
    
#    if algoParams.disp_results:
#        CF = np.zeros(algoParams.nit)
        
    #COMPUTE LIPSCHITZ CONSTANT   
    if not(hasattr(algoParams, 'L')):
        L  = compute_Lipschitz_constant(C_kin, n_constraints, n, d)
    else:
        L = algoParams.L
    tau = 1/L
    
    #COMPUTE DISTANCE TO CONSTRAINTS
#    if algoParams.disp_results:
#        D = np.zeros((n_constraints, algoParams.nit))
#        for k in np.arange(0,n_constraints):
#            D[k,0] = np.max(C_kin[k].f[0](C_kin[k].op(s)-C_kin[k].bound,0))
            
    #PRECOMPUTE As
    As = np.zeros((s.shape[0], s.shape[1], n_constraints))
    for i in np.arange(0,n_constraints):
        As[:,:,i] = C_kin[i].op(s)
    
    #INITIALIZE DUAL VARIABLES
    if R != np.array([]):
        if R.size == s.size*n_constraints:
            Q = R
        else:
            Q = np.zeros(As.shape)
            R = Q
            print("Input dual variables have wrong size, they are set to zero")
    else:
        Q = np.zeros(As.shape)
        R = Q
 ######################################################################################
 #ALGORITHM
    
    #Proximal gradient descent
    for k in np.arange(1,algoParams.nit+1):
        
        if k > 1:
            Atq_sum = Atq_sum_last
        else:
            Atq_sum = np.zeros(s.shape)
            for i in np.arange(0,n_constraints):
                Atq_sum = Atq_sum + C_kin[i].opT(Q[:,:,i])
                
        #COMPUTE COST FUNCTION
#        if algoParams.disp_results:
#            if C_lin.n_linear_constraints > 0:
#                z = s - Atq_sum
#                s_star = z +np.dot(C_lin.PI,(C_lin.v - np.dot(C_lin.A,z))) #eq 10 in Projection algo Chauffert et al.
#                CF[k] = np.sum(np.multiply(s_star,Atq_sum))+0.5*np.sum(s_star-s)
#                
#                for i in np.arange(0,n_constraints):
#                    CF[k] = CF[k]-C_kin[i].bound*C_kin[i].fct[2](Q[:,:,i])
#            else:
#                for i in np.arange(0,n_constraints):
#                    CF[k] = CF[k]-C_kin[i].bound*C_kin[i].fct[2](Q[:,:,i])
#                CF[k] = CF[k] - 0.5*np.power(np.linalg.norm(Atq_sum),2) + np.sum(s*Atq_sum)
        
        #COMPUTE NEW ITERATE
        R_prev = np.copy(R) #copy, otherwise when R is modified so is R_prev!
        z = s - Atq_sum
        
        if C_lin.n_linear_constraints > 0:
            s_star = z + np.dot(C_lin.PI,(C_lin.v - np.dot(C_lin.A,z)))
        else:
            s_star = z
        
        #C_kin [i].f[3] is the proximal operator defined in contraints_sets.py
        for i in np.arange(0,n_constraints):
            R[:,:,i] = C_kin[i].f[3](Q[:,:,i] + tau*(C_kin[i].op(s_star)),tau*C_kin[i].bound) #proximal operator in Algorithm 1
            
        Q = R + (k-1)/(k+2)*(R-R_prev) #Projection algorithm 1, second equation (Chauffert et al., 2016)
        
        # Needed to compute next iterate (after eq (10) before the proof)
        Atq_sum = np.zeros(s.shape, dtype = np.float64)
        
        for i in np.arange(0,n_constraints):
            Atq_sum = Atq_sum + C_kin[i].opT(Q[:,:,i])
            
        Atq_sum_last = Atq_sum
        
        #COMPUTE DIST TO CONSTRAINTS
#        if algoParams.disp_results:
#            for i in np.arange(0,n_constraints):
#                D[i,k] = np.max(C_kin[i].f[0](C_kin[i].op(s)-C_kin[i].bound,0))
        
        #SHOW ALGORITHM PROGRESSION
         
        #TODO
        
        #end of Proximal Gradient Descent
        
    #COMPUTE THE OUTPUT
    z = s - Atq_sum
    
    if C_lin.n_linear_constraints > 0:
        s = z + np.dot(C_lin.PI,(C_lin.v - np.dot(C_lin.A,z)))
    else:
        s = z
    
    return s, R    