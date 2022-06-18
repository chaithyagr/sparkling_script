#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May  9 13:10:25 2019

@author: nc258476
"""
#%%
import numpy as np
#import np.matlib
import math

#%%
class kinetic_Constraint:
    
    def __init__(self, function, operator, operatorT, bound):    
        self.f = function
        self.op = operator
        self.opT = operatorT
        self.bound = bound         

#The set of parametrized curves s for which As = v
class linear_Constraint :

    def __init__(self):
        self.A = [] #matrix that defines the set of Affine constraints
        self.AT = [] #A transposed
        self.v = [] # encodes the affine constraints
        self.PI = [] #pseudo inverse of A
        self.n_linear_constraints = 0
        
#%%
def proxL2(x, alphas):
    norm2_vec = np.linalg.norm(x, axis=2)
    epsilon = 1e-10
    epsVec = epsilon * np.ones(norm2_vec.shape)
    max1 = np.max(np.vstack([epsVec[np.newaxis], norm2_vec[np.newaxis]]),
                  axis=0)
    max2 = np.zeros(x.shape)
    for i in range(x.shape[2]):
        norm_alpha = norm2_vec - alphas
        max2[:, :, i] = norm_alpha * (norm_alpha > 0)
    y = np.multiply(
        np.divide(x, np.repeat(max1[:, :, np.newaxis], x.shape[2], axis=2)),
        max2
    )
    return y

def bound_normL2():
    norm2_vec = lambda x : np.sqrt(np.sum(np.power(x,2),axis = 1))
    infLim = lambda x : np.max(norm2_vec(x))
    dualL2 = lambda x : np.sum(norm2_vec(x))    
    proxD = lambda x,alpha : proxL2(x,alpha)
    return norm2_vec, infLim, dualL2, proxD

def Prime(s, Dt):
    first_derivative = np.zeros(s.shape)
    first_derivative[:, 1:, :] = np.diff(s, axis=1)
    first_derivative = first_derivative / Dt
    return first_derivative


def PrimeT(s, Dt):
    first_derivative = np.zeros(s.shape)
    first_derivative[:, 0, :] = -s[:, 1, :]
    first_derivative[:, 1:-1, :] = - np.diff(s[:, 1:, :], axis=1)
    first_derivative[:, -1, :] = s[:, -1, :]
    first_derivative = first_derivative / Dt
    return first_derivative


def second_derivative(data, dwell_time=1):
    first_d = Prime(data, dwell_time)
    second_d = PrimeT(first_d, dwell_time)
    return second_d

#%%
def set_lin_constraints_kTE(L_Cstr, n, d, *args):
    
    #d the dimension of the signal
    p = len(args)/2
    A = np.empty([0,n], dtype = float) #necessary for np.vstack
    v = np.empty([0,2], dtype = float)
    n_linear_constraints = 0
    
    if len(args) != 0:
        for k in np.arange(0,len(args),2):
            n_linear_constraints += 1
            
            if args[k] == "start_point":
                s0 = args[k+1]
                A = np.vstack((A, np.append(np.array([1]), np.zeros(n-1))))
                v = np.vstack((v, s0))
            elif args[k] == "end_point":
                se = args[k+1]
                A = np.vstack((A, np.append(np.zeros(n-1),np.array([1]))))
                v = np.vstack((v,se))
            elif args[k] == "gradient_moment_nulling":
                moment = args[k+1]
                T_D = lambda x : PrimeT(x,1)
                A = np.vstack((A, T_D(np.power(np.arange(1,n+1),moment))))
                v = np.vstack((v, np.zeros(d)))
            elif args[k] == "curve_splitting":
                TR = args[k+1]
                L = math.floor(n/TR)
                M = np.zeros(k,n)
                for j in np.arange(0,L):
                    M[j,(j-1)*TR+1] = 1
                A = np.vstack((A,M))
                v = np.vstack((v,np.zeros(L,d)))
            elif args[k] == "TE_point":
                sTE = args[k+1]
                B = np.hstack((np.zeros(sTE-1),np.array([1]),np.zeros(n-sTE)))
                A = np.vstack((A, B))
                v = np.vstack((v, np.array([0,0])))
            else:
                print("ERROR : Invalid arguments" + str(args[k]))
                    
    L_Cstr.n_linear_constraints = n_linear_constraints
    
    if n_linear_constraints > 0:
        U, indices = np.unique(A, axis = 0, return_index = True)
        A = A[indices, :]
        prodA = np.transpose(np.multiply(A,A))
        norm = np.transpose(np.sqrt(np.sum(prodA, axis = 0))) #row vector of the norm of columns
        #L_Cstr.A = np.divide(A,np.matlib.repmat(norm,1,n))
        L_Cstr.A = np.divide(A, np.tile(norm, (1,n)))
        L_Cstr.AT = np.transpose(L_Cstr.A)
        #L_Cstr.v = np.divide(v[indices,:],np.matlib.repmat(norm,1,d))
        L_Cstr.v = np.divide(v[indices,:], np.tile(norm, (1,d)))
        AAT = np.dot(L_Cstr.A, L_Cstr.AT)
        L_Cstr.PI = np.dot(L_Cstr.AT, np.linalg.inv(AAT)) #A+ in Chauffert et al., 2016
               