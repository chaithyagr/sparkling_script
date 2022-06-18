#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu July 22 10:10:25 2019

@author: Chaithya G R  : chaithyagr@gmail.com
This file holds the basic constraint maps for setting up the constraint sets
The optimization is done with forwardBackward method with FISTA speed up
"""
# %%
import numpy as np
from modopt.opt.algorithms import ForwardBackward
import scipy.sparse.linalg as sciLinAlg
import scipy.sparse as sp
from joblib import Parallel, delayed
# %%
from copy import deepcopy

def star_from_data(data, actual_data, C_lin, C_kin, method="normal"):
    # Implementing eq 10 in https://www.ncbi.nlm.nih.gov/pubmed/27019479
    # Calculate Z = C - M.*q1 -M:*q2
    Atq_sum = np.zeros(actual_data.shape)
    for i in np.arange(len(C_kin)):
        Atq_sum += C_kin[i].opT(data[:, :, :, i])
    z = actual_data - Atq_sum
    s = []
    if C_lin.n_linear_constraints > 0:
        if method=="sparse":
            # Implementing a Sparse version of below commented equation,
            # as it is faster and we dont have many linear constraints
            A_sparse = sp.csc_matrix(C_lin.A)
            PI_sparse = sp.csc_matrix(C_lin.PI)
            s = z + PI_sparse*(C_lin.v - A_sparse*z)
        elif method == 'trivial':
            x, y = np.where(C_lin.A)
            s = deepcopy(z)
            s[:, y, :] += 0
        elif method=="normal":
            s = z + np.dot(C_lin.PI, (C_lin.v - np.dot(C_lin.A, z)))
        else:
            raise("Bad method chosen!")
        '''Below lines are to test the sparse methodm commenting now'''
        #if method!="normal":
        #    s2 = z + np.dot(C_lin.PI, (C_lin.v - np.dot(C_lin.A, z)))
        #    if not(np.allclose(s, s2)):
        #        raise("S is not equal to s2")
    else:
        s = z
    return s


class GradientL2(object):
    """Returns the gradient of F. Grad(F) = (-M. dq/dt ; -M.. d2q/dt2)

    Parameters
    ----------
    data: np.ndarray
        input 2D data array.
    """

    def __init__(self, actual_data, kinetic_constraints, linear_constraints, method="normal"):
        self.kinetic_constraints = kinetic_constraints
        self.linear_constraints = linear_constraints
        self.actual_data = actual_data
        self.method=method

    def get_s_star(self, data):
        """
        Function to obtain s* from s based on linear constraints in eq10 of 2016 paper
        :param data: s*
        :return: s_star
        """
        n_constraints = len(self.kinetic_constraints)
        # Calculate Z = C - M.*q1 -M:*q2
        self.s_star = star_from_data(data, self.actual_data, self.linear_constraints, self.kinetic_constraints, method=self.method)

    def get_grad(self, data):
        r"""Get the gradient

        This method calculates the gradient step from the input data

        Parameters
        ----------
        data : np.ndarray
            Input data array

        Notes
        -----
        Implements the following equation:

        .. math::
            \nabla F(x) = (-M. dq/dt ; -M.. d2q/dt2)

        """
        n_constraints = len(self.kinetic_constraints)
        self.get_s_star(data)
        self.grad = np.zeros(np.shape(data))
        for i in np.arange(n_constraints):
            self.grad[:, :, :, i] = -self.kinetic_constraints[i].op(self.s_star)


class ProximityL2(object):
    """Returns the proximity of G. G(q1,q2) = alpha*||q1||+|beta*||q2||)

    Parameters
    ----------
    alpha
        input array.
    """

    def __init__(self, kinetic_constraints):
        self.kinetic_constraints = kinetic_constraints

    def op(self, data, extra_factor):
        out_data = np.zeros((data.shape))
        # np.dot(C_lin.PI,(C_lin.v - np.dot(C_lin.A,z)))
        n_kinetic_constraints = len(self.kinetic_constraints)
        for i in np.arange(n_kinetic_constraints):
            out_data[:, :, :, i] = self.kinetic_constraints[i].f[3](data[:, :, :, i],
                                                                 extra_factor * self.kinetic_constraints[i].bound)
        return out_data


def project_fast_fista(C_kin, C_lin, algoParams, algoProj, nc, n_jobs=1):
    """ The Fast FISTA optimization for projection of K space vector.
    Parameters
    ----------

    Returns
    -------
    algoProj: ndarray
        the estimated FISTA solution.
    transform: a WaveletTransformBase derived instance
        the wavelet transformation instance.
    """
    s = algoProj.s
    algoProj.s = np.zeros(s.shape)
    n_pts = s.shape[0]
    n_segs = int(n_pts / nc)  # needs to be understood as an int for what follows
    dim = s.shape[1]
    s_per_c = np.reshape(s, (nc, n_segs, dim))
    si = Project_Curve_Affine_Constraints(s_per_c, C_kin, C_lin, algoParams)
    algoProj.s = np.reshape(np.asarray(si), (nc*n_segs, dim))
    return algoProj


def compute_Lipschitz_constant(C_kin, n_constraints, n, d):
    y = lambda X: C_kin[0].opT(C_kin[0].op(np.reshape(X, (n, d), order='F')))

    for k in np.arange(1, n_constraints + 1):
        y += lambda X: C_kin[k].opT(C_kin[k].op(np.reshape(X, (n, d), order='F')))

    yMat = np.fromfunction(y, (n, d), dtype=float)
    L = sciLinAlg.eigs(yMat, k=1, which="LM")
    L = abs(L)

    return L


def Project_Curve_Affine_Constraints(c, C_kin, C_lin, algoParams):
    nc = c.shape[0]
    ns = c.shape[1]
    d = c.shape[2]
    n_constraints = len(C_kin)
    grad_linear_method="trivial"
    # Obtain initial X
    X = np.zeros((nc, ns, d, n_constraints))
    for i in np.arange(0, n_constraints):
        X[:, :, :, i] = C_kin[i].op(c)
    # Setup the gradient operator (note that we obtain s_star here)
    grad_op = GradientL2(c, C_kin, C_lin, grad_linear_method)
    # Compute Lipschitz Constant
    L = algoParams.L
    beta = 1 / L
    # Start optimization
    opt = ForwardBackward(
        x=X,
        grad=grad_op,
        prox=ProximityL2(C_kin),
        cost=None,
        auto_iterate=False,
        beta_param=beta,
        progress=False
    )
    opt.iterate(algoParams.nit)
    # Store final value
    x_final = opt._x_new
    s = star_from_data(x_final, c, C_lin, C_kin, grad_linear_method)
    return s