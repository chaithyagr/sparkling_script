#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May  9 13:35:26 2019

@author: nc258476
"""
# %%
import numpy as np
import numpy.linalg as LA
import scipy
from collections import namedtuple
import projectionOp as projOp
import bbfmm as fmm
import progressbar

# %%
class Kernel:

    def __init__(self, name):

        if name == "dist_L2":
            self.function = lambda x, y: np.sqrt(np.power(x, 2) + np.power(y, 2))
            self.grad_x = lambda x, y: x / np.maximum(np.sqrt(np.power(x, 2) + np.power(y, 2)), 1e-16)
            self.grad_y = lambda x, y: y / np.maximum(np.sqrt(np.power(x, 2) + np.power(y, 2)), 1e-16)

        elif name == "log_L2":
            self.function = lambda x, y: 0.5 * np.log(np.power(x, 2) + np.power(y, 2))
            self.grad_x = lambda x, y: x / np.maximum(np.power(x, 2) + np.power(y, 2), 1e-16)
            self.grad_y = lambda x, y: y / np.maximum(np.power(x, 2) + np.power(y, 2), 1e-16)

        elif name == "gaussian":
            self.funtion = lambda x, y: np.exp(-(np.power(x, 2) + np.power(y, 2)))
            self.grad_x = lambda x, y: -2 * x * np.exp(-(np.power(x, 2) + np.power(y, 2)))
            self.grad_y = lambda x, y: -2 * y * np.exp(-(np.power(x, 2) + np.power(y, 2)))

        else:
            print('You have not unlocked this kernel yet')


def fft(x):
    return np.fft.rfft2(x)


def ifft(x):
    return np.fft.irfft2(x)


# %%
def gradDescent(density, decim, shot, C_kin, C_lin, x_att, y_att, Att_field, kernelName, samplingOptions, tol, k_TE,
                stepDef, algoParams):
    # INITIALIZATION
    shot_prev = np.copy(shot)
    nc = samplingOptions.nc
    nCheb = algoParams.nCheb
    kernelID = algoParams.kernelID

    # Structure algoProj reinitialized for each decimation step + first projection
    algoProj = namedtuple("algoProj", "s")
    algoProj.s = np.copy(shot)
    algoProj = projOp.project_fast_fista(C_kin, C_lin, algoParams, algoProj, nc)
    # algoProj = proj_Cstr.Projector_On_Multiple_Curves_Center(C_kin, C_lin, algoParams, algoProj, nc)
    shot = np.copy(algoProj.s)

    # PRECOMPUTATION
    # Att_field can be computed only onces as it depends on VDS, so it it an input of the gradient descent

    # Main algorithm
    print("Started subgradient descent")
    with progressbar.ProgressBar(max_value=samplingOptions.gradit) as bar:
        for k in np.arange(0, samplingOptions.gradit):
            if samplingOptions.Shaking == 1:
                bar.update(k)
                max_shaking = (np.log(3 * samplingOptions.gradit) / np.log(2) - 6) / 10 * np.exp(
                    (-3 * (k + 1)) / samplingOptions.gradit) * 5
                if max_shaking < 0:
                    max_shaking = 0
                angle = 2 * np.pi * np.random.rand(shot.shape[0])
                magnitude = 2 * max_shaking * np.random.rand(shot.shape[0])
                AM = np.vstack((magnitude * np.cos(angle), magnitude * np.sin(angle)))
                shot += 2 * (1 / (2 * np.pi)) / shot.shape[0] * np.transpose(AM)
                shot[np.where(shot < -1 / (2 * np.pi))] = -1 / (2 * np.pi)
                shot[np.where(shot > 1 / (2 * np.pi))] = 1 / (2 * np.pi)

            if k >= 1:
                fprev = np.copy(forces)

            # COMPUTE SUBGRADIENTS

            if samplingOptions.kTE_crossing == 0:
                print("kTE crossing should be 1")
            else:
                shot_temp = np.copy(shot)
                # remove k_TE points for calculation except for first segment (one point in center)
                shot_temp = np.delete(shot_temp,
                                      np.arange(k_TE - 1 + shot.shape[0] // nc, shot.shape[0], shot.shape[0] // nc), axis=0)

                # ATTRACTION FORCE
                dFa = np.zeros((int(shot.shape[0] / nc), nc, 2))
                dFa_temp = Attraction(shot_temp, x_att, y_att, Att_field)
                dFa_temp = dFa_temp.real
                dFa_center_point = dFa_temp[k_TE - 1, :]
                dFa_temp = np.delete(dFa_temp, k_TE - 1, axis=0)
                dFa_temp_reshape = np.reshape(dFa_temp, (int(dFa_temp.shape[0] / nc), nc, 2), order='F')

                if k_TE > 1:
                    dFa[0:k_TE - 1, :, :] = dFa_temp_reshape[0:k_TE - 1, :, :]

                dFa[k_TE:, :, :] = dFa_temp_reshape[k_TE - 1:, :, :]
                dFa[k_TE - 1, 0, :] = dFa_center_point
                dFa = np.reshape(dFa, (shot.shape[0], 2), order='F')

                # REPULSION FORCE

                # N-body problems to compute the repulsive gradients are calculated separately (deprecated as "multiple" works)
                if algoParams.mode == 0:
                    N = shot_temp.shape[0]
                    dFr = np.zeros((int(shot.shape[0] / nc), nc, 2))
                    H = np.ones(shot_temp.shape[0])
                    Qh = fmm.single(shot_temp.T, H, kernelID, nCheb)  ##bbfmm_2D takes 2*N arrays
                    Qkx = fmm.single(shot_temp.T, shot_temp[:, 0], kernelID, nCheb)
                    Qky = fmm.single(shot_temp.T, shot_temp[:, 1], kernelID, nCheb)

                    # Equation (15) in sparkling 3D paper (Lazarus et al.)
                    dFr_temp = np.zeros(shot_temp.shape)
                    leftTerm_x = shot_temp[:, 0] * Qh[:, 0]
                    leftTerm_y = shot_temp[:, 1] * Qh[:, 0]

                    dFr_temp[:, 0] = 1 / N * (leftTerm_x - Qkx[:, 0])
                    dFr_temp[:, 1] = 1 / N * (leftTerm_y - Qky[:, 0])

                else:  # all N-body problems needed are calculated with one call
                    N = shot_temp.shape[0]
                    dFr = np.zeros((int(shot.shape[0] / nc), nc, 2))
                    Qh = fmm.multiple(shot_temp.T, nCheb)

                    # Equation (15) in sparkling 3D paper (Lazarus et al., 2019)
                    dFr_temp = np.zeros(shot_temp.shape)
                    leftTerm_x = shot_temp[:, 0] * Qh[0:N, 0]
                    leftTerm_y = shot_temp[:, 1] * Qh[0:N, 0]

                    dFr_temp[:, 0] = 1 / N * (leftTerm_x - Qh[N:2 * N, 0])
                    dFr_temp[:, 1] = 1 / N * (leftTerm_y - Qh[2 * N:, 0])

                dFr_center_point = dFr_temp[k_TE - 1, :]
                dFr_temp = np.delete(dFr_temp, k_TE - 1, axis=0)
                dFr_temp_reshape = np.reshape(dFr_temp, (int(dFr_temp.shape[0] / nc), nc, 2), order='F')

                if k_TE > 1:
                    dFr[0:k_TE - 1, :, :] = dFr_temp_reshape[0:k_TE - 1, :, :]

                dFr[k_TE:, :, :] = dFr_temp_reshape[k_TE - 1:, :, :]
                dFr[k_TE - 1, 0, :] = dFr_center_point
                dFr = np.reshape(dFr, (shot.shape[0], 2), order='F')

                # grad(Attraction - Repulsion)
            forces = dFa - dFr

            # Step size for gradient descent
            if k < 5:
                alpha = stepDef
            else:
                dForce = forces - fprev
                dShot = shot - shot_prev
                alpha = np.sum(dForce * dShot) / (np.power(LA.norm(dForce), 2))  # Barzilai-Borwein rule

            shot_prev = np.copy(shot)
            shot = shot - alpha * forces

            # The algorithm alternates between gradient descent for SPARKLING and projection
            algoProj.s = np.copy(shot)
            algoProj = projOp.project_fast_fista(C_kin, C_lin, algoParams, algoProj, nc)  # using modopt and greedy FISTA
            # algoProj = proj_Cstr.Projector_On_Multiple_Curves_Center(C_kin, C_lin, algoParams, algoProj, nc)
            shot = np.copy(algoProj.s)
            shot[np.where(shot < -1 / (2 * np.pi))] = -1 / (2 * np.pi)
            shot[np.where(shot > 1 / (2 * np.pi))] = 1 / (2 * np.pi)

    return shot


# %%
def Precompute_Attraction_Field(density, kernelName):
    # In most cases n1 = n2
    n1 = density.shape[0]
    n2 = density.shape[1]
    kernel = Kernel(kernelName)

    x = np.linspace(-2 / (2 * np.pi), 2 / (2 * np.pi), 2 * n2)
    y = np.linspace(-2 / (2 * np.pi), 2 / (2 * np.pi), 2 * n1)

    X, Y = np.meshgrid(x, y)

    h1 = np.fft.fftshift(kernel.grad_x(X, Y))
    h2 = np.fft.fftshift(kernel.grad_y(X, Y))

    out = np.zeros((2 * n1, 2 * n2, 2))
    fpi = np.zeros((2 * n1, 2 * n2))

    # Put the density into a "finer" grid and center it before computing the FFT
    fpi[0:n1, 0:n2] = density
    fpi = np.roll(fpi, int(n1 / 2) + 1, axis=0)
    fpi = np.roll(fpi, int(n2 / 2) + 1, axis=1)
    fpi = fft(fpi)

    # Convulotion using FFT
    out[:, :, 0] = ifft(fpi * fft(h1))
    out[:, :, 1] = ifft(fpi * fft(h2))

    #    outplot1 = plt.figure(figsize = (10,10))
    #    ax1 = outplot1.gca(projection='3d')
    #    surf = ax1.plot_surface(X, Y, out[:,:,0], cmap=cm.coolwarm,linewidth=0, antialiased=False)
    #    outplot1.colorbar(surf, shrink=0.5, aspect=5)
    #    plt.title("Att field 0")
    #    plt.show()
    #
    #    outplot2 = plt.figure(figsize = (10,10))
    #    ax2 = outplot2.gca(projection='3d')
    #    surf = ax2.plot_surface(X, Y, out[:,:,1], cmap=cm.coolwarm,linewidth=0, antialiased=False)
    #    outplot2.colorbar(surf, shrink=0.5, aspect=5)
    #    plt.title("Att field 1")
    #    plt.show()

    return out, x, y  # vectors (instead of meshgrid components) for RectBivariateSpline Interpolation


# %%
def Attraction(shot, x_att, y_att, Att_field):
    sub_s = np.zeros(shot.shape)
    fax = scipy.interpolate.RectBivariateSpline(x_att, y_att, Att_field[:, :, 0])
    fay = scipy.interpolate.RectBivariateSpline(x_att, y_att, Att_field[:, :, 1])
    sub_s[:, 0] = fax.ev(shot[:, 1], shot[:, 0])
    sub_s[:, 1] = fay.ev(shot[:, 1], shot[:, 0])

    # This more "intuitive" calculation is false (samples distributed along a line at the end)
    # sub_s[:,0] = fax.ev(shot[:,0], shot[:,1])
    # sub_s[:,1] = fay.ev(shot[:,0], shot[:,1])

    return sub_s
