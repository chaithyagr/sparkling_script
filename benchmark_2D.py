#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 17 14:47:53 2019

@author: nc258476
"""
import os
import os.path as op
import time
import numpy as np
import matplotlib.pyplot as plt
import sys
import matplotlib
import skimage.io as io
from skimage.measure import compare_ssim as ssim
from collections import namedtuple
from pynufft import NUFFT_cpu
import scipy.io as spio
import copy
import datetime
from pylab import *

#%%
#Loading of a specific mask
#get current working dir
cwd = os.getcwd()
dirimg_2d = op.join(cwd,"../data")
img_size = 512   #assuming a square image
FOV = 0.2 #field of view in meters
pixelSize = FOV/img_size

#load data file corresponding to the target resolution
filename = "BrainPhantom" + str(img_size) + ".png"
mri_filename = op.join(dirimg_2d, filename)
mri_img = io.imread(mri_filename, as_gray=True)
plt.figure()
plt.title("Numerical Brain Phantom, size = "+ str(img_size))
if mri_img.ndim == 2:
    plt.imshow(mri_img, cmap=plt.cm.gray)
else:
    plt.imshow(mri_img)
plt.show()

kshot = np.load("init.npy")

#%%
#Dsiplay the mask

k_spark = plt.figure(figsize=(10,10))
plt.scatter(kshot[:,0],kshot[:,1], marker = '.', s = 0.1)
plt.grid()

#Figure layout

unit = 1/4 ; tick = np.arange(-0.5, 0.5 + unit, unit)

label = [r"$-\frac{1}{2\pi}$", r"$-\frac{1}{4\pi}$", r"$0$", r"$+\frac{1}{4\pi}$",  r"$+\frac{1}{2\pi}$"]

plt.xticks(tick/np.pi,labels = label, fontsize = 16) ; plt.yticks(tick/np.pi,labels = label, fontsize = 16)

plt.xlabel(r"$k_x$", fontsize = 22) ; plt.ylabel(r"$k_y$", fontsize = 22)

plt.title("K-space SPARKLING sampling, radial in-out initialization",fontsize = 18)

plt.show()

#%%
#Reconstructions with NFFT

NufftObj_spark = NUFFT_cpu()

Nd = (img_size, img_size)  # image size
print('setting image dimension Nd...', Nd)
Kd = (img_size, img_size)  # k-space size
print('setting spectrum dimension Kd...', Kd)
Jd = (6, 6)  # interpolation size
print('setting interpolation size Jd...', Jd)

NufftObj_spark.plan(kshot*2*np.pi, Nd, Kd, Jd) #k_vec in [-0.5, 0.5]^2 for pynufft

mri_img = mri_img*1.0/np.amax(mri_img) #image normalization

recons, axes = plt.subplots(1,3)

kspace_spark_data = NufftObj_spark.forward(mri_img)
print('setting non-uniform data')
print('kspace is an (M,) list',type(kspace_spark_data), kspace_spark_data.shape)

## Conjugate gradient reconstruction
start1 = time.time()
image1 = NufftObj_spark.solve(kspace_spark_data, solver='cg',maxiter = 200)
end1 = time.time()

clock1 = (end1 - start1) #elapsed wall-clock time in seconds

ssim_recon = ssim(mri_img, image1,data_range = mri_img.max() - image1.min())
ssim_recon = float(round(abs(ssim_recon),4))

axes[0].set_title('Restored image (cg) : SSIM = %1.4f, t = %1.2f s' %(ssim_recon, clock1), fontsize = 10)
axes[0].imshow(image1.real, cmap=matplotlib.cm.gray, norm=matplotlib.colors.Normalize(vmin=0.0, vmax=1))

#L1TV OLS
start3 = time.time()
image3 = NufftObj_spark.solve(kspace_spark_data, solver='L1TVOLS', maxiter=200, rho=0.8)
end3 = time.time()

clock3 = (end3 - start3)

ssim_3 = ssim(mri_img, image3,data_range = mri_img.max() - image3.min())
ssim_3 = float(round(abs(ssim_3),4))

axes[1].set_title('Restored image (L1TV OLS) : SSIM = %1.4f, t = %1.2f s' %(ssim_3, clock3), fontsize = 10)
axes[1].imshow(image3.real, cmap=matplotlib.cm.gray,norm=matplotlib.colors.Normalize(vmin=0.0, vmax=1))

#L1TV LAD
start4 = time.time()
image4 = NufftObj_spark.solve(kspace_spark_data, solver='L1TVLAD', maxiter=200, rho = 2)
end4 = time.time()

clock4 = (end4 - start4)

ssim_4 = ssim(mri_img, image4,data_range = mri_img.max() - image4.min())
ssim_4 = float(round(abs(ssim_4),4))

axes[2].set_title('Restored image (L1TV LAD) : SSIM = %1.4f, t = %1.2f s' %(ssim_4, clock4), fontsize = 10)
axes[2].imshow(image4.real, cmap=matplotlib.cm.gray,norm=matplotlib.colors.Normalize(vmin=0.0, vmax=1))

fig = gcf()
plt.suptitle("SPARKLING with tau = 1, nCheb = 6 and spiralIO initialization ", fontsize = 12)

#%%
# fista rec using PySAP (branch pogm_addition: https://github.com/zaccharieramzi/pysap/tree/pogm_addition)

#from modopt.opt.linear import Identity
#from modopt.opt.proximity import SparseThreshold, LinearCompositionProx
#from pysap.numerics.fourier import NFFT
#from pysap import Image
#from pysap.numerics.gradient import GradAnalysis2
#from pysap.numerics.linear import Wavelet2
#from pysap.numerics.reconstruct import sparse_rec_pogm
#from pysap.numerics.utils import convert_mask_to_locations
##%%
### ops init
#startP = time.time()
#kspace_loc = convert_mask_to_locations(kshot*np.pi)
#linear_op = Wavelet2(
#    nb_scale=4,
#    wavelet_name="Db4",
#    padding_mode="periodization",
#)
#
#fourier_op = NFFT(
#    samples= kshot * np.pi,
#    shape= mri_img.shape,
#)
#
###compute the kspace data
#kspace_data_nfft = fourier_op.op(mri_img)
#
### now back to ops
#gradient_op = GradAnalysis2(
#    data=kspace_data_nfft,
#    fourier_op=fourier_op,
#)
#
## define the proximity operator
#prox_op = LinearCompositionProx(
#    linear_op=linear_op,
#    prox_op=SparseThreshold(Identity(), None, thresh_type="soft"),
#)
#
#if 1:
#    ## run pogm' (ie POGM with restart)
#    x_final, metrics = sparse_rec_pogm(prox_op=prox_op, gradient_op=gradient_op, im_shape=mri_img.shape,
#                                   mu=0.05, max_iter=100, xi_restart=0.96, metrics_={}, metric_call_period=20)
#
#pogm_rec = np.abs(x_final)
#img_rec = Image(data=pogm_rec)
#endP = time.time()
#
#clockP = endP - startP
##img_rec.show()
##img_rec = np.abs(x_final)
##print(metrics)
##SSIM
#
#ssim_pogm = ssim(mri_img, pogm_rec,data_range=mri_img.max() - pogm_rec.min())
#ssim_pogm = float(round(abs(ssim_pogm),4))
#
#axes[1,1].set_title('Restored image (L1TV LAD) : SSIM = %1.4f, t = %1.2f s' %(ssim_pogm, clockP), fontsize = 10)
#
##plt.figure()
##plt.title('Restored image (POGM) : SSIM = ' + str(ssim_pogm))
##plt.imshow(pogm_rec, cmap=matplotlib.cm.gray, norm=matplotlib.colors.Normalize(vmin=0.0, vmax=1))
##plt.show()