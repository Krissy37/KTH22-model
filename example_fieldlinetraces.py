# -*- coding: utf-8 -*-
"""
Created on Mon Sep 18 22:55:24 2023

@author: kristin
"""

import pandas as pd
import numpy as np 
from matplotlib import pyplot as plt, dates as mdates
from matplotlib.patches import Wedge

import sys
sys.path.append(' YOUR PATH ') # adapt this local path!
import kth22_model_for_mercury_v8 as kth
control_param_path = 'control_params_v8b.json'
fit_param_path = 'kth_own_cf_fit_parameters_opt_total_March23.dat'

# =============================================================================
# define contants
# =============================================================================

R_M = 2440 #in km, Mercury Radius 

x_start = np.array([1.1 ])* R_M
y_start = np.array([0.0])
z_start = np.array([-0.5])*R_M

r_hel = np.array([0.39])
di = np.array([50])

delta_t = 0.7 #step size for Runge Kutta Algorithm. Will be changed in the model 
              #automatically, depending on the magnetic field strength for speed 
              #improvement. 


fieldline = kth.trace_fieldline_v8(x_start, y_start, z_start, 
                                    r_hel, di, control_param_path, fit_param_path, 
                                    delta_t = delta_t)

#print('footpoint /end of calculated fieldline: )
#print('x_mso: ', (fieldline[0])[-1] )
#print(z_mso: ', (fieldline[2])[-1])

# =============================================================================
# plot 
# =============================================================================
theta1, theta2 = 90, 90 + 180
radius = 1
center = (0, 0)
m_day1 = Wedge(center, radius, theta1, theta2, fc='0.4', edgecolor='0.4')
m_night1 = Wedge(center, radius, theta2, theta1, fc='0.8', edgecolor='0.8')


lim = 3
title = 'fieldlinetrace'

fig = plt.figure(figsize=(6, 6))
fig.suptitle(title, fontsize=16)

#x-z
ax = plt.subplot(1, 1, 1)
ax.axis('square')             # creates square ax
plt.xlim((lim,-lim))          # set axis limits
plt.ylim((-lim,lim))
plt.plot(fieldline[0]/R_M, fieldline[2]/R_M, color = '0.1')
ax.add_artist(m_day1)   
ax.add_artist(m_night1)                                                              # creats grey circle (mercury)     
plt.xlabel(r'$X_{MSO}$' +' in '+'$R_M$')
plt.ylabel(r'$Z_{MSO}$' +' in '+'$R_M$')
plt.grid()

