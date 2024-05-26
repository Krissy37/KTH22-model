# -*- coding: utf-8 -*-
"""
Created on Mon Sep 18 14:09:23 2023

@author: kristin
"""
import pandas as pd
import numpy as np 
from matplotlib import pyplot as plt, dates as mdates
from matplotlib.patches import Wedge

import sys
sys.path.append(' Add your path here') # adapt this local path!
import kth22_model_for_mercury_v8 as kth
control_param_path = 'control_params_v8b.json'
fit_param_path = 'kth_own_cf_fit_parameters_opt_total_March23.dat'

# =============================================================================
# define contants
# =============================================================================

R_M = 2440 #in km, Mercury Radius 
    
# =============================================================================
# load data
# =============================================================================
orbit_number = 32
orbit_number_str = str(int(orbit_number))

print('load dataframe')

df = pd.read_csv('orbit_' + orbit_number_str + '.csv')


x_mso_km = df['x_mso_km'].to_numpy()
y_mso_km = df['y_mso_km'].to_numpy()
z_mso_km = df['z_mso_km'].to_numpy() 

r_hel_AU = df['r_hel_AU'].to_numpy()
di = df['di'].to_numpy() 

bx_mes_nT = df['bx_mes_nT'].to_numpy() 
by_mes_nT = df['by_mes_nT'].to_numpy()  
bz_mes_nT = df['bz_mes_nT'].to_numpy()  

dates_mes = pd.to_datetime(df['time_YYYYMMDDHHmm']).to_numpy() 

bx_KTH_nT = np.zeros(len(x_mso_km)) * np.nan
by_KTH_nT = np.zeros(len(x_mso_km)) * np.nan
bz_KTH_nT = np.zeros(len(x_mso_km)) * np.nan


# =============================================================================
#  calculate B from KTH22 model 
# =============================================================================

#call KTH 
'''
b_KTH_nT = kth.kth22_model_for_mercury_v8(  YOUR TASK: INSERT INPUT )

bx_KTH_nT = b_KTH_nT[0]
by_KTH_nT = b_KTH_nT[1]
bz_KTH_nT = b_KTH_nT[2]
'''

# =============================================================================
# filter for indices inside magnetopause
# =============================================================================
#turn filter_mag = True if you only want to plot the data inside the calculated magnetosphere 
#otherwise the data of the whole orbit is shown 

filter_mag = False

if filter_mag == True: 

    filtered_indices = np.where(np.isfinite(bx_KTH_nT))
    
    
    #length_check_filtered_indices = len(filtered_indices[0])
    
    x_mso_km = x_mso_km[filtered_indices]
    y_mso_km = y_mso_km[filtered_indices]
    z_mso_km = z_mso_km[filtered_indices]
    
    bx_mes_nT = bx_mes_nT[filtered_indices]
    by_mes_nT = by_mes_nT[filtered_indices]
    bz_mes_nT = bz_mes_nT[filtered_indices]
    
    bx_KTH_nT = bx_KTH_nT[filtered_indices]
    by_KTH_nT = by_KTH_nT[filtered_indices]
    bz_KTH_nT = bz_KTH_nT[filtered_indices]
    
    r_hel_AU = r_hel_AU[filtered_indices]
    di = di[filtered_indices]
    dates_mes = dates_mes[filtered_indices]



# =============================================================================
# calc res
# =============================================================================


delta_Bx = bx_mes_nT - bx_KTH_nT
delta_By = by_mes_nT - by_KTH_nT
delta_Bz = bz_mes_nT - bz_KTH_nT

res_total = np.sqrt(delta_Bx**2 + delta_By**2 + delta_Bz**2)
delta_B_mean = np.round(np.nanmean(res_total),2)
delta_B_median = np.round(np.nanmedian(res_total),2)

print('Total residuum (mean)    for orbit ' + orbit_number_str + ': ' + str(delta_B_mean) + ' nT')
print('Total residuum (median)  for orbit ' + orbit_number_str + ': ' + str(delta_B_median) + ' nT')        




# Create grey circle to show size of Mercury 
mercury3 = plt.Circle((0, 0), 1, color = '0.4')

theta1, theta2 = 90, 90 + 180
radius = 1
center = (0, 0)
m_day1 = Wedge(center, radius, theta1, theta2, fc='0.4', edgecolor='0.4')
m_night1 = Wedge(center, radius, theta2, theta1, fc='0.8', edgecolor='0.8')

m_day2 = Wedge(center, radius, theta1, theta2, fc='0.4', edgecolor='0.4')
m_night2 = Wedge(center, radius, theta2, theta1, fc='0.8', edgecolor='0.8')   

# Create grey circle to show size of Mercury 
mercury3b = plt.Circle((0, 0), 1, color = '0.4')

theta1, theta2 = 90, 90 + 180
radius = 1
center = (0, 0)
m_day1b = Wedge(center, radius, theta1, theta2, fc='0.4', edgecolor='0.4')
m_night1b = Wedge(center, radius, theta2, theta1, fc='0.8', edgecolor='0.8')

m_day2b = Wedge(center, radius, theta1, theta2, fc='0.4', edgecolor='0.4')
m_night2b = Wedge(center, radius, theta2, theta1, fc='0.8', edgecolor='0.8')

###############################################################################
# find indeces for under/above  -> grey/black
###############################################################################
'''
indices_x = np.where(x_mso_km >= 0)
indices_y = np.where(y_mso_km >= 0)
indices_z = np.where(z_mso_km >= -(479/2440))
'''


# =============================================================================
# closest approach
# =============================================================================
r = np.sqrt(x_mso_km**2 + y_mso_km**2 + z_mso_km**2)
r_R_M = r/R_M



title = r'Orbit ' + orbit_number_str + ', DI = ' + str(int(np.unique(di)[0])) + ', r$_{hel}$ = ' + str(np.round(np.unique(r_hel_AU)[0],2))
#title2 =   r'ca: ' + str(np.round(np.amin(r_R_M),2)) + ' R$_M$, $\Delta$B (mean) = ' + str(delta_B_mean) + ' nT, $\Delta$B (median) = ' + str(delta_B_median)
title2 = ''
str_date_YMD = str(dates_mes[0])[0:10]     

###############################################################################
# plot
###############################################################################
lim = 3

fig = plt.figure(figsize=(11.5, 15))
fig.suptitle(title + '\n' + '\n' + title2, fontsize=16)
#plt.title(title2, fontsize=16)

#x-z
ax = plt.subplot(4, 3, 1)
ax.axis('square')             # creates square ax
plt.xlim((lim,-lim))          # set axis limits
plt.ylim((-lim,lim))
plt.plot(x_mso_km/R_M, z_mso_km/R_M, color = '0.1')
ax.add_artist(m_day1)   
ax.add_artist(m_night1)                                                              # creats grey circle (mercury)     
plt.xlabel(r'$X_{MSM}$' +' in '+'$R_M$')
plt.ylabel(r'$Z_{MSM}$' +' in '+'$R_M$')
plt.grid()

#x-y
ax = plt.subplot(4, 3, 2)
plt.plot(x_mso_km/R_M, y_mso_km/R_M, color = '0.1')
ax.add_artist(m_day2)   
ax.add_artist(m_night2)
ax.axis('square')
plt.xlim((lim,-lim))
plt.ylim((-lim,lim))
plt.xlabel(r'$X_{MSM}$' +' in '+'$R_M$')
plt.ylabel(r'$Y_{MSM}$' +' in '+'$R_M$')
plt.grid()

#y-z
ax = plt.subplot(4, 3, 3)
plt.plot(y_mso_km/R_M, z_mso_km/R_M, color = '0.1')
ax.add_artist(mercury3)
ax.axis('square')
plt.xlim((lim,-lim))
plt.ylim((-lim,lim))
plt.grid()
plt.xlabel(r'$Y_{MSM}$' +' in '+'$R_M$')
plt.ylabel(r'$Z_{MSM}$' +' in '+'$R_M$')

#Bx
ax = plt.subplot(4, 1, 2)
formatter = mdates.DateFormatter("%H:%M")
ax.xaxis.set_major_formatter(formatter)
plt.plot(dates_mes, bx_mes_nT, label = 'MESSENGER Data')
plt.plot(dates_mes, bx_KTH_nT, label = 'KTH Model Version 8')
plt.legend()
plt.grid()

#By
plt.ylabel('Bx in nT')
ax = plt.subplot(4, 1, 3)
formatter = mdates.DateFormatter("%H:%M")
ax.xaxis.set_major_formatter(formatter)
plt.plot(dates_mes, by_mes_nT, label = 'MESSENGER Data')
plt.plot(dates_mes, by_KTH_nT, label = 'KTH Model')
plt.grid()

#Bz
plt.ylabel('By in nT')
ax = plt.subplot(4, 1, 4)
formatter = mdates.DateFormatter("%H:%M")
ax.xaxis.set_major_formatter(formatter)
plt.plot(dates_mes, bz_mes_nT,label = 'MESSENGER Data')
plt.plot(dates_mes, bz_KTH_nT,label = 'KTH Model')
plt.ylabel('Bz in nT')
plt.xlabel('time in HH:MM on ' + str_date_YMD)
plt.grid()


plt.show()


#filename_pdf = 'orbit_' + str_orbit_number + '_overview.png'
#plt.savefig(filename_pdf, orientation = 'portrait', dpi = 200)


# =============================================================================
# create same plot with axis limits 
# =============================================================================
'''

fig = plt.figure(figsize=(11.5, 15))
fig.suptitle(title + '\n' + '\n' + title2, fontsize=16)
#plt.title(title2, fontsize=16)

#x-z
ax = plt.subplot(4, 3, 1)
ax.axis('square')             # creates square ax
plt.xlim((lim,-lim))          # set axis limits
plt.ylim((-lim,lim))
plt.plot(x_mso_km/R_M, z_mso_km/R_M, color = '0.1')
ax.add_artist(m_day1b)   
ax.add_artist(m_night1b)                                                              # creats grey circle (mercury)     
plt.xlabel(r'$X_{MSM}$' +' in '+'$R_M$')
plt.ylabel(r'$Z_{MSM}$' +' in '+'$R_M$')
plt.grid()

#x-y
ax = plt.subplot(4, 3, 2)
plt.plot(x_mso_km/R_M, y_mso_km/R_M, color = '0.1')
ax.add_artist(m_day2b)   
ax.add_artist(m_night2b)
ax.axis('square')
plt.xlim((lim,-lim))
plt.ylim((-lim,lim))
plt.xlabel(r'$X_{MSM}$' +' in '+'$R_M$')
plt.ylabel(r'$Y_{MSM}$' +' in '+'$R_M$')
plt.grid()

#y-z
ax = plt.subplot(4, 3, 3)
plt.plot(y_mso_km/R_M, z_mso_km/R_M, color = '0.1')
ax.add_artist(mercury3b)
ax.axis('square')
plt.xlim((lim,-lim))
plt.ylim((-lim,lim))
plt.grid()
plt.xlabel(r'$Y_{MSM}$' +' in '+'$R_M$')
plt.ylabel(r'$Z_{MSM}$' +' in '+'$R_M$')

#Bx
ax = plt.subplot(4, 1, 2)
formatter = mdates.DateFormatter("%H:%M")
ax.xaxis.set_major_formatter(formatter)
plt.plot(dates_mes, bx_mes_nT, label = 'MESSENGER Data')
plt.plot(dates_mes, bx_KTH_nT, label = 'KTH Model Version 8')
plt.legend()
plt.grid()
plt.ylim([-300, 500])

#By
plt.ylabel('Bx in nT')
ax = plt.subplot(4, 1, 3)
formatter = mdates.DateFormatter("%H:%M")
ax.xaxis.set_major_formatter(formatter)
plt.plot(dates_mes, by_mes_nT, label = 'MESSENGER Data')
plt.plot(dates_mes, by_KTH_nT, label = 'KTH Model')
plt.grid()
plt.ylim([-250, 250])

#Bz
plt.ylabel('By in nT')
ax = plt.subplot(4, 1, 4)
formatter = mdates.DateFormatter("%H:%M")
ax.xaxis.set_major_formatter(formatter)
plt.plot(dates_mes, bz_mes_nT,label = 'MESSENGER Data')
plt.plot(dates_mes, bz_KTH_nT,label = 'KTH Model')
plt.ylabel('Bz in nT')
plt.xlabel('time in HH:MM on ' + str_date_YMD)
plt.grid()
plt.ylim([-500, 100])


plt.show()

'''


print('done')
