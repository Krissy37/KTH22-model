# -*- coding: utf-8 -*-
"""
Created on Wed May 17 15:37:25 2023

@author: Kristin
"""
# -*- coding: utf-8 -*-

import sys
import numpy as np
sys.path.append('C:\\Users\\Kristin\\Documents\\PhD\\KTH_V8_ext_user\\KTH22-model-main\\KTH22-model-main\\') # adapt this local path!
import kth22_model_for_mercury_v8 as kth
control_param_path = 'control_params_v8b.json'
fit_param_path = 'kth_own_cf_fit_parameters_opt_total_March23.dat'

# =============================================================================
# This is an example for how to run the KTH-Model to calculate the magnetic 
# field inside the hermean magnetosphere. 
# If you run this routine and get the output written below, everything works as it shoud. 
#
# Note: The model only calculates the magnetic field for points inside the magnetosphere. 
# Otherwise the output will be 'nan'. 
#
# If you want to change the magnetopause distance, this must be translated in a change in r_hel. 
# Only change the parameters in the function call. Don't change the parameter files. 
#
# There will be updates of parameters and modules. Status as of 17.05.2023. 
#
# If you have any questions, do not hesitate to cantact me (Kristin Pump, email: k.pump@tu-bs.de)
# =============================================================================


# radius of Mercury in km 
R_M = 2440

#define coordinates in mso
x_mso = np.linspace(-1.0, -2.0, 5)*R_M
y_mso = np.linspace(-0.800, -1.0, 5)*R_M
z_mso = np.linspace(-0.2, 0.2, 5)*R_M

#define heliocentric distance and disturbance index per data point
r_hel = np.ones(len(x_mso))* 0.38
di = np.ones(len(x_mso))*50


#run model 
B_KTH =  kth.kth22_model_for_mercury_v8(x_mso, y_mso, z_mso, r_hel, di, 
                                        control_param_path, fit_param_path, 
                                        dipole=True, neutralsheet=True, 
                                        ringcurrent=True, internal=True, 
                                        external=True)

Bx_KTH = B_KTH[0]
By_KTH = B_KTH[1]
Bz_KTH = B_KTH[2]

#print input coordinates
print('x (in MSO in km): ', x_mso)
print('y (in MSO in km): ', y_mso)
print('z (in MSO in km): ', z_mso)
print('\n')

#print magnetic field output
print('Bx KTH in nT:', Bx_KTH)
print('By KTH in nT:', By_KTH)
print('Bz KTH in nT:', Bz_KTH)

# =============================================================================
# if the model works correctly it should print: 
    
#    x (in MSO in km):  [-2440. -3050. -3660. -4270. -4880.]
#    y (in MSO in km):  [-1952. -2074. -2196. -2318. -2440.]
#    z (in MSO in km):  [-488. -244.    0.  244.  488.]


# Bx KTH in nT: [-65.00453696 -35.1217966  -18.94133967  -8.65453666   0.55134291]
# By KTH in nT: [-44.9178099  -18.18511262  -6.56756208  -1.79356723   0.14939532]
# Bz KTH in nT: [72.98469275 51.90732468 32.95535278 21.22285886 16.7856876 ]
# =============================================================================

