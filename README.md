# KTH22-model
Model of Mercury's Magnetospheric Magnetic Field (KTH22) 

This model calculates the magnetic field inside the Hermean magnetosphere. 

Publication (under review): "Revised Magnetospheric Model Reveals Signatures of1
Field-Aligned Current Systems at Mercury" 

Description:
Calculates the magnetospheric field for Mercury. Based on Korth et al., (2015) with  improvements.
Model is intended for planning purposes of the BepiColombo mission. 
If you plan to make a publication with the aid of this model, the opportunity to participate as co-author
would be appreciated. 
If you have suggestions for improvements, do not hesitate to write me an email.
     
Takes into account:
- internal dipole field (offset dipole)
- field from neutral sheet current
- respective shielding fields from magnetopause currents
- aberration effect due to orbital motion of Mercury
- scaling with heliocentric distance
- scaling with Disturbance Indec (DI)


Required python packages: numpy, scipy

If you want to change the aberration, change the aberration value in the control parameter file; aberration angle in radiants, not in degrees. 

# Input Parameters:
x_mso: in, required, X-positions (array) in MSO base given in km
y_mso: in, required, Y-positions (array) in MSO base given in km
z_mso: in, required, z-positions (array) in MSO base given in km
r_hel: in, required, heliocentric distance in AU, use values between 0.3 and 0.47  
DI: in, required, disturbance index (0 < DI < 100), if not known:  50 (mean value) 
modules: dipole (internal and external), neutralsheet (internal and external)
"external = True" calculates the cf-fields (shielding fields) for each module which is set true
 
# Return: 
 Bx, By, Bz in nT for each coordinate given (x_mso, y_mso, z_mso)
(if outside of MP or inside Mercury, the KTH-model will return 'nan')
    
      
Authors:
Daniel Heyner, Institute for Geophysics and extraterrestrial Physics, Braunschweig, Germany, d.heyner@tu-bs.de
Kristin Pump, Institute for Geophysics and extraterrestrial Physics, Braunschweig, Germany, k.pump@tu-bs.de
