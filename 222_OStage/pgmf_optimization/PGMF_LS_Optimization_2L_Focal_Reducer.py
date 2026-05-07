# -*- coding: utf-8 -*-
"""
======================================
  Script: Classical Optimization Setup
======================================

@author: MORGANRHAINAJERAROA

Description:
------------
This script sets up and optimizes an optical system using KrakenOS, starting 
from an SPD defined by first- and third-order analysis. It uses the classical 
least-squares method (scipy) to refine radii and spacing based on a physically 
grounded merit function derived from Fermat’s principle, coma cancellation, 
Coddington equations, and chromatic path equalization. Final performance is 
evaluated through metrics such as GEO radius, RMS radius, EFFL deviation, 
convergence time  and spot diagram visualization.

Dependencies:
-------------
- NumPy: For numerical operations.
- SciPy: For optimization routines.
- KrakenOS (Kos): Optical simulation environment.
- MOS_Class: Custom module containing the merit function, aberration 
             evaluation methods, and the `Optimizer` class.
- MOS_equation: Custom module for ray tracing routines and spot 
                diagram visualization.
"""

# ===============================
#      Library Imports
# ===============================
import time
import numpy as np
import pkg_resources
import scipy

# Import KrakenOS and custom modules
from utils import (Paraxial_Cal, Gaussian_Quadrature, BestFocus,
                   Aberration_Info, Optimizer, configure_and_trace,  
                   run_spots_for_fields, airy_data, 
                   configure_pupil_and_ab,  plot_all_EE_for_fields) 

# Check if KrakenOS is installed, otherwise append its path
required = {'KrakenOS'}
installed = {pkg.key for pkg in pkg_resources.working_set}
missing = required - installed

if missing:
    print("KrakenOS not installed")
    import sys
    sys.path.append("../..")

import KrakenOS as Kos



# ===============================
#    Constants Definition
# ===============================

per_red = 50.05329978714254 
Prx_data = Paraxial_Cal(per_red)


# First order    
d_lens1 = 30.                        # Thickness of Lens L1
d_lens2 = 8.5                        # Thickness of Lens L2

F_Diameter = 105.                    # Firts Lens Diameter
S_Diameter = 90.0                    # Second Lens Diameter


d_1 = Prx_data.d_1
d_2 = 4657.886720609288
d_3 = 29.5944124283825


# Third Order
R_1 = 88.26054458103616 
R_2 = -383.5298465374939 
R_3 = -139.29006826264188 
R_4 = 292.84629048211497 

d_4 = 88.91841344048457


Per = 0.20

# # Third Order
R_1 = 88.26054458103616   + R_1*Per
R_2 = -383.5298465374939  + R_2*Per
R_3 = -139.29006826264188 + R_3*Per
R_4 = 292.84629048211497  + R_4*Per

d_4 = 88.91841344048457 + d_4*Per

# Calculate the field angle for the CCD
ccd_co = 11.25
Field_ccd = ccd_co / Prx_data.EFFL_Tr

# Initialize initial height 
h_i  = 1076.00


# ______________________________________#

# ======================================
#    Optical Element Initialization
# ======================================


# Object surface configuration

P_Obj = Kos.surf()
P_Obj.Rc = 0
P_Obj.Thickness = 1000 + Prx_data.d_1
P_Obj.Glass = "AIR"
P_Obj.Diameter = h_i * 2.0

# ______________________________________#

# ======================================
#  Telescope Initialization
# ======================================

# Mirror M1 configuration

M1 = Kos.surf()
M1.Rc = -11.176E+003
M1.Thickness = - Prx_data.d_1
M1.k = -1.070110000000E+000
M1.Glass = "MIRROR"
M1.Diameter = h_i * 2.0

# ______________________________________#

# Mirror M2 configuration

M2 = Kos.surf()
M2.Rc = -4.4300E+003
M2.Thickness = d_2
M2.k = -4.32070000000000E+000
M2.Glass = "MIRROR"
M2.Diameter = 3.175E+002 * 2.0


# ______________________________________#

# ======================================
#    Lens Initialization
# ======================================

# Lens L1a configuration

L1a = Kos.surf()
L1a.Rc = R_1
L1a.Thickness = d_lens1
L1a.Glass = "K-PFK85"
L1a.Diameter = F_Diameter

# ______________________________________#

# Lens L1b configuration

L1b = Kos.surf()
L1b.Rc = R_2
L1b.Thickness = d_3
L1b.Glass = "AIR"
L1b.Diameter = F_Diameter

# ______________________________________#

# Lens L2a configuration

L2a = Kos.surf()
L2a.Rc = R_3
L2a.Thickness = d_lens2
L2a.Glass = "ADF355"
L2a.Diameter = S_Diameter + 0.1*S_Diameter

# ______________________________________#

# Lens L2b configuration

L2b = Kos.surf()
L2b.Rc = R_4
L2b.Thickness = d_4
L2b.Glass = "AIR"
L2b.Diameter = S_Diameter + 0.1*S_Diameter

# ______________________________________#

# ======================================
#  Image Plane Initialization
# ======================================

P_Ima = Kos.surf()
P_Ima.Rc = 0.0
P_Ima.Thickness = 0.0
P_Ima.Glass = "AIR"
P_Ima.Diameter = 50.0
P_Ima.Name = "Plano imagen"

#______________________________________#

# ======================================
#  System Configuration Setup
# ======================================

A = [P_Obj, M1, M2, L1a, L1b, L2a, L2b, P_Ima]

config_1 = Kos.Setup()


# ======================================
#  System Initialization
# ======================================

# Initialize the optical system with the defined elements and configuration
Telescope_f85_FR = Kos.system(A, config_1)
# Create a RayKeeper instance for storing and accessing ray tracing results.
Rays = Kos.raykeeper(Telescope_f85_FR)


# ======================================
#  Pupil Calculation Setup
# ======================================

# Define the parameters for the pupil calculation:
W = 0.43032015             # Reference wavelength in micrometers
# Define the wavelength range:
RW = [0.35, W, 0.5499996]    
sup = 1                    # Number of the surface representing the opening of the system
AperType = "EPD"           # AperType sets the aperture ("STOP") or entrance pupil diameter ("EPD").
AperVal = M1.Diameter      # Diameter of the entrance pupil

# Initialize the Pupil calculation for the system
Pup = Kos.PupilCalc(Telescope_f85_FR , sup, W, AperType, AperVal)

# Configure the sampling for the pupil and its field representation:
Pup.Samp = 7              # Integer number for pupil ray sampling
Pup.FieldType = "angle"   # Field type, this in terms of object height and distance from the plane.
Pup.FieldX = np.rad2deg(Field_ccd) # Field value in degrees on the X-axis


# Third Order EFFL
EFFL_3O = Telescope_f85_FR.EFFL 

# ======================================
#    Aberration Analysis Setup
# ======================================
# Initializes the optical system for aberration analysis
InfSystem = [Telescope_f85_FR, Rays, Pup]
Aberration = Aberration_Info(InfSystem, W)

# Set design wavelengths for chromatic analysis
Aberration.dw_1 = RW[0]
Aberration.dw_2 = RW[2]

# Display surface radii

print("\nSurface Radii Configuration and working distance:")
print(f"R1: {R_1:.2f} mm")
print(f"R2: {R_2:.2f} mm")
print(f"R3: {R_3:.2f} mm")
print(f"R4: {R_4:.2f} mm")
print(f"Working distance (d4): {d_4:.2f} mm")
print(" ")

# ======================================
#  Display Initial Values
# ====================================== 

print("======================================")
print("Initial Aberrations and EFFL:")
print(f"Chromatic Aberration: {Aberration.Chromatic(1, [0.0, 0.0])[1]*(1000/W):.2f}")
print(f"Spherical Aberration: {Aberration.Spheric(1, [0.0, 0.0])[1]*(1000/W):.2f}")
print(f"Coma Aberration: {Aberration.Coma(1, [0., -Field_ccd])[1]*(1000/W):.2f}")
print(f"Astigmatism Aberration: {Aberration.Astigmatism(1, [0., -Field_ccd])[1]*(1000/W):.2f}")
print(f"Effective Focal Length: {EFFL_3O:.2f}")
print("======================================")
print('')

# ===============================
#    Classical Optimization Process
# ===============================
print("\nStarting Classical Optimization...")

start_time = time.time()

# Initialize the optimizer with system and rays
Classic_Result = Optimizer([Telescope_f85_FR,Rays])
Classic_Result.Field = Field_ccd
Classic_Result.ApVal = M1.Diameter

# Desired EFFL
EFFL_Des = Classic_Result.EFFL_Tr

# Set the initial guess and bounds
set_R0 = [R_1, R_2, R_3, R_4, d_4]
Liminf = [-1e6] * 5
Limsup = [1e6] * 5
bounds = (Liminf, Limsup)

# Perform the optimization
Result = scipy.optimize.least_squares(Classic_Result.Set_RcValues, set_R0, bounds=bounds, verbose=0, 
                                      ftol = 1e-3)
R1, R2, R3, R4, d4 = Result.x
elapsed_time = time.time() - start_time


# Update the optical system
Telescope_f85_FR.SDT[3].Rc = R1
Telescope_f85_FR.SDT[4].Rc = R2
Telescope_f85_FR.SDT[5].Rc = R3
Telescope_f85_FR.SDT[6].Rc = R4

Telescope_f85_FR.SDT[6].Thickness = d4 

Telescope_f85_FR.SetData()
Telescope_f85_FR.SetSolid()

# ======================================
#  Pupil Calculation Setup (Recalculation)
# ======================================

# Reinitialize the Pupil calculation for the optimized system
W = 0.43032015 
Pup = Kos.PupilCalc(Telescope_f85_FR, sup, W, AperType, AperVal)
Pup.Samp = 7         
Pup.FieldType = "angle" 
Pup.FieldX = np.rad2deg(Field_ccd)

# Initializes the optical system for aberration analysis
InfSystem = [Telescope_f85_FR, Rays, Pup]

# Define function to avoid repetition of sampling + coordinate extraction
def sample_gaussian_rays(wavelengths, n_nodes=3, n_arms=6, fx=0.0, fy=-np.rad2deg(Field_ccd), resp=0):
    samples = [Gaussian_Quadrature(InfSystem, wl).Coordinates_GQ(n_nodes, n_arms, fx, fy, resp) for wl in wavelengths]
    return [np.concatenate(items) for items in zip(*samples)]

# Perform ray sampling at three wavelengths
wavelengths = [Aberration.dw_1, W, Aberration.dw_2]
all_points_x, all_points_y, all_points_z, all_points_l, all_points_m, all_points_n = sample_gaussian_rays(wavelengths)


# Apply best focus correction
system, deltaZ = BestFocus(all_points_x, all_points_y, all_points_z,
                                all_points_l, all_points_m, all_points_n, Telescope_f85_FR)


# ======================================
#  Display Optimization Results
# ======================================

print("======================================")
print("Optimization Complete")
print(f"Elapsed Time: {elapsed_time:.2f} seconds")
print("Optimized Parameters:")
print(f"R1: {R1:.2f} mm")
print(f"R2: {R2:.2f} mm")
print(f"R3: {R3:.2f} mm")
print(f"R4: {R4:.2f} mm")
print(f"Working distance (d4): {d4+deltaZ:.2f} mm")
print("======================================")

print("======================================")
print("Comparison with Initial Parameters:")
print(f"ΔR1: {np.abs(R1 - R_1):.2f} mm")
print(f"ΔR2: {np.abs(R2 - R_2):.2f} mm")
print(f"ΔR3: {np.abs(R3 - R_3):.2f} mm")
print(f"ΔR4: {np.abs(R4 - R_4):.2f} mm")
print(f"ΔWorking distance (d4): {np.abs(d4 - d_4):.2f} mm")
print("======================================")
#Calculates the Airy disk radius and its coordinates for plotting.
    
#Parameters:
# W: # Reference wavelength
# Telescope_f85_FR.EFFL: Effective Focal Length
# M1.Diameter: Diameter of the primary mirror

# ======================================
#  Paraxial Analysis Recalculation
# ======================================

# Perform a paraxial analysis of the system at the specified wavelength (W).
Prx = Telescope_f85_FR.Parax(W)

# Extract the paraxial matrix (ABCD Matrix) of the optical system.
M_System = np.array(Prx[0]) 

# From the paraxial matrix, extract the D element.
MS_d = M_System[1][1] 


# ======================================
#  Seidel Aberration Analysis Recalculation
# ======================================

# Pupil setup and Seidel computation recalculation
Pup, AB = configure_pupil_and_ab(Telescope_f85_FR, Kos, sup, RW, AperType,
                                 AperVal, Field_ccd, samp=7)



#Calculates the Airy disk radius and its coordinates for plotting.
Rairy, xairy, yairy = airy_data(Telescope_f85_FR, W, M1)

# ======================================
#    Plot Generation for All Fields
# ======================================

wavelengths = [AB.Wf, W, AB.Wc]

name_save = f"PGMF_NOP_Classic_{L1a.Glass}_{L2a.Glass}"

List_Radius, meta = run_spots_for_fields(
    Pup, Rays, Field_ccd, wavelengths,
    xairy, yairy, name_save=name_save, ptype="hexapolar",
    save=True, show = True, show_geo_circle=False,
    show_rms_circle=False, lock_box_across_fields=True,    
    box_include_airy=False, save_dir = 'Images\SPT_Diagrams_New')


EE_Example_information = plot_all_EE_for_fields(
                            Pup, Rays, Field_ccd, RW, save_dir='Images\EE_Diagrams_New',
                            show_r50 = True, save = True,  show=True,
                            filename=f"EE_NOP_Classic_{L1a.Glass}_{L2a.Glass}_fields.pdf"
                            )   

# Convert to NumPy array for easy manipulation
List_Radius = np.array(List_Radius)


# Geo_radius in mm 
GEO_R = List_Radius[:, 0] * 1000
GEO_R_average = np.average(GEO_R)

# RMS_radius in mm 
RMS_R = List_Radius[:, 1] * 1000
RMS_R_average = np.average(RMS_R)


# ======================================
#    Plot for All the System
# ======================================

# ------------------------------------------------
# Field Configuration for Hexapolar Sampling
# ------------------------------------------------

field_configs = [
    (np.rad2deg(Field_ccd), 0.0, "hexapolar", 1),
    (0.0, 0.0, "hexapolar", 0),
    (-np.rad2deg(Field_ccd), 0.0, "hexapolar", 0)
]

traced_fields = []

for fx, fy, pattern, clean in field_configs:
    x, y, z, L, M, N = configure_and_trace(Pup, fx, fy, pattern, W, clean)
    Kos.TraceLoop(x * 0.99, y * 0.99, z * 0.99, 
                  L * 0.99, M * 0.99, N * 0.99, 
                  W, Rays, clean)
    traced_fields.append((x, y, z, L, M, N))

# ------------------------------------------------
# Display the 2D Optical System Layout
# ------------------------------------------------
Kos.display2d(Telescope_f85_FR, Rays, 1, arrow=0)

# Reinitialize the Pupil calculation
W = 0.43032015 
Pup = Kos.PupilCalc(Telescope_f85_FR, sup, W, AperType, AperVal)

# ======================================
#    Aberration Analysis Setup
# ======================================
# Initializes the optical system for aberration analysis
InfSystem = [Telescope_f85_FR, Rays, Pup]
Aberration = Aberration_Info(InfSystem, W)

# Set design wavelengths for chromatic analysis
Aberration.dw_1 = RW[0]
Aberration.dw_2 = RW[2]

# ======================================
#  Display Final Results
# ======================================

print("======================================")
print("Final Aberrations after Optimization:")
print(f"Chromatic Aberration: {Aberration.Chromatic(1, [0.0, 0.0])[1]*(1000/W):.2f}")
print(f"Spherical Aberration: {Aberration.Spheric(1, [0.0, 0.0])[1]*(1000/W):.2f}")
print(f"Coma Aberration: {Aberration.Coma(1, [0., -Field_ccd])[1]*(1000/W):.2f}")
print(f"Astigmatism Aberration: {Aberration.Astigmatism(1, [0., -Field_ccd])[1]*(1000/W):.2f}")
print(f"Effective Focal Length: {Telescope_f85_FR.EFFL:.2f} mm")
print(f"EFFL deviation to Target: {np.abs(Telescope_f85_FR.EFFL - EFFL_Des):.2f} mm")
print(f"Airy disk radius: {Rairy * 1000:.2f}")
print(f"Average GEO radius: {GEO_R_average:.2f}")
print(f"Average RMS radius: {RMS_R_average:.2f}")
print(f"Airy's disk comparison with GEO and RMS: "
      f"GEO/Airy = {(GEO_R_average / (Rairy * 1000)):.2f}, "
      f"RMS/Airy = {(RMS_R_average / (Rairy * 1000)):.2f}")
print("======================================")
print('')


