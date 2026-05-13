# -*- coding: utf-8 -*-
"""
======================================
  Script: Third Order Optical System
======================================

@author: MORGANRHAINAJERAROA

Description:
------------
This script sets up and optimizes a third-order optical system using KrakenOS.
It initializes various optical components, performs paraxial analysis,
calculates Seidel aberrations, and optimizes the curvature and thickness
of optical surfaces. Finally, it visualizes the spot diagrams for different
fields and evaluates the final optical performance.

Dependencies:
-------------
- NumPy: For numerical operations.
- SciPy: For optimization tasks.
- KrakenOS (Kos): Main optical simulation environment.
- MOS_Class: Custom module for paraxial calculations.
- MOS_equation: Custom module for handling ray tracing and aberrations.

======================================
"""

# ===============================
#      Library Imports
# ===============================

from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import time
import numpy as np
import pkg_resources
import scipy
import matplotlib.pyplot as plt

# Import KrakenOS and custom modules
from utils import (Paraxial_Cal, ThirdOrder_Cal, 
                   Set_Initial_Radius, apply_system_actualization, configure_and_trace,  
                   Prin_Plane, run_spots_for_fields, seidel_terms, airy_data, 
                   configure_pupil_and_ab, plot_all_EE_for_fields) 
                    

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

per_red = 50.05329978714254                       # Reduction percentage
EFFL_JSTelescope = 18273.877041856547             # Effective Focal Length

b_3 = -0.009626281725433176                       # Optical power of Lens L1
b_4 =  0.01065114262935968                        # Optical power of Lens L2

d_lens1 = 30.                                     # Thickness of Lens L1
d_lens2 = 8.5                                     # Thickness of Lens L2

F_Diameter = 105.                                 # Firts Lens Diameter
S_Diameter = 90.0                                 # Second Lens Diameter

# Initialize paraxial calculations for the system
Prx_data = Paraxial_Cal(per_red)
Prx_data.EFFL_Tel = EFFL_JSTelescope

d_2 = Prx_data.d_1 + Prx_data.d_2 
d_3 = Prx_data.d_3

# Desired EFFL
EFFL_Des = Prx_data.EFFL_Tr

# Calculate the field angle for the CCD
ccd_co = 11.25
Field_ccd = ccd_co / Prx_data.EFFL_Tr

# Initialize initial height 
h_i = 1076.00

# ===============================
#    Starting Points Definition
# ===============================

# Initial radii setup with R2 = -R1 approximation (Lens-Maker)
Phi_L1 = b_3  
n_L1   = 1.5
d_L1   = d_lens1  
R1_initial, R2_initial= Set_Initial_Radius(Phi_L1, n_L1, d_L1)


Phi_L2 = b_4   
n_L2   = 1.6
d_L2   = d_lens2  
R3_initial, R4_initial = Set_Initial_Radius(Phi_L2, n_L2, d_L2)


# Principal plane calculation
H_1, H_2 = Prin_Plane(n_L1, R1_initial, R2_initial, d_L1)
H_3, H_4 = Prin_Plane(n_L2, R3_initial, R4_initial, d_L2)


# ===============================
#    Initialization of list 
# ===============================

Glasses_History = []
Optical_Parameters_Set = []
System_History_Initial = []
Chromatic_History_Initial = []
System_History_Final = []
Chromatic_History_Final = []
EE_History_Initial = []
EE_History_Final = []


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
L1a.Rc = R1_initial
L1a.Thickness = d_lens1
L1a.Glass = "K-PFK85"
# L1a.Glass = "S-FPL51"
L1a.Diameter = F_Diameter

# ______________________________________#

# Lens L1b configuration

L1b = Kos.surf()
L1b.Rc = R2_initial
L1b.Thickness = Prx_data.d_3
L1b.Glass = "AIR"
L1b.Diameter = F_Diameter

# ______________________________________#


# Lens L2a configuration

L2a = Kos.surf()
L2a.Rc = R3_initial
L2a.Thickness = d_lens2
L2a.Glass = "ADF355"
# L2a.Glass = "F2HT"
L2a.Diameter = S_Diameter + 0.1*S_Diameter

# ______________________________________#

# Lens L2b configuration

L2b = Kos.surf()
L2b.Rc = R4_initial
L2b.Thickness = Prx_data.d_4 
L2b.Diameter = S_Diameter + 0.1*S_Diameter

# ______________________________________#

# ======================================
#  Image Plane Initialization
# ======================================

P_Ima = Kos.surf()
P_Ima.Rc = 0.0
P_Ima.Thickness = 0.0
P_Ima.Glass = "AIR"
P_Ima.Diameter = 35.0
P_Ima.Name = "Plano imagen"

# ______________________________________#

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


# Consider the principal planes when accounting for lens thickness.
Telescope_f85_FR.SDT[2].Thickness = d_2 - H_1
Telescope_f85_FR.SDT[4].Thickness = d_3 - H_2 - H_3
Telescope_f85_FR.SDT[6].Thickness = Prx_data.d_4 - H_4

# Update of the optical system
Telescope_f85_FR.SetData()

# Guarda el primer conjunto de vidios
Glasses_History.append([L1a.Glass, L2a.Glass])

# ======================================
#  Paraxial Calculation Setup
# ======================================

# Define the reference wavelength:
W = 0.43032015       # in micrometers
# Define the wavelength range:
RW = [0.35, W, 0.5499996]    
    
# Perform a paraxial analysis of the system at the specified wavelength (W).
Prx = Telescope_f85_FR.Parax(W)

# Extract the list of refractive indices for each surface in the system.
N = Prx[11]

# Assign the refractive indices of the two main lenses for reference.
n_l1i = N[3]          # Refractive index of the first lens.
n_l2i = N[5]          # Refractive index of the second lens.

# Extract the paraxial matrix (ABCD Matrix) of the optical system.
M_System = np.array(Prx[0]) 

# From the paraxial matrix, extract the D element.
MS_d = M_System[1][1] 

# ======================================
#  Pupil Calculation Setup and Seidel Aberration Calculation
# ======================================

sup = 1              # Number of the surface representing the opening of the system
AperType = "EPD"     # AperType sets the aperture ("STOP") or entrance pupil diameter ("EPD").
AperVal = M1.Diameter # Diameter of the entrance pupil

# Pupil setup and Seidel computation
Pup, AB = configure_pupil_and_ab(Telescope_f85_FR, Kos, sup, RW, AperType,
                                 AperVal, Field_ccd, samp=7)

# Extract individual aberration dimensionless terms
Sph, Coma, Ast, CLon = seidel_terms(AB, W)

# Guarda información inicial de las aberraciones e cromatica individual
System_History_Initial.append([Sph**2 + Coma**2 + Ast**2 + CLon**2, 
                              MS_d,Telescope_f85_FR.EFFL])
Chromatic_History_Initial.append(CLon) 

# ======================================
#  Display Initial Values
# ====================================== 

print("======================================")
print("Initial Seidel Aberrations:")
print(f"Spherical Aberration: {Sph:.2f}")
print(f"Coma: {Coma:.2f}")
print(f"Astigmatism: {Ast:.2f}")
print(f"Longitudinal Chromatic Aberration: {CLon:.2f}")
print(f"Matrix Element (D) from Paraxial Analysis: {MS_d:.2f}")
print(f"Effective Focal Length: {Telescope_f85_FR.EFFL:.2f} mm")
print("======================================")
print('')

# Return to initial configuration
Telescope_f85_FR.RestoreData()

# ======================================
#  Process of Optimization
# ======================================
print("\nStarting Optimization...")
# Start timing the optimization process
start_time = time.time()

# Initialize third-order aberration analysis with the current lens parameters
TO_data = ThirdOrder_Cal(Telescope_f85_FR, [b_3, b_4])
TO_data.W = W

# Define the initial guess for the solver
initial_guess = [L1a.Rc, L1b.Rc, L2a.Rc, L2b.Rc, Prx_data.d_4]

# Define the boundaries for the solver
LimInf = [-10000000, -100000000, -10000000, -100000000, -1000]
LimSup = [ 100000000, 100000000,  100000000, 100000000, 1000]
b = (LimInf, LimSup)

# Perform the least-squares optimization to find optimal curvatures and thickness
Curvatur_Rad = scipy.optimize.least_squares(TO_data.SeedPar, initial_guess, bounds=b, verbose=1)


# Extract the results
[R_1, R_2, R_3, R_4, d4] = Curvatur_Rad.x
elapsed_time = time.time() - start_time
Set_Initial_Opt = [R_1, R_2, R_3, R_4, d4]

# ======================================
#  Save Parameters
# ======================================


optimizedparameters_ROOT = Path(__file__).resolve().parents[1]
output_dir = optimizedparameters_ROOT / "optimized_parameters"
output_dir.mkdir(parents=True, exist_ok=True)

file_path = (
    output_dir
    / f"Third_Order_Parameters_{L1a.Glass}_{L2a.Glass}.txt"
)

with open(file_path, "w", encoding="utf-8") as f:

    f.write(
        f"Third_Order_Parameters_{L1a.Glass}_{L2a.Glass}= [\n"
    )

    for value in Set_Initial_Opt:
        f.write(f"{value:.8f},\n")

    f.write("]\n")

print(f"[ok] Archivo '{file_path}' guardado con los valores actuales.")


# ======================================
#  System Update with Optimized Values
# ======================================

Telescope_f85_FR, deltaZ = apply_system_actualization(
    system=Telescope_f85_FR,
    R1=Set_Initial_Opt[0], R2=Set_Initial_Opt[1],
    R3=Set_Initial_Opt[2], R4=Set_Initial_Opt[3],
    n_l1=n_l1i, d_lens1=d_lens1,
    n_l2=n_l2i, d_lens2=d_lens2,
    TO_data=TO_data, d4=Set_Initial_Opt[4],
    sup=sup, AperType=AperType, AperVal=AperVal, Field_ccd=Field_ccd,
    W_ref = 0.43032015,
    Rays=Rays, Kos=Kos, AB=AB,
    n_nodes=3, n_arms=6, samp=7
)


Set_Initial_Opt[4] =  d4 + deltaZ

# Guarda el primer set de parametros ópticos

Optical_Parameters_Set.append(Set_Initial_Opt)

# ======================================
#   Display Optimization Results
# ======================================

print("System parameters updated with optimized values.")
print("======================================")
print("Optimization Complete")
print(f"Elapsed Time: {elapsed_time:.2f} seconds")
print("Optimized Parameters:")
print(f"R1: {Set_Initial_Opt[0]:.2f} mm")
print(f"R2: {Set_Initial_Opt[1]:.2f} mm")
print(f"R3: {Set_Initial_Opt[2]:.2f} mm")
print(f"R4: {Set_Initial_Opt[3]:.2f} mm")
print(f"Lens Thickness (d4): {Set_Initial_Opt[4]:.2f} mm")
print("======================================")

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

# Extract individual aberration dimensionless terms
Sph, Coma, Ast, CLon = seidel_terms(AB, W)


#Calculates the Airy disk radius and its coordinates for plotting.
Rairy, xairy, yairy = airy_data(Telescope_f85_FR, W, M1)

# ======================================
#    Plot Generation for All Fields
# ======================================

wavelengths = [AB.Wf, W, AB.Wc]

name_save = f"Third_Order_{L1a.Glass}_{L2a.Glass}"

List_Radius, meta = run_spots_for_fields(
    Pup, Rays, Field_ccd, wavelengths,
    xairy, yairy, name_save=name_save, ptype="hexapolar",
    save=True, show = True, show_geo_circle=False,
    show_rms_circle=False, lock_box_across_fields=True,    
    box_include_airy=False, save_dir = 'figures\SPT_Diagrams\Ch1_2L')


EE_Example_information = plot_all_EE_for_fields(
                            Pup, Rays, Field_ccd, RW,
                            show_r50 = True, save = True, save_dir='figures\EE_Diagrams\Ch1_2L',
                            show=True,
                            filename=f"EE_{L1a.Glass}_{L2a.Glass}_fields.pdf",
                            # airy_radius_um=Rairy*1000.,
                            # multiply_by_diff_limit=True, 
                            )   

# Convert to NumPy array for easy manipulation
List_Radius = np.array(List_Radius)

EE_History_Final.append(EE_Example_information)

# Geo_radius in mm 
GEO_R = List_Radius[:, 0] * 1000
GEO_R_average = np.average(GEO_R)

# RMS_radius in mm 
RMS_R = List_Radius[:, 1] * 1000
RMS_R_average = np.average(RMS_R)


#Guarda información final de las metricas del sistema y cromatica individual
System_History_Final.append([Sph**2 + Coma**2 + Ast**2 + CLon**2, Telescope_f85_FR.EFFL,
                            np.abs(Telescope_f85_FR.EFFL - EFFL_Des), GEO_R_average,
                            RMS_R_average, (GEO_R_average / (Rairy * 1000)),
                            (RMS_R_average / (Rairy * 1000)) ])
Chromatic_History_Final.append(CLon) 


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
#  Display Final Results
# ======================================

print("======================================")
print("Final Seidel Aberrations after Optimization:")
print(f"Spherical Aberration: {Sph:.2f}")
print(f"Coma: {Coma:.2f}")
print(f"Astigmatism: {Ast:.2f}")
print(f"Longitudinal Chromatic Aberration: {CLon:.2f}")
print(f"Matrix Element (D) from Paraxial Analysis: {MS_d:.2f}")
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

# ======================================
#  Display Merit Function Evolution
# ======================================

iters = TO_data.history['iter']
merit = TO_data.history['Merit_fun']

plt.figure(figsize=(8,6))
plt.plot(iters, merit, linestyle='-', color='black')
plt.xlabel("Iteraciones")
plt.ylabel("Merit Function")

plt.grid(True, alpha=0.3)
plt.legend()
plt.show()





