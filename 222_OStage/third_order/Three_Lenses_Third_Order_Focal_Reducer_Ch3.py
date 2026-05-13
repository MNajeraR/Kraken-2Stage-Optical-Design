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


# Import KrakenOS and custom modules
from utils import (Paraxial_Cal, ThirdLens_3O_20, 
                     configure_and_trace,  run_spots_for_fields, seidel_terms, airy_data, 
                    configure_pupil_and_ab,  plot_all_EE_for_fields,
                    Gaussian_Quadrature, BestFocus, seidel_terms_20) 
                    

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
# ===============================

Glass = 'K-PFK85 & ADF355 & K-PFK85'
#Glass = 'S-FPL51 & F2HT & S-FPL51'

glass_tag = Glass.replace(" ", "").replace("&", "_")
Glass_list = [g.strip() for g in Glass.split('&')]

def read_first_order_parameters(file_path):
    params = []

    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()

            # Ignorar comentarios y líneas vacías
            if not line or line.startswith("#"):
                continue

            # Ignorar corchetes y encabezado
            if "[" in line or "]" in line or "=" in line:
                continue

            # Convertir a float
            params.append(float(line.replace(",", "")))

    return params

optimizedparameters_ROOT = Path(__file__).resolve().parents[1]

file_path = (
    optimizedparameters_ROOT
    / "optimized_parameters"
    / f"First_Order_Parameters_{glass_tag}_Ch3.txt"
)

First_Order_Params = read_first_order_parameters(file_path)

# ===============================
#    Constants Definition
# ===============================

per_red = 50.05329978714254                    # Reduction percentage
EFFL_JSTelescope = 18273.877041856547          # Effective Focal Length


d_lens1 = 22.5                                 # Thickness of Lens L1
d_lens2 = 9.                                   # Thickness of Lens L2
d_lens3 = 16.                                  # Thickness of Lens L3

F_Diameter = 113.4                             # Firts Lens Diameter
S_Diameter = 99.                               # Second Lens Diameter
T_Diameter = 85.

# Initialize paraxial calculations for the system
Prx_data = Paraxial_Cal(per_red)

d_2 = 4657.886720609288
d_3 = 2.706860254305352E+001
d_4 = 13.86001147
d_5 = 107.56555422

# Desired EFFL
EFFL_Des = Prx_data.EFFL_Tr

# Calculate the field angle for the CCD
ccd_co = 11.25
Field_ccd = ccd_co / Prx_data.EFFL_Tr

# Initialize initial height 
h_i = 1076.00


# Desired EFFL
EFFL_Des = Prx_data.EFFL_Tr

# Calculate the field angle for the CCD
ccd_co = 11.25
Field_ccd = ccd_co / Prx_data.EFFL_Tr

# Initialize initial height 
h_i = 1076.0

# ===============================
#    Starting Points Definition
# ===============================

# Obtained through the script Three_Lenses_Paraxial.py

R1_initial, R2_initial, R3_initial, R4_initial, R5_initial, R6_initial = First_Order_Params


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
L1a.Glass = Glass_list[0]
L1a.Diameter = F_Diameter

# ______________________________________#

# Lens L1b configuration

L1b = Kos.surf()
L1b.Rc = R2_initial
L1b.Thickness = d_3 
L1b.Glass = "AIR"
L1b.Diameter = F_Diameter

# ______________________________________#


# Lens L2a configuration

L2a = Kos.surf()
L2a.Rc = R3_initial
L2a.Thickness = d_lens2
L2a.Glass = Glass_list[1]
L2a.Diameter = S_Diameter + 0.1*S_Diameter

# ______________________________________#

# Lens L2b configuration

L2b = Kos.surf()
L2b.Rc = R4_initial
L2b.Thickness = d_4 
L2b.Glass = "AIR"
L2b.Diameter = S_Diameter + 0.1*S_Diameter

# ______________________________________#


# Lens L3a configuration

L3a = Kos.surf()
L3a.Rc = R5_initial
L3a.Thickness = d_lens3
L3a.Glass = Glass_list[2]
L3a.Diameter = T_Diameter

# ______________________________________#

# Lens L3b configuration

L3b = Kos.surf()
L3b.Rc = R6_initial
L3b.Thickness = d_5 
L3b.Glass = "AIR"
L3b.Diameter = T_Diameter

# ______________________________________#

# ======================================
#  Image Plane Initialization
# ======================================

P_Ima = Kos.surf()
P_Ima.Rc = 0.0
P_Ima.Thickness = 0.0
P_Ima.Glass = "AIR"
P_Ima.Diameter = 100.0
P_Ima.Name = "Plano imagen"

# ______________________________________#

# ======================================
#  System Configuration Setup
# ======================================

A = [P_Obj, M1, M2, L1a, L1b, L2a, L2b, L3a, L3b, P_Ima]

config_1 = Kos.Setup()

# ======================================
#  System Initialization
# ======================================

# Initialize the optical system with the defined elements and configuration
Telescope_f85_FR = Kos.system(A, config_1)

# Create a RayKeeper instance for storing and accessing ray tracing results.
Rays = Kos.raykeeper(Telescope_f85_FR)

# ======================================
#  Paraxial Calculation Setup
# ======================================

# Define the reference wavelength:
W = 0.8067       # in micrometers
# Define the wavelength range:
RW = [0.67, W, 1.0]    
    
# Perform a paraxial analysis of the system at the specified wavelength (W).
Prx = Telescope_f85_FR.Parax(W)

# Extract the list of refractive indices for each surface in the system.
N = Prx[11]

# Assign the refractive indices of the two main lenses for reference.
n_l1i = N[3]          # Refractive index of the first lens.
n_l2i = N[5]          # Refractive index of the second lens.
n_l3i = N[7]          # Refractive index of the third lens.


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
Sph, Coma, Ast, FCur, Dis, CLon = seidel_terms_20(AB, W)

# ======================================
#  Display Initial Values
# ====================================== 

print("======================================")
print("Initial Seidel Aberrations:")
print(f"Spherical Aberration: {Sph:.2f}")
print(f"Coma: {Coma:.2f}")
print(f"Astigmatism: {Ast:.2f}")
print(f"Field Curvature: {FCur:.2f}")
print(f"Distortion: {Dis:.2f}")
print(f"Longitudinal Chromatic Aberration: {CLon:.2f}")
print(f"Matrix Element (D) from Paraxial Analysis: {MS_d:.2f}")
print(f"Effective Focal Length: {Telescope_f85_FR.EFFL:.2f} mm")
print("======================================")
print('')

# ======================================
#  Process of Optimization
# ======================================

print("\nStarting Optimization...")
# Start timing the optimization process
start_time = time.time()

# Initialize third-order aberration analysis with the current lens parameters
TO_data = ThirdLens_3O_20(Telescope_f85_FR)
TO_data.W = W
TO_data.W_1 = RW[0]
TO_data.W_2 = RW[2]

# Define the initial guess for the solver
initial_guess = [L1a.Rc, L1b.Rc, L2a.Rc, L2b.Rc, L3a.Rc, L3b.Rc, d_5]

# Define the boundaries for the solver
LimInf = [-10000000, -100000000, -10000000, -100000000, -10000000, -100000000, -1000]
LimSup = [ 100000000, 100000000,  100000000, 100000000,  100000000, 100000000,  1000]
b = (LimInf, LimSup)


# Perform the least-squares optimization to find optimal curvatures and thickness
Curvatur_Rad = scipy.optimize.least_squares(TO_data.SeedPar, initial_guess, bounds=b, verbose=1, 
                                            method = 'trf', ftol = 1e-2)

# Extract the results
[R_1, R_2, R_3, R_4, R_5, R_6, d5] = Curvatur_Rad.x
elapsed_time = time.time() - start_time
Set_Initial_Opt = [R_1, R_2, R_3, R_4, R_5, R_6, d5]

# ======================================
#  System Update with Optimized Values
# ======================================

# --- 1) Set radios
Telescope_f85_FR.SDT[3].Rc = R_1
Telescope_f85_FR.SDT[4].Rc = R_2
Telescope_f85_FR.SDT[5].Rc = R_3
Telescope_f85_FR.SDT[6].Rc = R_4
Telescope_f85_FR.SDT[7].Rc = R_5
Telescope_f85_FR.SDT[8].Rc = R_6

Telescope_f85_FR.SDT[8].Thickness = d5

# --- 4) Apply changes to the optical system
Telescope_f85_FR.SetData()
Telescope_f85_FR.SetSolid()

Pup = Kos.PupilCalc(Telescope_f85_FR, sup, W, AperType, AperVal)
Pup.Samp = 7
Pup.FieldType = "angle"
Pup.FieldX = np.rad2deg(Field_ccd)

# --- 6) Gaussian ray sampling at three wavelengths
InfSystem = [Telescope_f85_FR, Rays, Pup]

def sample_gaussian_rays(wavelengths, n_nodes=3, n_arms=6, fx=np.rad2deg(Field_ccd), fy=-np.rad2deg(Field_ccd), resp=0):
    samples = [Gaussian_Quadrature(InfSystem, wl).Coordinates_GQ(n_nodes, n_arms, fx, fy, resp)
                for wl in wavelengths]
    return [np.concatenate(items) for items in zip(*samples)]

wavelengths = [AB.Wf, W, AB.Wc]
(all_x, all_y, all_z,
  all_l, all_m, all_n) = sample_gaussian_rays(wavelengths, n_nodes=3, n_arms=6)

# --- 7) Best focus
Telescope_f85_FR, deltaZ = BestFocus(all_x, all_y, all_z,
                                    all_l, all_m, all_n, Telescope_f85_FR)

Set_Initial_Opt[6] =  d5 + deltaZ

# ======================================
#  Save Parameters
# ======================================

output_dir = optimizedparameters_ROOT / "optimized_parameters"
output_dir.mkdir(parents=True, exist_ok=True)

file_path = (
    output_dir
    / f"Third_Order_Parameters_{L1a.Glass}_{L2a.Glass}_{L3a.Glass}_Ch3.txt"
)

with open(file_path, "w", encoding="utf-8") as f:

    f.write(
        f"Third_Order_Parameters_{L1a.Glass}_{L2a.Glass}_{L3a.Glass}_Ch3= [\n"
    )

    for value in Set_Initial_Opt:
        f.write(f"{value:.8f},\n")

    f.write("]\n")

print(f"[ok] Archivo '{file_path}' guardado con los valores actuales.")

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
print(f"R5: {Set_Initial_Opt[4]:.2f} mm")
print(f"R6: {Set_Initial_Opt[5]:.2f} mm")
print(f"Lens Thickness (d5): {Set_Initial_Opt[6]:.2f} mm")
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

name_save = f"Third_Order_Three_Lenses_{L1a.Glass}_{L2a.Glass}_{L3a.Glass}_Ch3"

List_Radius, meta = run_spots_for_fields(
    Pup, Rays, Field_ccd, wavelengths,
    xairy, yairy, name_save=name_save,
    save_dir='figures\SPT_Diagrams\Ch3', ptype="hexapolar",
    save=True, show = True, show_geo_circle=False,
    show_rms_circle=False, lock_box_across_fields=True,    
    box_include_airy=False)


EE_Example_information = plot_all_EE_for_fields(
                            Pup, Rays, Field_ccd, RW,
                            show_r50 = True, save_dir='figures\EE_Diagrams\Ch3',
                            save = True,  show=True,
                            filename=f"EE_Third_Order_Three_Lenses_{L1a.Glass}_{L2a.Glass}_{L3a.Glass}_Ch3.pdf",
                            airy_radius_um=Rairy*1000.,
                            multiply_by_diff_limit=True, 
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
W = 0.8067 
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