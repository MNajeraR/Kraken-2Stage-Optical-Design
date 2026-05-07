# -*- coding: utf-8 -*-
"""
Created on Wed Nov 19 15:58:40 2025

@author: MORGANRHAINAJERAROA
"""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import time
import numpy as np
import pkg_resources
import scipy

# Import KrakenOS and custom modules
from utils import (Paraxial_Cal, Gaussian_Quadrature, BestFocus,
                   Aberration_Info, Three_Lens_Optimizer, configure_and_trace,  
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
# ===============================

Glass = 'K-PFK85 & ADF355 & K-PFK85'
# Glass = 'S-FPL51 & F2HT & S-FPL51'

glass_tag = Glass.replace(" ", "").replace("&", "_")
Glass_list = [g.strip() for g in Glass.split('&')]

def read_parameters(file_path):
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
    / f"Third_Order_Parameters_{glass_tag}_Ch1.txt"
)

Third_Order_Params = read_parameters(file_path)

# ===============================
#    Constants Definition
# ===============================

per_red = 50.05329978714254                    # Reduction percentage
EFFL_JSTelescope = 18273.877041856547          # Effective Focal Length


d_lens1 = 22.5                                 # Thickness of Lens L1
d_lens2 = 9.                                  # Thickness of Lens L2
d_lens3 = 16.                                 # Thickness of Lens L3

F_Diameter = 113.4                             # Firts Lens Diameter
S_Diameter = 99.                               # Second Lens Diameter
T_Diameter = 85.

# Initialize paraxial calculations for the system
Prx_data = Paraxial_Cal(per_red)

d_2 = 4657.886720609288
d_3 = 2.706860254305352E+001
d_4 = 13.86001147


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

# Obtained through the script Three_Lenses_Third_Order_Focal_Reducer_Ch1.py

if Glass == 'K-PFK85 & ADF355 & K-PFK85':
    #Percentaje of perturbation
    Per = 0.005
    
    R = np.array(Third_Order_Params[:6], dtype=float)
    R *= (1.0 + Per)
    
elif Glass == 'S-FPL51 & F2HT & S-FPL51':
    
    #Percentaje of perturbation
    Per = -0.0015
    
    R = np.array(Third_Order_Params[:6], dtype=float)
    R *= (1.0 + Per)
    

## Seed Curvature Radii
R1_initial, R2_initial, R3_initial, R4_initial, R5_initial, R6_initial = R

## Seed Wroking Distance 
d_5 = Third_Order_Params[6]


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
print(f"R_1: {R1_initial:.2f} mm")
print(f"R_2: {R2_initial:.2f} mm")
print(f"R_3: {R3_initial:.2f} mm")
print(f"R_4: {R4_initial:.2f} mm")
print(f"R_5: {R5_initial:.2f} mm")
print(f"R_6: {R6_initial:.2f} mm")
print(f"Working distance (d_5): {d_5:.2f} mm")
print(" ")

# ======================================
#  Display Initial Values
# ====================================== 

print("======================================")
print("Initial Aberrations and EFFL:")
print(f"Chromatic Aberration (0.0, 0.0) deg: {Aberration.Chromatic(1, [0.0, 0.0])[1]*(1000/W):.2f}")
print(f"Spherical Aberration (0.0, 0.0) deg: {Aberration.Spheric(1, [0.0, 0.0])[1]*(1000/W):.2f}")
print(f"Coma Aberration (0.0, -0.071) deg: {Aberration.Coma(1, [0,-Field_ccd])[1]*(1000/W):.2f}")
print(f"Astigmatism Aberration (0.0, -0.071) deg: {Aberration.Astigmatism(1, [0.,-Field_ccd])[1]*(1000/W):.2f}")
print(f"Chromatic Aberration (0.071, 0.071) deg: {Aberration.Chromatic(1, [0.0, 0.0])[1]*(1000/W):.2f}")
print(f"Spherical Aberration (0.071, -0.071) deg: {Aberration.Spheric(1, [0.0, 0.0])[1]*(1000/W):.2f}")
print(f"Coma Aberration (0.071, -0.071) deg: {Aberration.Coma(1, [0,-Field_ccd])[1]*(1000/W):.2f}")
print(f"Effective Focal Length: {EFFL_3O:.2f}")
print("======================================")
print('')

# ===============================
#    Classical Optimization Process
# ===============================
print("\nStarting Classical Optimization...")

start_time = time.time()

# Initialize the optimizer with system and rays
Classic_Result = Three_Lens_Optimizer([Telescope_f85_FR,Rays])
Classic_Result.Field = Field_ccd
Classic_Result.ApVal = M1.Diameter

# Desired EFFL
EFFL_Des = Classic_Result.EFFL_Tr

# Set the initial guess and bounds
set_R0 = [R1_initial, R2_initial, R3_initial, R4_initial, R5_initial, R6_initial, d_5]
Liminf = [-1e6] * 7
Limsup = [1e6] * 7
bounds = (Liminf, Limsup)

# Perform the optimization
Result = scipy.optimize.least_squares(Classic_Result.Set_RcValues, set_R0, bounds=bounds, verbose=0, 
                                      ftol = 1e-4)
R1, R2, R3, R4, R5, R6, d5 = Result.x
elapsed_time = time.time() - start_time


# Update the optical system
Telescope_f85_FR.SDT[3].Rc = R1
Telescope_f85_FR.SDT[4].Rc = R2
Telescope_f85_FR.SDT[5].Rc = R3
Telescope_f85_FR.SDT[6].Rc = R4
Telescope_f85_FR.SDT[7].Rc = R5
Telescope_f85_FR.SDT[8].Rc = R6

Telescope_f85_FR.SDT[8].Thickness = d5 

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
def sample_gaussian_rays(wavelengths, n_nodes=3, n_arms=6, fx=np.rad2deg(Field_ccd), fy=-np.rad2deg(Field_ccd), resp=0):
    samples = [Gaussian_Quadrature(InfSystem, wl).Coordinates_GQ(n_nodes, n_arms, fx, fy, resp) for wl in wavelengths]
    return [np.concatenate(items) for items in zip(*samples)]

# Perform ray sampling at three wavelengths
wavelengths = [Aberration.dw_1, W, Aberration.dw_2]
all_points_x, all_points_y, all_points_z, all_points_l, all_points_m, all_points_n = sample_gaussian_rays(wavelengths)


# Apply best focus correction
system, deltaZ = BestFocus(all_points_x, all_points_y, all_points_z,
                                all_points_l, all_points_m, all_points_n, Telescope_f85_FR)

# ======================================
#  Save Parameters
# ======================================

Set_PGMF_LS_OP_Opt = [R1, R2, R3, R4, R5, R6, d5+deltaZ]

output_dir = optimizedparameters_ROOT / "optimized_parameters"
output_dir.mkdir(parents=True, exist_ok=True)

file_path = (
    output_dir
    / f"PGMF_LS_Parameters_{L1a.Glass}_{L2a.Glass}_{L3a.Glass}_Ch1.txt"
)

with open(file_path, "w", encoding="utf-8") as f:

    f.write(
        f"PGMF_LS_Parameters_{L1a.Glass}_{L2a.Glass}_{L3a.Glass}_Ch1= [\n"
    )

    for value in Set_PGMF_LS_OP_Opt:
        f.write(f"{value:.8f},\n")

    f.write("]\n")

print(f"[ok] Archivo '{file_path}' guardado con los valores actuales.")


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
print(f"R5: {R5:.2f} mm")
print(f"R6: {R6:.2f} mm")
print(f"Working distance (d5): {d5+deltaZ:.2f} mm")
print("======================================")

print("======================================")
print("Comparison with Initial Parameters:")
print(f"ΔR1: {np.abs(R1 - R1_initial):.2f} mm")
print(f"ΔR2: {np.abs(R2 - R2_initial):.2f} mm")
print(f"ΔR3: {np.abs(R3 - R3_initial):.2f} mm")
print(f"ΔR4: {np.abs(R4 - R4_initial):.2f} mm")
print(f"ΔR5: {np.abs(R5 - R5_initial):.2f} mm")
print(f"ΔR6: {np.abs(R6 - R6_initial):.2f} mm")
print(f"ΔWorking distance (d5): {np.abs(d5 + deltaZ - d_5):.2f} mm")
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

name_save = f"PGMF_Three_Lens_LS_{L1a.Glass}_{L2a.Glass}_{L1a.Glass}_Ch1"

List_Radius, meta = run_spots_for_fields(
    Pup, Rays, Field_ccd, wavelengths,
    xairy, yairy, name_save=name_save, ptype="hexapolar",
    save=True, show = True, show_geo_circle=False,
    show_rms_circle=False, lock_box_across_fields=True,    
    box_include_airy=False, save_dir = 'figures\SPT_Diagrams\Ch1')


EE_Example_information = plot_all_EE_for_fields(
                            Pup, Rays, Field_ccd, RW, save_dir='figures\EE_Diagrams\Ch1',
                            show_r50 = True, save = True,  show=True,
                            filename=f"EE_PGMF_Three_Lens_LS_{L1a.Glass}_{L2a.Glass}_{L1a.Glass}_Ch1.pdf",
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
print(f"Chromatic Aberration (0.0, 0.0) deg: {Aberration.Chromatic(1, [0.0, 0.0])[1]*(1000/W):.2f}")
print(f"Spherical Aberration (0.0, 0.0) deg: {Aberration.Spheric(1, [0.0, 0.0])[1]*(1000/W):.2f}")
print(f"Coma Aberration (0.0, -0.071) deg: {Aberration.Coma(1, [0,-Field_ccd])[1]*(1000/W):.2f}")
print(f"Astigmatism Aberration (0.0, -0.071) deg: {Aberration.Astigmatism(1, [0.,-Field_ccd])[1]*(1000/W):.2f}")
print(f"Chromatic Aberration (0.071, 0.071) deg: {Aberration.Chromatic(1, [0.0, 0.0])[1]*(1000/W):.2f}")
print(f"Spherical Aberration (0.071, -0.071) deg: {Aberration.Spheric(1, [0.0, 0.0])[1]*(1000/W):.2f}")
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