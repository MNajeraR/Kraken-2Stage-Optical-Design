# -*- coding: utf-8 -*-
"""
======================================
  Script: ABC Optimization Setup
======================================

@author: MORGANRHAINAJERAROA

Description:
------------
This script sets up and optimizes an optical system using the Artificial Bee Colony (ABC) algorithm. 
Starting from a design derived from first- and third-order analysis, it explores the parameter 
space around the previously optimized configuration. The ABC algorithm refines surface curvatures 
and spacing based on a physically grounded merit function constructed from Fermat’s principle, 
coma cancellation, Coddington equations, and chromatic path equalization. Final system performance 
is evaluated using aberration metrics and spot diagram visualization.

Dependencies:
-------------
- NumPy: For numerical computations.
- KrakenOS (Kos): Optical simulation environment for system modeling and ray tracing.
- MOS_Class: Custom module with aberration evaluation, merit function, and optimization tools.
- MOS_equation: Custom module for ray tracing routines and diagram generation.
- bees_algorithm: Module containing the BeesAlgorithm implementation used for global optimization.
"""
# ===============================
#      Library Imports
# ===============================

import time
import numpy as np
import pkg_resources
required = {'KrakenOS'}
installed = {pkg.key for pkg in pkg_resources.working_set}
missing = required - installed

if missing:
    print("No instalado")
    import sys
    sys.path.append("../..")


import KrakenOS as Kos

import MOS_Class as MOSCL
import MOS_equation as Meq
from bees_algorithm import BeesAlgorithm

# ===============================
#    Constants Definition
# ===============================

per_red = 27.879562
Prx_data = MOSCL.Paraxial_Cal(per_red)

d_1 = Prx_data.d_1
d_2 = 4674.568305701443
d_3 = 27.792111342523572

# First order    
d_lens1 = 13.584 
d_lens2 = 9.878

F_Diameter = 104.491 
S_Diameter = 75.988 

# Third Order
R_1 = 239.06104309576506
R_2 = -101987.90085962816
R_3 = 9998.050905262751
R_4 = 495.20317019855105

d_4 = 186.6409959040964

# Calculate the field angle for the CCD
ccd_co = 16.1
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
L1a.Glass = "S-FPL51"
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
L2a.Glass = "F2HT"
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
P_Ima.Diameter = 100.0
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
Aberration = MOSCL.Aberration_Info(InfSystem, W)

# Set design wavelengths for chromatic analysis
Aberration.dw_1 = 0.35
Aberration.dw_2 = 0.5499996

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
print(f"Chromatic Aberration: {Aberration.Chromatic(1, [0.0, 0.0])[1]:.2f}")
print(f"Spherical Aberration: {Aberration.Spheric(1, [0.0, 0.0])[1]:.2f}")
print(f"Coma Aberration: {Aberration.Coma(1, Field_ccd)[1]:.2f}")
print(f"Astigmatism Aberration: {Aberration.Astigmatism(1, Field_ccd)[1]:.2f}")
print(f"Effective Focal Length: {EFFL_3O:.2f}")
print("======================================")
print('')


# ======================================
#  Artificial Bee Colony Optimization Process
# ======================================

print("\nStarting Artificial Bee Colony Optimization...")
start_time = time.time()


# Initialize the optimizer with the current optical system and rays
ABC_Result = MOSCL.Optimizer([Telescope_f85_FR, Rays])
ABC_Result.Field = Field_ccd
ABC_Result.ApVal = M1.Diameter

# Desired EFFL
EFFL_Des = ABC_Result.EFFL_Tr

# Define the merit function goal (numerically minimized output)
desired_output = 10

# Fitness function: evaluates how close a solution is to the desired merit value.
def fitness_func(solution):
    output = ABC_Result.Set_RcValues(solution)
    fitness = 1.0 / np.abs(output - desired_output)
    return fitness

# Define the search range (bounds) for each parameter based on third-order values
Delta_search_Rc = 0.25
Delta_search_d = 2

search_boundaries = (
    [R_1 - Delta_search_Rc, R_2 - Delta_search_Rc, R_3 - Delta_search_Rc, R_4 - Delta_search_Rc, d_4 - Delta_search_d], 
    [R_1 + Delta_search_Rc, R_2 + Delta_search_Rc, R_3 + Delta_search_Rc, R_4 + Delta_search_Rc, d_4 + Delta_search_d]
)

# Initialize and execute the ABC algorithm
alg = BeesAlgorithm(fitness_func, search_boundaries[0], search_boundaries[1])
alg.performFullOptimisation(max_iteration=100)

# Extract the best solution parameters
best = alg.best_solution
R1 = best.values[0]
R2 = best.values[1]
R3 = best.values[2]
R4 = best.values[3]
d4 = best.values[4]

# Compute and report the total time taken by the optimization
elapsed_time = time.time() - start_time
print("Elapsed time:", elapsed_time, "seconds")

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
print(f"Working distance (d4): {d4:.2f} mm")
print("======================================")

print("======================================")
print("Comparison with Initial Parameters:")
print(f"ΔR1: {np.abs(R1 - R_1):.2f} mm")
print(f"ΔR2: {np.abs(R2 - R_2):.2f} mm")
print(f"ΔR3: {np.abs(R3 - R_3):.2f} mm")
print(f"ΔR4: {np.abs(R4 - R_4):.2f} mm")
print(f"ΔWorking distance (d4): {np.abs(d4 - d_4):.2f} mm")
print("======================================")

# Update the optical system
Telescope_f85_FR.SDT[3].Rc = R1
Telescope_f85_FR.SDT[4].Rc = R2
Telescope_f85_FR.SDT[5].Rc = R3
Telescope_f85_FR.SDT[6].Rc = R4

Telescope_f85_FR.SDT[6].Thickness = d4

Telescope_f85_FR.SetData()
Telescope_f85_FR.SetSolid()


#Calculates the Airy disk radius and its coordinates for plotting.
    
#Parameters:
# W: # Reference wavelength
# Telescope_f85_FR.EFFL: Effective Focal Length
# M1.Diameter: Diameter of the primary mirror

Rairy = (1.22 * (Aberration.dw_2 / 1000) * Telescope_f85_FR.EFFL) / (M1.Diameter)
num_segmentos = 100
angulo = np.linspace(0, 2 * np.pi, num_segmentos + 1)
xairy = Rairy * np.cos(angulo)
yairy = Rairy * np.sin(angulo)


# ======================================
#    Plot Generation for All Fields
# ======================================

fields = [
    (0.0, 0.0, 'Field_0'),
    (np.rad2deg(Field_ccd), 0.0, 'Field_1'),
    (0.0, -np.rad2deg(Field_ccd), 'Field_2'),
    (0.035, -0.035, 'Field_3')
]

wavelengths = [Aberration.dw_1, W, Aberration.dw_2]

List_Radius = [] 

for fx, fy, name in fields:
    
    # Configure the pattern sampling for the pupil and fields:
    Pup.Ptype = "hexapolar"     
    Pup.FieldX = fx
    Pup.FieldY = fy
    
    # Perform ray tracing for all wavelengths
    traced_data = Meq.trace_rays(Pup, wavelengths, Rays)
    
    # Extract the data
    (Xa, Ya, Za, La, Ma, Na), (Xb, Yb, Zb, Lb, Mb, Nb), (Xc, Yc, Zc, Lc, Mc, Nc) = traced_data
    
    # Plot the diagram
    geo_r, rms_r = Meq.plot_spot_diagram(Xa, Ya, Xb, Yb, Xc, Yc, xairy, yairy, name, "ABC_Method")
    
    # Append to the list of radii
    List_Radius.append((geo_r, rms_r))

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
    x, y, z, L, M, N = Meq.configure_and_trace(Pup, fx, fy, pattern, W, clean)
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
print("\nFinal Aberration Analysis:")
print(f"Chromatic Aberration: {Aberration.Chromatic(1, [0.0, 0.0])[1]:.2f}")
print(f"Spherical Aberration: {Aberration.Spheric(1, [0.0, 0.0])[1]:.2f}")
print(f"Coma Aberration: {Aberration.Coma(1, Field_ccd)[1]:.2f}")
print(f"Astigmatism Aberration: {Aberration.Astigmatism(1, Field_ccd)[1]:.2f}")
print(f"Effective Focal Length: {Telescope_f85_FR.EFFL:.2f} mm")
print(f"EFFL deviation to Third Order's EFFL: {np.abs(Telescope_f85_FR.EFFL - EFFL_3O):.2f} mm")
print(f"EFFL deviation to Target: {np.abs(Telescope_f85_FR.EFFL - EFFL_Des):.2f} mm")
print(f"Airy disk radius: {Rairy * 1000:.2f}")
print(f"Average GEO radius: {GEO_R_average:.2f}")
print(f"Average RMS radius: {RMS_R_average:.2f}")
print(f"Airy's disk comparison with GEO and RMS: "
      f"GEO/Airy = {(GEO_R_average / (Rairy * 1000)):.2f}, "
      f"RMS/Airy = {(RMS_R_average / (Rairy * 1000)):.2f}")
print("======================================")
print('')


