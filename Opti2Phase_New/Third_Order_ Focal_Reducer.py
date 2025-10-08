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
import time
import numpy as np
import pkg_resources
import scipy
import matplotlib.pyplot as plt

# Import KrakenOS and custom modules
from utils import MOS_Class as MOSCL
from utils import MOS_equation as Meq

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

d_lens1 = 27.                                     # Thickness of Lens L1
d_lens2 = 9.2                                     # Thickness of Lens L2

F_Diameter = 105.                                 # Firts Lens Diameter
S_Diameter = 90.0                                 # Second Lens Diameter

# Initialize paraxial calculations for the system
Prx_data = MOSCL.Paraxial_Cal(per_red)
Prx_data.EFFL_Tel = EFFL_JSTelescope

# Desired EFFL
EFFL_Des = Prx_data.EFFL_Tr

# Calculate the field angle for the CCD
ccd_co = 11.25
Field_ccd = ccd_co / Prx_data.EFFL_Tr

# Initialize initial height 
h_i = 1076.00


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
M2.Thickness = Prx_data.d_1 + Prx_data.d_2
M2.k = -4.32070000000000E+000
M2.Glass = "MIRROR"
M2.Diameter = 3.175E+002 * 2.0


# ______________________________________#

# ======================================
#    Lens Initialization
# ======================================

# Lens L1a configuration

L1a = Kos.surf()
L1a.Rc = 100.
L1a.Thickness = d_lens1
L1a.Glass = "S-FPL51"
L1a.Diameter = F_Diameter

# ______________________________________#

# Lens L1b configuration

L1b = Kos.surf()
L1b.Rc = -100.
L1b.Thickness = Prx_data.d_3 + 12.0
L1b.Glass = "AIR"
L1b.Diameter = F_Diameter

# ______________________________________#

# Lens L2a configuration

L2a = Kos.surf()
L2a.Rc = -100.
L2a.Thickness = d_lens2
L2a.Glass = "F2HT"
L2a.Diameter = S_Diameter + 0.1*S_Diameter

# ______________________________________#

# Lens L2b configuration

L2b = Kos.surf()
L2b.Rc = 100.
L2b.Thickness = Prx_data.d_4
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


# ======================================
#  Pupil Calculation Setup
# ======================================

# Define the parameters for the pupil calculation:
W = 0.43032015       # Reference wavelength in micrometers
sup = 1              # Number of the surface representing the opening of the system
AperType = "EPD"     # AperType sets the aperture ("STOP") or entrance pupil diameter ("EPD").
AperVal = M1.Diameter # Diameter of the entrance pupil

# Initialize the Pupil calculation for the system
Pup = Kos.PupilCalc(Telescope_f85_FR , sup, W, AperType, AperVal)

# Configure the sampling for the pupil and its field representation:
Pup.Samp = 7         # Integer number for pupil ray sampling
Pup.FieldType = "angle" # Field type, this in terms of object height and distance from the plane.
Pup.FieldX = np.rad2deg(Field_ccd) # Field value in degrees on the X-axis


# ======================================
#  Paraxial Calculation Setup
# ======================================

# Perform a paraxial analysis of the system at the specified wavelength (W).
Prx = Telescope_f85_FR.Parax(W)

# Extract the list of refractive indices for each surface in the system.
N = Prx[11]

# Assign the refractive indices of the two main lenses for reference.
n_l1 = N[3]          # Refractive index of the first lens.
n_l2 = N[5]          # Refractive index of the second lens.

# Extract the paraxial matrix (ABCD Matrix) of the optical system.
M_System = np.array(Prx[0]) 

# From the paraxial matrix, extract the D element.
MS_d = M_System[1][1] 

# Save important lens specifications for future analysis.
Lens_data = [
             [d_lens1, d_lens2], # Thicknesses of the two lenses.
             [b_3, b_4],         # Power of the lenses.
             [n_l1, n_l2]        # Refractive indices.  
             ]

# ======================================
#  Seidel Aberration Calculation
# ======================================

# Perform Seidel aberration analysis on the system using the configured pupil.
AB = Kos.Seidel(Pup)

AB.Wf = 0.35                    # Set the first adjacent design wavelength (shorter) 
AB.Wd = W                       # Set the center design wavelength
AB.Wc = 0.5499996               # Set the second adjacent design wavelength (longer)

# Extract individual aberration terms

Sph =  AB.SAC_TOTAL[0]*(1000/W)   # Spherical Aberration
Coma = AB.SAC_TOTAL[1]*(1000/W)  # Coma
Ast =  AB.SAC_TOTAL[2]*(1000/W)   # Astigmatism
CLon = np.sum(AB.CL)*(1000/W)   # Longitudinal Chromatic Aberration


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

# # ______________________________________#
# # ______________________________________#

# ======================================
#  Process of Optimization
# ======================================
print("\nStarting Optimization...")
# Start timing the optimization process
start_time = time.time()

# Initialize third-order aberration analysis with the current lens parameters
TO_data = MOSCL.ThirdOrder_Cal(AB, Lens_data)
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



# ======================================
#  System Update with Optimized Values
# ======================================
Telescope_f85_FR.SDT[3].Rc = R_1
Telescope_f85_FR.SDT[4].Rc = R_2
Telescope_f85_FR.SDT[5].Rc = R_3
Telescope_f85_FR.SDT[6].Rc = R_4

# Recalculate principal planes with the new configuration
H1_a, H2_a, H1_b, H2_b = TO_data.Prin_Plane(R_1, R_2, R_3, R_4) 

# Update the thickness of the system based on the calculated principal planes
Telescope_f85_FR.SDT[2].Thickness = TO_data.d_1 + TO_data.d_2 - H1_a
Telescope_f85_FR.SDT[4].Thickness = TO_data.d_3 + H2_a - H1_b 
Telescope_f85_FR.SDT[6].Thickness = d4 + H2_b

# Apply the changes to the optical system
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
    samples = [MOSCL.Gaussian_Quadrature(InfSystem, wl).Coordinates_GQ(n_nodes, n_arms, fx, fy, resp) for wl in wavelengths]
    return [np.concatenate(items) for items in zip(*samples)]

# Perform ray sampling at three wavelengths
wavelengths = [AB.Wf, W, AB.Wc]
all_points_x, all_points_y, all_points_z, all_points_l, all_points_m, all_points_n = sample_gaussian_rays(wavelengths)


# Apply best focus correction
system, deltaZ = Meq.BestFocus(all_points_x, all_points_y, all_points_z,
                                all_points_l, all_points_m, all_points_n, Telescope_f85_FR)


# ======================================
#  Display Optimization Results
# ======================================

print("System parameters updated with optimized values.")
print("======================================")
print("Optimization Complete")
print(f"Elapsed Time: {elapsed_time:.2f} seconds")
print("Optimized Parameters:")
print(f"R1: {R_1:.2f} mm")
print(f"R2: {R_2:.2f} mm")
print(f"R3: {R_3:.2f} mm")
print(f"R4: {R_4:.2f} mm")
print(f"Lens Thickness (d4): {d4 + H2_b + deltaZ :.2f} mm")
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

# Perform Seidel aberration analysis on the system using the optimized configuration
AB = Kos.Seidel(Pup)

AB.Wf = 0.35                    # Set the first adjacent design wavelength (shorter) 
AB.Wd = W                       # Set the center design wavelength
AB.Wc = 0.5499996               # Set the second adjacent design wavelength (longer)


# Extract individual aberration terms

Sph =  AB.SAC_TOTAL[0]*(1000/W)  # Spherical Aberration
Coma = AB.SAC_TOTAL[1]*(1000/W)  # Coma
Ast =  AB.SAC_TOTAL[2]*(1000/W)   # Astigmatism
CLon = np.sum(AB.CL)*(1000/W)   # Longitudinal Chromatic Aberration


#Calculates the Airy disk radius and its coordinates for plotting.
    
#Parameters:
# W: # Reference wavelength
# Telescope_f85_FR.EFFL: Effective Focal Length
# M1.Diameter: Diameter of the primary mirror

AiryPrx = Telescope_f85_FR.Parax(W)

NA = (M1.Diameter)/(2*AiryPrx[7])
Rairy = 1.22*((W/1000)/(2*NA))
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
    (np.rad2deg(Field_ccd), -np.rad2deg(Field_ccd), 'Field_3')
]

wavelengths = [AB.Wf, W, AB.Wc]

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
    
    Spot_Set = [(Xa, Ya), (Xb, Yb), (Xc, Yc)]
    field_Set = [fx,fy]
    
    # Plot the diagram
    geo_r, rms_r = Meq.plot_spot_diagram(Spot_Set, field_Set, xairy, yairy, name, "Third_Order")
    
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

iters = TO_data.history['iter']
merit = TO_data.history['Merit_fun']

plt.figure(figsize=(6,5))
plt.plot(iters, merit, linestyle='-', color='b', label='Merit_fun')
plt.xlabel("Iteraciones")
plt.ylabel("Merit_fun")
plt.title("Evolución de la función de mérito")
plt.grid(True, alpha=0.3)
plt.legend()
plt.show()


plt.figure(figsize=(7,6))

for key in ['B_1','B_2','F_1','F_2','Merit_fun']:
    plt.plot(TO_data.history['iter'],TO_data.history[key], linestyle='-', label=key)

plt.xlabel("Iteraciones")
plt.ylabel("Valor")
plt.title("Progreso por iteración")
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()
