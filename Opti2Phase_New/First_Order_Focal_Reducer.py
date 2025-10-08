# -*- coding: utf-8 -*-
"""
======================================
  Script: First-Order Lens Setup
======================================

@author: MORGANRHAINAJERAROA

Description:
------------
This script performs the first-order setup for an optical system using 
the KrakenOS optical simulator. It includes the calculation of paraxial 
parameters, matrix analysis for the focal reducer, and configuration of 
lenses and surfaces for the optical system.

Dependencies:
-------------
- NumPy: For numerical operations.
- SciPy: For optimization tasks.
- KrakenOS (Kos): Main optical simulation environment.
- MOS_Class: Custom module for paraxial calculations.

======================================
"""

# ===============================
#      Library Imports
# ===============================
import numpy as np
import scipy
from utils import MOS_Class as MOSCL
import KrakenOS as Kos

# ===============================
#      Function Definitions
# ===============================
def calculate_lens_powers(d3, d4, d_Tel, b_T, b_Tel):
    """
    Calculate the optical powers (b_3 and b_4) for two lenses based on the 
    provided distances and optical parameters.

    Parameters:
    - d3 (float): Thickness or separation of the first lens.
    - d4 (float): Thickness or separation of the second lens.
    - d_Tel (float): Distance for the Telescope.
    - b_T (float): Target total optical power.
    - b_Tel (float): Optical power in the Telescope.

    Returns:
    - b3 (float): Calculated optical power for the first lens.
    - b4 (float): Calculated optical power for the second lens.
    """
    # Calculate b3 using the provided formula
    b_L1 = (-d4 * b_Tel - d3 * b_Tel - d_Tel - d4 * (b_T - b_Tel)) / (d3 * d_Tel)
    
    # Calculate b4 using the provided formula
    b_L2 = (b_T - b_Tel - b_L1 * d_Tel) / (d3 * b_Tel + d_Tel + d3 * d_Tel * b_L1)
    
    return b_L1, b_L2

# ===============================
#    Constants Definition
# ===============================
per_red = 50.05329978714254        # Reduction percentage
h_i = 1076.00               # Height of incidence in mm
Prx_data = MOSCL.Paraxial_Cal(per_red)  # Initialize paraxial calculations


# ===============================
#    Field and Matrix Setup
# ===============================
Detector_co = 11.25
Field_Detector = Detector_co / Prx_data.EFFL_Tr  # Field angle calculation

# Generate transfer matrix for the telescope until the first lens
MS_t1 = Prx_data.MS_Telescope(Prx_data.d_2)


# ===============================
#    Solver Analytic function
# ===============================
b_f85 = MS_t1[0][1]
d_f85 = MS_t1[1][1]
b_Tgt = -1/Prx_data.EFFL_Tr
b_3, b_4 = calculate_lens_powers(Prx_data.d_3, Prx_data.d_4, d_f85, b_Tgt, b_f85)

# Display the analytic solution
print("Analytic Solution:", b_3, b_4)

# ===============================
#    Solver Configuration
# ===============================
# Initial guess for the optimization and bounds
initial_guess = [0, 0]
LimInf = [-1000, -1000]
LimSup = [1000, 1000]
bounds = (LimInf, LimSup)

# Least Squares Optimization to solve paraxial equations
B_sistem = scipy.optimize.least_squares(Prx_data.Prx_equation, initial_guess, bounds=bounds, verbose=0)
[b_3c, b_4c] = B_sistem.x

# Display the solution
print("Numerical Solution:", b_3c, b_4c)

# ===============================
#    Comparisson
# ===============================
print("Comparisson:", np.abs(b_3 - b_3c), np.abs(b_4 - b_4c))


# ===============================
#    Lens Diameter Analysis
# ===============================

# Generate transfer matrices for the telescope and first lens until the second lens, 
# and the total system matrix
MS_t2, MS_Tot = Prx_data.MS_FocalReducer(b_3, b_4)

# Definition of the input matrix for the marginal rays
Thhf_mar1 = np.array([[Field_Detector], [h_i]])
Thhf_mar2 = np.array([[-Field_Detector], [h_i]])

# ======================================
#    Matrix Multiplication for Marginal Rays
# ======================================
# Calculate the marginal rays through the first matrix
First_mard1 = MS_t1[1] @ Thhf_mar1
First_mard2 = MS_t1[1] @ Thhf_mar2
print("First Lens Marginal Rays Output:")
print(f"Ray 1: {First_mard1}")
print(f"Ray 2: {First_mard2}")

# Identify the maximum height for the first lens
First_d = max(First_mard1, First_mard2)

# Calculate the marginal rays through the second matrix 
Secon_mard1 = MS_t2[1] @ Thhf_mar1
Secon_mard2 = MS_t2[1] @ Thhf_mar2
print("Second Lens Marginal Rays Output:")
print(f"Ray 1: {Secon_mard1}")
print(f"Ray 2: {Secon_mard2}")

# Identify the maximum height for the second lens
Secon_d = max(Secon_mard1, Secon_mard2)

# ======================================
#    Lens Diameters Calculation
# ======================================
# Compute the diameters with an additional 25% margin to account for vignetting and alignment
F_Diameter = 2 * (First_d[0] + First_d[0] * 0.25)
S_Diameter = 2 * (Secon_d[0] + Secon_d[0] * 0.25)

# Compute the estimated thicknesses as 13% of the calculated diameters
d_lens1 = F_Diameter * 0.13
d_lens2 = S_Diameter * 0.13

# Display the computed lens diameters and thicknesses
print("======================================")
print("Calculated Lens Dimensions:")
print(f"First Lens Diameter: {F_Diameter:.3f} mm | Thickness: {d_lens1:.3f} mm")
print(f"Second Lens Diameter: {S_Diameter:.3f} mm | Thickness: {d_lens2:.3f} mm")
print("======================================")

# ===============================
#    Optical Components Definition
# ===============================
# Object plane
P_Obj = Kos.surf()
P_Obj.Rc = 0.0
P_Obj.Thickness = Prx_data.d_Obj
P_Obj.Glass = "AIR"
P_Obj.Diameter = h_i * 2.

# Thin lenses (representing mirrors as a lenses with specified power)
M1 = Kos.surf()
M1.Thin_Lens = 1.118E004 / 2
M1.Thickness = Prx_data.d_1
M1.Rc = 0.0
M1.Glass = "AIR"
M1.Diameter = h_i * 2.

M2 = Kos.surf()
M2.Thin_Lens = -4430 / 2
M2.Thickness = Prx_data.d_1 + Prx_data.d_2
M2.Rc = 0.0
M2.Glass = "AIR"
M2.Diameter = 317.5 * 2.

# Lenses with calculated optical power
L1 = Kos.surf()
L1.Thin_Lens = -1 / b_3
L1.Thickness = Prx_data.d_3
L1.Rc = 0.0
L1.Glass = "AIR"
L1.Diameter = F_Diameter

L2 = Kos.surf()
L2.Thin_Lens = -1 / b_4
L2.Thickness = Prx_data.d_4
L2.Rc = 0.0
L2.Glass = "AIR"
L2.Diameter = S_Diameter

# Image plane
P_Ima = Kos.surf()
P_Ima.Rc = 0.0
P_Ima.Thickness = 0.0
P_Ima.Glass = "AIR"
P_Ima.Diameter = 100.0
P_Ima.Name = "Plano imagen"

# ===============================
#    System Configuration
# ===============================
# Define the sequence of elements
A = [P_Obj, M1, M2, L1, L2, P_Ima]
config_1 = Kos.Setup()

# Create the optical system for the Telescope f/8.5 with the focal reducer
Telescope_f85_FR = Kos.system(A, config_1)
Rayos = Kos.raykeeper(Telescope_f85_FR)

# ===============================
#    Pupil Calculation and Ray Tracing
# ===============================
Surf, W, AperVal, AperType = 1, 0.45, M1.Diameter, "EPD"
P = Kos.PupilCalc(Telescope_f85_FR, Surf, W, AperType, AperVal)

# Trace rays for three fields
fields = [np.rad2deg(Field_Detector), 0.0, -np.rad2deg(Field_Detector)]
for f in fields:
    P.FieldX = f
    xb, yb, zb, Lb, Mb, Nb = P.Pattern2Field()
    Kos.TraceLoop(xb, yb, zb, Lb, Mb, Nb, W, Rayos, clean=0)

# ===============================
#    Display the Layout
# ===============================

Kos.display2d(Telescope_f85_FR, Rayos, 1, arrow=0)