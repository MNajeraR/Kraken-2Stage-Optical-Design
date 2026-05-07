# -*- coding: utf-8 -*-
"""
Created on Mon Oct 13 11:32:08 2025

@author: MORGANRHAINAJERAROA
"""

import numpy as np
import math
import scipy


#########################################################################################################

def Set_Initial_Radius(Phi, n, d):
    """
    Computes the initial radii R1 and R2 for a thick symmetric lens (R2 = -R1),
    given the lens power (Phi), refractive index (n), and center thickness (d).

    Parameters
    ----------
    Phi : float
        Lens power (1/length units).
        Note: Make sure the sign convention for Phi is consistent:
              positive for converging lenses, negative for diverging lenses.
    n : float
        Refractive index of the lens material.
    d : float
        Center thickness of the lens (same units as used for R, e.g., mm).

    Returns
    -------
    tuple (R1, R2)
        Selected radii of curvature from the quadratic solution.
        The radius with the larger absolute value (smaller curvature) is chosen.
    """

   
    a = ((n-1)**2*d)/n
    b = -2*(n-1)
    c = -Phi


    discriminante = b**2 - 4*a*c
    x1 = (-b + np.sqrt(discriminante)) / (2*a)
    x2 = (-b - np.sqrt(discriminante)) / (2*a)

    # Convert to radii
    R1_plus, R1_minus = 1.0 / x1, 1.0 / x2

    # Select the radius with the largest absolute value
    R1_selected = R1_plus if abs(R1_plus) > abs(R1_minus) else R1_minus
    R2_selected = -R1_selected

    return R1_selected, R2_selected

#########################################################################################################

"""
======================================
  Function: R_RMS_delta
======================================

This function calculates the Root Mean Square (RMS) radius for the 
intersection coordinates of rays with a surface or plane.

Parameters:
- Z1: Distance along the optical axis to the surface or plane.
- L, M, N: Direction cosines of the rays.
- X0, Y0: Initial coordinates of the rays.

Steps:
1. Prevent division by zero by replacing zeros in N with NaN.
2. Compute the intersection coordinates (X1, Y1) on the plane at Z1.
3. Calculate the mean position (cenX, cenY) of the intersection points.
4. Compute the shift of each point from the mean position.
5. Calculate the squared radius for each ray from the mean position.
6. Compute the RMS value of these distances.

Returns:
- R_RMS: The Root Mean Square (RMS) radius, representing the average 
  spread of the rays around their mean intersection point.

======================================
"""

def R_RMS_delta(Z1, L, M, N, X0, Y0):
    
    # Avoid a crash for division by zero
    N = np.where(N == 0, np.nan, N)
    
    # Calculation for the intersection coordinates of rays with a surface or plane 
    X1 = ((L / N) * Z1) + X0
    Y1 = ((M / N) * Z1) + Y0
    
    # Calculate the mean of the intersection coordinates
    cenX = np.mean(X1)
    cenY = np.mean(Y1)
    
    # Determine the shift from the mean position
    x1 = (X1 - cenX)
    y1 = (Y1 - cenY)
    
    # Calculate the squared radius for each ray
    R2 = ((x1 * x1) + (y1 * y1))
    
    # Compute the RMS radius
    R_RMS = np.sqrt(np.mean(R2))
    
    return R_RMS

#########################################################################################################


"""
======================================
  Function: BestFocus
======================================

This function optimizes the position of the last optical surface 
in the system to minimize the RMS radius of the spot diagram, 
representing the best focus position.

Parameters:
- X, Y, Z: Coordinates of the rays.
- L, M, N: Direction cosines of the rays.
- system: The optical system configuration.
- mod: (optional) If set to 1, updates the system with the best focus position.

Steps:
1. Initialize the starting point for the focus shift (delta_Z).
2. Define a tuple with the ray data (direction cosines and coordinates).
3. Use a solver to find the Z displacement that minimizes the RMS radius.
4. If `mod` is set to 1, update the system's thickness for the last surface.
5. Return the modified system and the optimized displacement value.

Returns:
- system: The updated optical system if `mod` is 1.
- v[0]: The optimal displacement value for best focus.

======================================
"""

def BestFocus(X, Y, Z, L, M, N, system, mod=1):

    # Initialize the starting point for the Z-axis shift (delta_Z)
    delta_Z = 0
    
    # Prepare the data for optimization: direction cosines and initial positions
    ZZ = (L, M, N, X, Y)
    
    # Solve for the optimal Z-axis shift that minimizes the RMS radius
    v = scipy.optimize.fsolve(R_RMS_delta, delta_Z, args=ZZ)
    
    # If mod is set to 1, update the optical system with the new focus position
    if mod == 1:
        
        # Adjust the thickness of the last surface in the optical system
        system.SDT[-2].Thickness = system.SDT[-2].Thickness + v[0]
        
        # Apply the changes to the system configuration
        system.SetData()
        system.SetSolid()
        
    # Return the modified system and the optimal displacement
    return system, v[0]


#########################################################################################################

"""
======================================
  Function: Min_Dis_point
======================================

This function calculates the coordinates of the point where the distance 
is minimal between two lines defined by the initial coordinates and 
direction cosines. It solves the geometric problem of finding the closest
approach between two skew rays.

Parameters:
- x1, y1, z1 (float): Initial coordinates of the first ray.
- x2, y2, z2 (float): Initial coordinates of the second ray.
- L1, M1, N1 (float): Direction cosines of the first ray.
- L2, M2, N2 (float): Direction cosines of the second ray.

Steps:
1. Compute the cross product of the direction vectors to obtain the normal vector.
2. Calculate the parameter 't' that determines the intersection in the minimum distance.
3. Compute the coordinates of the minimal distance point using parameter 't'.

Returns:
- x, y, z (float): Coordinates of the point where the distance is minimal.

======================================
"""

def Min_Dis_point(x1, y1, z1, x2, y2, z2, L1, M1, N1, L2, M2, N2):

    # Compute the vector perpendicular to the two direction vectors
    Vx = M1 * N2 - N1 * M2
    Vy = N1 * L2 - L1 * N2
    Vz = L1 * M2 - M1 * L2

    # Calculate the parameter 't' for the line of intersection
    t = (x2 - x1) / Vx

    # Coordinates of the point where the distance is minimal
    x = x1 + t * Vx
    y = y1 + t * Vy
    z = z1 + t * Vz

    return x, y, z

#########################################################################################################

"""
======================================
  Function: calculate_average
======================================

This function calculates the average of a list of numbers.

Parameters:
- lst (list): A list of numeric values.

Returns:
- average (float): The average of the numbers in the list. 
  Returns 0 for an empty list.

======================================
"""

def calculate_average(lst):

    if not lst:
        return 0  # Return 0 for an empty list
    
    total = sum(lst)
    average = total / len(lst)
    
    return average



