# -*- coding: utf-8 -*-
"""
Created on Tue May 16 11:25:13 2023

@author: MORGANRHAINAJERAROA
"""

import numpy as np
from .ops import (R_RMS_delta, BestFocus, Min_Dis_point)

#########################################################################################################

              
"""
======================================
  Function: MS_Telescope
======================================

This function computes the ABCD matrix (paraxial matrix) for a telescope 
configuration, considering a deistanc shift 'd' and a set of constants.

Parameters:
- d: Distance shift applied to the telescope's optical path.
- cons: A tuple containing the coefficients:
    - b_1: Power coefficient for the primary mirror.
    - b_2: Power coefficient for the secondary mirror.
    - d_1: Separation distance between the primary and secondary mirrors.

Steps:
1. Define the ABCD matrix for each segment of the telescope path:
   - M6 to M10 represent different sections of the optical path.
2. Multiply the matrices to obtain the global paraxial matrix (MSJS).

Returns:
- MSJS: The final paraxial matrix representing the telescope configuration.

======================================
"""
def MS_Telescope(d, cons):
    
    # Unpack the constants
    b_1, b_2, d_1 = cons
    
    # Define the ABCD matrices for each optical segment
    M6  = np.array([1, 0, d_1 + d, 1]).reshape(2, 2)
    M7  = np.array([1, b_2, 0, 1]).reshape(2, 2)
    M8  = np.array([1, 0, d_1, 1]).reshape(2, 2)
    M9  = np.array([1, b_1, 0, 1]).reshape(2, 2)
    M10 = np.array([1, 0, 1, 1]).reshape(2, 2)

    # Compute the global paraxial matrix
    MSJS = M6 @ M7 @ M8 @ M9 @ M10
    return MSJS

#########################################################################################################

"""
======================================
  Function: BestRMS
======================================

This function calculates the RMS radius for a set of 
coordinates and direction cosines for multiple nodes and arms following the 
Gaussian Quadrature in the optical system. It determines the optimal focus 
position and calculates the RMS value for each node in the original configuration.

Parameters:
- info_coordinates: A tuple containing:
    - set_x, set_y, set_z: Coordinates of the rays.
    - set_l, set_m, set_n: Direction cosines of the rays.
- system: The optical system configuration.

Steps:
1. Unpack the coordinate information from the tuple.
2. Determine the number of nodes and arms in the configuration.
3. Reshape the coordinates into a single list of elements for processing.
4. Optimize the focus position using `BestFocus`.
5. Iterate through each node and calculate the RMS radius.
6. Store the RMS value for each node in a list.

Returns:
- rms: A list of RMS values for each node.

======================================
"""

def BestRMS(info_coordinates, system):
    
    # Unpack coordinates information
    set_x = info_coordinates[0]
    set_y = info_coordinates[1]
    set_z = info_coordinates[2]
    set_l = info_coordinates[3]
    set_m = info_coordinates[4]
    set_n = info_coordinates[5]
    
    # # Obtain the number of nodes and arms
    # n_nodes = len(set_x)
    # n_arms = len(set_x[0])
    
    # # Reshape the coordinates of information into a single list with n_nodes * n_arms elements
    # x_reshaped = set_x.reshape(1, n_nodes * n_arms)
    # y_reshaped = set_y.reshape(1, n_nodes * n_arms)
    # z_reshaped = set_z.reshape(1, n_nodes * n_arms)
    # l_reshaped = set_l.reshape(1, n_nodes * n_arms)
    # m_reshaped = set_m.reshape(1, n_nodes * n_arms)
    # n_reshaped = set_n.reshape(1, n_nodes * n_arms)
    
    # # Initialize the counter for iteration
    # i_ray = 0
    
    # Optimize the position of the best focus without modifying the system (mod = 0)
    system, deltaZ = BestFocus(set_x, set_y, set_z, set_l, set_m, set_n, system, mod=0)
    
    # # Initialize a list to keep track of the RMS values for each node
    rms = R_RMS_delta(deltaZ, set_l, set_m, set_n, set_x, set_y)
    
    # # Iterate through each node and calculate the RMS radius
    # while i_ray < n_nodes:
    #     rms.append(R_RMS_delta(deltaZ, set_l[i_ray], set_m[i_ray], set_n[i_ray], set_x[i_ray], set_y[i_ray]))
    #     i_ray += 1
    
    return rms

#########################################################################################################


# ======================================
#  Section for sphere aberration
# ======================================

"""
======================================
  Function: Sphere_subtract
======================================

This function computes the absolute differences between each element 
in a list and the first element of that list. It is specifically used 
to calculate the deviation of marginal rays from the chief ray in 
total optical path analysis.

Parameters:
- lst (list): A list of numeric values representing total optical path lengths 
             for different rays.
             
Steps:
1. Initialize an empty list to store results.
2. Iterate over each element in the input list.
3. Compute the absolute difference between the chief ray (lst[0]) 
   and the current element.
4. Append the difference to the result list.
5. Return the populated result list.
             

Returns:
- result (list): A list containing the absolute differences between 
                 each element and the first element (chief ray).
    
======================================
"""

def Sphere_subtract(lst):

    result = []
    
    for i in range(len(lst)):
        subtraction = np.abs(lst[0] - lst[i])
        result.append(subtraction)
        
    return result


#########################################################################################################

"""
======================================
  Function: Coma_Substraction
======================================

This function calculates the radial deviation of marginal rays from the 
principal (chief) ray in the context of coma aberration analysis. It finds 
the minimal intersection points for pairs of marginal rays and compares their 
radial distances to the chief ray.

Parameters:
- Sample (int): The number of samples or marginal rays to analyze.
- Inf_Rays (tuple): A tuple containing:
    - Siscoor (list): A list of coordinates (x, y, z) for all the rays.
    - Dircos (list): A list of direction cosines (L, M, N) for all the rays.

Steps:
1. Extract the (x, y) coordinates of the chief ray (first element of Siscoor).
2. Initialize lists to store coordinates and direction cosines for the 
   marginal rays.
3. Loop through each sample, and for each pair of marginal rays:
   - Extract their coordinates and direction cosines.
   - Calculate the minimal intersection point using `Min_Dis_point`.
   - Compute the radial distances (r) of the intersection points for:
     a) The marginal rays.
     b) The chief ray.
   - Compute the absolute difference in radii and store it in `diff`.
   
4. The difference represents the displacement of the marginal rays 
   with respect to the principal ray in the image plane.

Returns:
- diff (list): A list containing the absolute differences in radial 
               distances between the minimal intersection points of 
               the marginal rays and the principal ray.

======================================
"""

def coordinates_inter(ya, yb, za, zb, ma, mb, na, nb):
    ym = (ma*mb)/(na*mb-nb*ma)*(zb-za-yb*(nb/mb)+ya*(na/ma))
    zm = (ym-yb)*(nb/mb)+zb
    return ym, zm



def Coma_Substraction(Sample, Inf_Rays):
    
    # Extract coordinates and direction cosines
    Siscoor = Inf_Rays[0]
    Dircos = Inf_Rays[1]
    # print(Siscoor)
    # Chief ray information
    yp0 = Siscoor[1][0]
    zp0 = Siscoor[2][0]
    

    Mp0 = Dircos[1][0]
    Np0 = Dircos[2][0]

    # Initialize lists for marginal ray information
    xma, xmb, yma, ymb, zma, zmb = [], [], [], [], [], []
    Lma, Lmb, Mma, Mmb, Nma, Nmb = [], [], [], [], [], []
    diff = []
    
    # Loop through each sample
    for i in range(Sample):

        # Extract marginal ray information
        xma.append(Siscoor[0][2*i+1])
        xmb.append(Siscoor[0][2*i+2])
        yma.append(Siscoor[1][2*i+1])
        ymb.append(Siscoor[1][2*i+2])
        zma.append(Siscoor[2][2*i+1])
        zmb.append(Siscoor[2][2*i+2])
        
        Lma.append(Dircos[0][2*i+1])
        Lmb.append(Dircos[0][2*i+2])
        Mma.append(Dircos[1][2*i+1])
        Mmb.append(Dircos[1][2*i+2])
        Nma.append(Dircos[2][2*i+1])
        Nmb.append(Dircos[2][2*i+2])
        
        # Find the minimal intersection point
        
        # xm, ym, zm = Min_Dis_point(xma[i], yma[i], zma[i], Lma[i], Mma[i],
        #                            Nma[i], xmb[i], ymb[i], zmb[i], Lmb[i], 
        #                            Mmb[i], Nmb[i])
        
        ym, zm = coordinates_inter(yma[i], ymb[i], zma[i], zmb[i], 
                                   Mma[i], Mmb[i], Nma[i], Nmb[i])
        
        yp = yp0 + (Mp0/Np0)*(zm-zp0)
        
        # Calculate the radial distances and their difference
        
        # rm = np.sqrt(xm**2 + ym**2)
        # rp = np.sqrt(xp0**2 + yp0**2)
        diff.append(np.abs(ym - yp))
        
    return diff

# ======================================
#  Section for astigmatism aberration
# ======================================

"""
======================================
  Function: Astigmatism_Substraction
======================================

This function performs the subtraction of astigmatism between sagittal and 
tangential rays for a given sample size. It calculates the minimum distance 
points for both sagittal and tangential planes and computes the absolute 
difference between these distances.

Parameters:
- Sample (int): The number of samples to process.
- Inf_Rays (tuple): A tuple containing the spatial coordinates and direction 
                    cosines of the rays:
    - Inf_Rays[0]: Spatial coordinates of the rays.
    - Inf_Rays[1]: Direction cosines of the rays.

Steps:
1. Initialize lists to store coordinates and direction cosines for:
    - Sagittal plane (xms, yms, zms, Lms, Mms, Nms)
    - Tangential plane (xmt, ymt, zmt, Lmt, Mmt, Nmt)
    
2. Loop through the sample size:
    - Extract information from `Inf_Rays` for each ray in sagittal and 
      tangential planes.
    - Store the information in their corresponding lists.
    
3. For each sample:
    - Calculate the minimum distance point for the sagittal rays using 
      `Min_Dis_point`.
    - Calculate the minimum distance point for the tangential rays.
    - Compute the Euclidean distance for both sagittal and tangential points.
    - Subtract these distances and store the absolute difference.

4. Return the list of differences representing the astigmatism aberration.

Returns:
- drs (list): A list of the absolute differences in astigmatism for each sample.

======================================
"""

def Astigmatism_Substraction(Sample, Inf_Rays):
    
    # Extract spatial coordinates and direction cosines from the tuple
    Siscoor = Inf_Rays[0]
    Dircos = Inf_Rays[1]
 
    # Initialize lists for Sagittal Plane
    xmsa, xmsb, ymsa, ymsb, zmsa, zmsb = [], [], [], [], [], []
    Lmsa, Lmsb, Mmsa, Mmsb, Nmsa, Nmsb = [], [], [], [], [], []
    
    # Initialize lists for Tangential Plane
    xmta, xmtb, ymta, ymtb, zmta, zmtb = [], [], [], [], [], []
    Lmta, Lmtb, Mmta, Mmtb, Nmta, Nmtb = [], [], [], [], [], []
    
    # List to store the differences between sagittal and tangential distances
    drs = []
    
    # Loop through each sample to extract information
    for i in range(Sample):
        
        # --------------------------
        # Sagittal Plane Extraction
        # --------------------------
        
        # Coordinates
        xmsa.append(Siscoor[0][4 * i])
        xmsb.append(Siscoor[0][4 * i + 1])
        ymsa.append(Siscoor[1][4 * i])
        ymsb.append(Siscoor[1][4 * i + 1])
        zmsa.append(Siscoor[2][4 * i])
        zmsb.append(Siscoor[2][4 * i + 1])
        
        # Direction Cosines
        Lmsa.append(Dircos[0][4 * i])
        Lmsb.append(Dircos[0][4 * i + 1])
        Mmsa.append(Dircos[1][4 * i])
        Mmsb.append(Dircos[1][4 * i + 1])
        Nmsa.append(Dircos[2][4 * i])
        Nmsb.append(Dircos[2][4 * i + 1])
        
        # --------------------------
        # Tangential Plane Extraction
        # --------------------------
        
        # Coordinates
        xmta.append(Siscoor[0][4 * i + 2])
        xmtb.append(Siscoor[0][4 * i + 3])
        ymta.append(Siscoor[1][4 * i + 2])
        ymtb.append(Siscoor[1][4 * i + 3])
        zmta.append(Siscoor[2][4 * i + 2])
        zmtb.append(Siscoor[2][4 * i + 3])
        
        # Direction Cosines
        Lmta.append(Dircos[0][4 * i + 2])
        Lmtb.append(Dircos[0][4 * i + 3])
        Mmta.append(Dircos[1][4 * i + 2])
        Mmtb.append(Dircos[1][4 * i + 3])
        Nmta.append(Dircos[2][4 * i + 2])
        Nmtb.append(Dircos[2][4 * i + 3])
        
        # --------------------------
        # Distance Calculation
        # --------------------------
        
        # # Calculate the minimum distance points for the sagittal rays
        # xs, ys, zs = Min_Dis_point(
        #     xmsa[i], ymsa[i], zmsa[i], 
        #     Lmsa[i], Mmsa[i], Nmsa[i], 
        #     xmsb[i], ymsb[i], zmsb[i], 
        #     Lmsb[i], Mmsb[i], Nmsb[i]
        # )
        
        # # Calculate the minimum distance points for the tangential rays
        # xt, yt, zt = Min_Dis_point(
        #     xmta[i], ymta[i], zmta[i], 
        #     Lmta[i], Mmta[i], Nmta[i], 
        #     xmtb[i], ymtb[i], zmtb[i], 
        #     Lmtb[i], Mmtb[i], Nmtb[i]
        # )
        
        # # Calculate the Euclidean distances
        # rs = np.sqrt(xs**2 + ys**2 + zs**2)   # Sagittal radius
        # rt = np.sqrt(xt**2 + yt**2 + zt**2)   # Tangential radius
        
        yms, zms = coordinates_inter(ymsa[i], ymsb[i], zmsa[i], zmsb[i], 
                                   Mmsa[i], Mmsb[i], Nmsa[i], Nmsb[i])
        
        ymt, zmt = coordinates_inter(ymta[i], ymtb[i], zmta[i], zmtb[i], 
                                   Lmta[i], Lmtb[i], Nmta[i], Nmtb[i])
        # Append the absolute difference to the results
        drs.append(np.abs(zms - zmt))
        
    # Return the list of differences
    return drs





























