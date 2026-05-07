# -*- coding: utf-8 -*-
"""
Created on Mon Oct 13 11:47:57 2025

@author: MORGANRHAINAJERAROA
"""


import pkg_resources
required = {'KrakenOS'}
installed = {pkg.key for pkg in pkg_resources.working_set}
missing = required - installed

if missing:
    print("No instalado")
    import sys
    sys.path.append("../..")


import KrakenOS as Kos


#########################################################################################################

def configure_and_trace(Pup, fx, fy, pattern_type, wavelength, clean=1):
    """
    Configure pupil parameters, perform ray tracing, and return traced rays.
    
    Parameters:
    - Pup (object): Pupil object that handles the sampling and tracing of rays.
    - fx (float): X-coordinate of the field in degrees, representing the angular 
                  displacement of the field with respect to the optical axis.
    - fy (float): Y-coordinate of the field in degrees, representing the angular 
                  displacement in the vertical plane.
    - pattern_type (str): The sampling pattern for rays (e.g., "hexapolar").
    - wavelength (float): Wavelength of light for the ray tracing process.
    - clean (int, optional): Flag indicating whether the system should be reset 
                             before tracing (default is 1, which means "yes").
    
    Process:
    1. Configure the pupil sampling settings:
        - Set the number of sampling points (1 point in this case).
        - Assign the field coordinates (`fx`, `fy`) to the pupil.
        - Set the type of sampling pattern (e.g., "hexapolar").
    
    2. Perform the ray tracing by invoking `Pattern2Field()`, which projects 
       the sampled rays from the pupil onto the field plane according to the 
       specified parameters.
    
    3. Return the traced coordinates and direction cosines of the rays.
    
    Returns:
    - tuple: Contains the traced coordinates (`x`, `y`, `z`) and direction 
             cosines (`L`, `M`, `N`) of the rays.
    """
    
    # Configure the pupil object with the specified parameters
    Pup.Samp = 1                # Single sampling point for tracing
    Pup.FieldX = fx             # Assign X-coordinate of the field
    Pup.FieldY = fy             # Assign Y-coordinate of the field
    Pup.Ptype = pattern_type    # Set the pattern type (e.g., "hexapolar")
    
    # Perform the ray tracing and return the traced data
    
    return Pup.Pattern2Field()

#########################################################################################################

def trace_rays(Pup, wavelengths, Rays):
    
    """
    Traces rays for a given pupil configuration and list of wavelengths.

    Parameters:
    - Pup: The pupil object.
    - wavelengths: List of wavelengths to trace.
    - Rays: The ray keeper object.

    Returns:
    - A list of tuples with traced coordinates for each wavelength.
    """
    
    # The array of rays: transfer from the unit pupil to the real pupil
    x, y, z, L, M, N = Pup.Pattern2Field()
    results = []

    # Loop through each wavelength and perform the trace
    for w in wavelengths:
        Kos.TraceLoop(x * 0.99, y * 0.99, z * 0.99, L, M, N, w, Rays, clean=1)
        # Collect the traced rays from the image plane
        results.append(Rays.pick(-1))

    return results

#########################################################################################################

"""
======================================
  Function: ProcessPattern2Field
======================================

This function processes the ray tracing for a specific wavelength (wv) 
using the provided system information (sys_info), which includes the 
optical system, the raykeeper instance, and the pupil calculation.

Parameters:
- wv: The wavelength for the ray tracing process.
- sys_info: A tuple containing:
    - system: The optical system instance.
    - raykeeper: The raykeeper instance for tracking rays.
    - P: The pupil calculation instance for field representation.

Steps:
1. Extract the ray coordinates and direction cosines from the pupil sampling.
2. Define the starting point for the rays, slightly adjusted (0.99 factor).
3. Perform ray tracing through the optical system at the specified wavelength.
4. Store the traced ray data in the raykeeper instance.
5. Extract the traced data from the third surface for further processing.

Returns:
- X, Y, Z, L, M, N: Coordinates and direction cosines of the rays after tracing.

======================================
"""

def ProcessPattern2Field(wv, sys_info):
    
    # Unpack the information of the entire system:
    system = sys_info[0]    # Optical system with the defined elements and configuration 
    raykeeper = sys_info[1] # RayKeeper instance for storing and accessing ray tracing results.
    P = sys_info[2]         # Pupil calculation for the system
    
    # The array of rays: transfer from the unit pupil to the real pupil
    xR, yR, zR, LR, MR, NR = P.Pattern2Field()
    # Set of origin coordinates of the rays
    pSource_0 = [xR[0]*0.99, yR[0]*0.99, zR[0]*0.99]
    # Set of origin director cosines of the rays
    dCos = [LR[0], MR[0], NR[0]]
    
    # Trace of the rays in a sequential way through every surface taking into consideration
    # the set of origin coordinates and director cosines and the wavelength
    system.Trace(pSource_0, dCos, wv)
    
    # Store the information of the traced ray
    raykeeper.push()
    
    # Access the information of the surface
    X, Y, Z, L, M, N = raykeeper.pick(3)
  
    return X, Y, Z, L, M, N