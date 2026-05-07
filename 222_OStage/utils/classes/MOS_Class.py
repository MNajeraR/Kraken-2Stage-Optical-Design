# -*- coding: utf-8 -*-
"""
Created on Tue May 16 11:28:57 2023

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
# Inside MOS_Class.py
import sys
import os

# Add the path of `utils` to the Python Path
module_path = os.path.abspath(os.path.join(os.path.dirname(__file__)))
if module_path not in sys.path:
    sys.path.append(module_path)

import numpy as np
from ..classes. Aberration_Information import Aberration_Info

#--------------------------#
# Physical Grounded Optimizer Class
#--------------------------#


"""
======================================
  Class: Optimizer
======================================

The `Optimizer` class is designed to adjust the curvature of 
specific optical surfaces to minimize aberrations and achieve optimal optical 
performance. It includes methods to:
- Set radius of curvature (Rc) values for optical elements.
- Calculate optical path differences, aberrations (Spherical, Coma, Chromatic, and Astigmatism).
- Compute a merit function to evaluate system performance.

Attributes:
- system: The optical system being optimized.
- raykeeper: An object for managing ray tracing and keeping track of results.
- EFFL_Tr: Target effective focal length (EFFL).
- W: Design wavelength for the optimization process.
- Surf: Surface index for the pupil calculation.
- ApVal: Aperture value.
- AperType: Type of aperture used ('EPD').
- Field: Field value for off-axis aberration analysis.

- set_solution: Stores the list of solution values for conic and radius adjustments.
- set_objetivevalue: Stores the objective function values for analysis.

======================================
"""

class Optimizer():
    
    def __init__(self, fun_info):
        """
        Initializes the Optimizer with the optical system, ray keeper, and sets up 
        default parameters for the aperture, surface, wavelength, and effective focal length.

        Parameters:
        - fun_info (list): A list containing:
            - Optical system instance.
            - Ray keeper instance.
        """
        
        # Optical system and ray keeper initialization
        self.system = fun_info[0]
        self.raykeeper = fun_info[1]
        
        
        # Target effective focal length and design wavelength
        self.EFFL_Tr =  9127.198583362275
        self.W = 0.43032015  
        
        # Optical parameters for aperture and surface
        self.Surf = 1
        self.ApVal = 2152
        self.AperType = 'EPD'
        
        # Field for off-axis evaluation
        self.Field = 0.0012216219347070795
        
        # Containers for solutions and objective values
        self.set_solution = []
        self.set_objetivevalue = []

  

    """
    ======================================
      Function: Set_RcValues
    ======================================
    
    This function sets the radius of curvature and thickness values for specific 
    surfaces in the optical system. It then evaluates aberrations (Spherical, Coma, 
    Chromatic, and Astigmatism), computes the merit function, and restores the 
    original system state.

    Parameters:
    - V (list): A list containing the radii of curvature and thickness for 
               the selected optical elements.

    Steps:
    1. Store the curvature values in the solution list.
    2. Apply the new radii of curvature and thickness to the specified surfaces.
    3. Perform a pupil calculation and initialize aberration analysis.
    4. Compute spherical, coma, astigmatism, and chromatic aberrations.
    5. Calculate the total merit function, including deviations from the target EFFL.
    6. Restore the system state and clean the ray keeper.

    Returns:
    - merit_fun (float): The computed merit function value.
    """
    
    def Set_RcValues(self, V):
        
        # Store the curvature values
        Values_CH = V
        self.set_solution.append(Values_CH)
        
        # Set the radii of curvature and thickness for the specified surfaces
        self.system.SDT[3].Rc = Values_CH[0]
        self.system.SDT[4].Rc = Values_CH[1]
        self.system.SDT[5].Rc = Values_CH[2]
        self.system.SDT[6].Rc = Values_CH[3]
        self.system.SDT[6].Thickness = Values_CH[4]
        
        # Apply the changes
        self.system.SetData()
        
        # Calculate the pupil and perform aberration analysis
        self.P = Kos.PupilCalc(self.system, self.Surf, self.W, self.AperType, self.ApVal)
        self.InfSystem = [self.system, self.raykeeper, self.P]
        self.Aberration = Aberration_Info(self.InfSystem, self.W)

        # Compute aberrations and handle nan for coma
        self.valueChrom = self.Aberration.Chromatic(1, [0.0, 0.0])[1] * self.W * 1000
        self.valueShp = self.Aberration.Spheric(1, [0.0, 0.0])[1] * self.W * 1000
        self.valueComa = self.Aberration.Coma(1, self.Field)[1] * self.W * 1000
        
        if np.isnan(self.valueComa):
            self.valueComa = 1e6
        
        self.valueAst = self.Aberration.Astigmatism(1, self.Field)[1] * self.W * 1000
        
        # Compute the merit function
        self.value = 0.5 * (self.valueShp**2 + self.valueComa**2 + self.valueAst**2) + self.valueChrom**2
        
        # Evaluate the final merit function with EFFL deviation
        self.D_EFFL = np.abs(self.system.EFFL - self.EFFL_Tr)
        self.merit_fun = self.value + self.D_EFFL**2

        # print(self.merit_fun)

        # Store the values and restore the system
        self.set_objetivevalue.append([self.value, self.D_EFFL])
        self.system.RestoreData()
        self.raykeeper.clean()
        
        return self.merit_fun
    
        
    

                    
     


















































#############################################################################
#############################################################################
        
            