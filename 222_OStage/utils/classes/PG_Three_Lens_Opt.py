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

class Three_Lens_Optimizer():
    
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
        self.W_1 = 0.35 
        self.W_3 = 0.55
        
        # Optical parameters for aperture and surface
        self.Surf = 1
        self.ApVal = 2152
        self.AperType = 'EPD'
        
        # Field for off-axis evaluation
        self.Field = 0.0012325797337758514
        
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
        
        Units = 1000.
        
        w_1 = (0.5)**2
        w_2 = (1.)**2
        w_3 = (1.)**2
        w_4 = (0.1)**2
        w_5 = (1.)**2
        w_6 = (1.)**2
        w_7 = (1.)**2
        w_8 = (1.)**2
        
        # print('PGMF_info')
        # print(self.W_1, self.W, self.W_3)
        
        # Store the curvature values
        Values_CH = V
        self.set_solution.append(Values_CH)
        
        # Set the radii of curvature and thickness for the specified surfaces
        self.system.SDT[3].Rc = Values_CH[0]
        self.system.SDT[4].Rc = Values_CH[1]
        self.system.SDT[5].Rc = Values_CH[2]
        self.system.SDT[6].Rc = Values_CH[3]
        self.system.SDT[7].Rc = Values_CH[4]
        self.system.SDT[8].Rc = Values_CH[5]
        self.system.SDT[8].Thickness = Values_CH[6]
        
        
        # Apply the changes
        self.system.SetData()
        
        
        print(self.W_1, self.W, self.W_3)
        # Calculate the pupil and perform aberration analysis
        self.P = Kos.PupilCalc(self.system, self.Surf, self.W, self.AperType, self.ApVal)
        self.InfSystem = [self.system, self.raykeeper, self.P]
        self.Aberration = Aberration_Info(self.InfSystem, self.W)
        self.Aberration.dw_1 = self.W_1
        self.Aberration.dw_2 = self.W_3
        
        ###################################################################################
        #First Field Aberration information
        # Compute aberrations and handle nan for coma
        
        # Non Dimensionless
        # self.valueChrom_F1 = w_1*(self.Aberration.Chromatic(1, [0., 0.])[1])
        # self.valueShp_F1   = w_2*(self.Aberration.Spheric(1, [0., 0.])[1])
        # self.valueComa_F1  = w_3*(self.Aberration.Coma(1, [0., -self.Field])[1])
        
        # Dimensionless
        self.valueChrom_F1 = w_1*(self.Aberration.Chromatic(1, [0., 0.])[1])*(Units/self.W)
        self.valueShp_F1   = w_2*(self.Aberration.Spheric(1, [0., 0.])[1])*(Units/self.W)
        self.valueComa_F1  = w_3*(self.Aberration.Coma(1, [0., -self.Field])[1])*(Units/self.W)
        
        if np.isnan(self.valueComa_F1):
            self.valueComa = 1e6
        
        # self.valueAst_F1 = w_4*(self.Aberration.Astigmatism(1, [0., -self.Field])[1])
        
        self.valueAst_F1 = w_4*(self.Aberration.Astigmatism(1, [0., -self.Field])[1])*(Units/self.W)
        
        ###################################################################################
        #Second Field Aberration information
        # Compute aberrations and handle nan for coma
        
        # Non Dimensionless
        # self.valueChrom_F2 = w_5*(self.Aberration.Chromatic(1, [self.Field, -self.Field])[1])
        # self.valueShp_F2   = w_6*(self.Aberration.Spheric(1,   [self.Field, -self.Field])[1])
        # self.valueComa_F2  = w_7*(self.Aberration.Coma(1, [self.Field, -self.Field])[1])
        
        # Dimensionless
        self.valueChrom_F2 = w_5*(self.Aberration.Chromatic(1, [self.Field, -self.Field])[1])*(Units/self.W)
        self.valueShp_F2   = w_6*(self.Aberration.Spheric(1,   [self.Field, -self.Field])[1])*(Units/self.W)
        self.valueComa_F2  = w_7*(self.Aberration.Coma(1, [self.Field, -self.Field])[1])*(Units/self.W)
        
        
        if np.isnan(self.valueComa_F2):
            self.valueComa_F2 = 1e6
        
        
        # Compute the merit function
        self.value = w_1*self.valueChrom_F1**2 + w_2*self.valueShp_F1**2+w_3*self.valueComa_F1**2+w_4*self.valueAst_F1**2
        
        # Evaluate the final merit function with EFFL deviation
        # Non Dimensionless
        # self.D_EFFL = w_8*np.abs(self.system.EFFL - self.EFFL_Tr)
        # Dimensionless
        self.D_EFFL = w_8*np.abs(self.system.EFFL - self.EFFL_Tr)*(Units/self.W)
        
        self.merit_fun = self.value + self.D_EFFL**2
        # print(self.merit_fun)
        
        # Store the values and restore the system
        self.set_objetivevalue.append([self.value, self.D_EFFL])
        
        
        # Clean the raykeepar and restore to initial optical parameters
        self.system.RestoreData()
        self.raykeeper.clean()
    
        return self.valueChrom_F1, self.valueShp_F1, self.valueComa_F1, self.valueAst_F1, self.valueChrom_F2, self.valueShp_F2, self.valueComa_F2, self.D_EFFL
  