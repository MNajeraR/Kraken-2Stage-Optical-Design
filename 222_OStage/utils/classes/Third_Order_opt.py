# -*- coding: utf-8 -*-
"""
Created on Mon Oct 13 16:44:03 2025

@author: MORGANRHAINAJERAROA
"""

import numpy as np
from ..equations.opt_ecuation import Lens_Maker, Prin_Plane
import KrakenOS as Kos


#--------------------------#
# Third Order Optimization Class
#--------------------------#


"""
======================================
  Class: ThirdOrder_Cal
======================================

This class encapsulates the third-order aberration calculations and lens 
optimization for a given optical system. It includes methods for lens 
maker's equation, principal plane calculation, parameter seeding, and 
merit function evaluation for optical optimization.

Initialization Parameters:
- SeidelTool: A tool for computing Seidel aberration terms.
- info_lens: A tuple containing:
    - Lens thicknesses (d_l1, d_l2)
    - Lens bending parameters (b3, b4)
    - Refractive indices of the lenses (n_L1, n_L2)

Attributes:
- W (float): Design wavelength for optimization.
- d_1, d_2, d_3 (float): Predefined distances for the optical system.
- EFFL_Tr (float): Target Effective Focal Length (EFFL).

======================================
"""

class ThirdOrder_Cal:
    
    def __init__(self, System, lens_power):
        """
        Initializes the ThirdOrder_Cal class with necessary parameters.

        Parameters:
        - SeidelTool: Instance to calculate Seidel aberrations.
        - info_lens (tuple): Contains lens thickness, bending parameters, 
                             and refractive indices.
        """
        
        self.System = System
        
        # Wavelength and system distances
        self.W = 0.43032015
        
        self.d_2  = self.System.SDT[2].Thickness
        self.d_3 = self.System.SDT[4].Thickness

        self.EFFL_Tr =  9127.198583362275         # Target Effective Focal Length        
        
        Initial_parax = self.System.Parax(self.W)
        Initial_n = Initial_parax[11]
        
        self.d_L1 = self.System.SDT[3].Thickness  # Lens thicknesses
        self.d_L2 = self.System.SDT[5].Thickness  # Lens thicknesses
        self.b3, self.b4 = lens_power[0], lens_power[1]   # Bending parameters
        self.n_L1 = Initial_n[3]                  # Refractive indices
        self.n_L2 = Initial_n[5]                  # Refractive indices
        
       # --- History---
        self.history = {
                'iter': [],
                'PL_fun': [],
                'Aberration_fun': [],
                'D_EFFL': [],
                'd': [],
                'Merit_fun': [],
                'B_1': [],
                'B_2': [],
                'F_1': [],
                'F_2': []}
        self._iter_counter = 0
    
    
     
    """
    ======================================
      Method: SeedPar
    ======================================
    """
    
    
    def SeedPar(self, variables):
        """
        Seeds the optical system with initial lens parameters and 
        performs the aberration and merit function calculation.

        Parameters:
        - variables (tuple): Contains the radii of curvature for the 
                             lenses and the separation distance.
        
        Returns:
        - list: Effective radii, merit function value, EFFL deviation, 
                and the calculated parameter 'd'.
        """
        
        #Define constants
        Units = 1000.0
        
        # Define the parameters for the pupil calculation:
        W = 0.43032015       # Reference wavelength in micrometers
        sup = 1              # Number of the surface representing the opening of the system
        AperType = "EPD"     # AperType sets the aperture ("STOP") or entrance pupil diameter ("EPD").
        AperVal = 2152.0 # Diameter of the entrance pupil
        
        
        #Define constrain 
        w_1 = (1.0)**2
        w_2 = (1.0)**2
        # w_3 = (0.3043)**2
        # w_4 = (0.3043)**2
        # w_5 = (0.3043)**2
        w_3 = (1)**2
        w_4 = (1)**2
        w_5 = (1)**2
        w_6 = (1.)**2
        # w_6 = (0.3043)**2
        w_7 = (1.)**2
        w_8 = (1.)**2
          
        # Unpack the variables
        self.R1, self.R2, self.R3, self.R4, self.d4 = variables
        # print(self.R1, self.R2, self.R3, self.R4, self.d4)
        # Access the optical system
        # self.System = self.SeidelInfo.SYSTEM
        
        ###########################################################################################
        
        # Set radii of curvature in the optical system
        
        self.System.SDT[3].Rc = self.R1
        self.System.SDT[4].Rc = self.R2
        self.System.SDT[5].Rc = self.R3
        self.System.SDT[6].Rc = self.R4
        
         
        # Calculate the principal planes
        
        self.H1_a, self.H2_a = Prin_Plane(self.n_L1, self.R1, self.R2, self.d_L1) 
        self.H1_b, self.H2_b = Prin_Plane(self.n_L2, self.R3, self.R4, self.d_L2)
        
        
        self.System.SDT[2].Thickness = self.d_2 - self.H1_a
        self.System.SDT[4].Thickness = self.d_3 - self.H2_a - self.H1_b
        self.System.SDT[6].Thickness = self.d4  - self.H2_b
        
        
        # Update the system's internal parameters
        
        self.System.SetData()
    
        ###########################################################################################
        
        # Calculate the effective lens radii

        
        self.theoric_p1 = Lens_Maker(self.n_L1, self.R1, self.R2, self.d_L1)
        self.theoric_p2 = Lens_Maker(self.n_L2, self.R3, self.R4, self.d_L2)
        
        
        self.Dif_b1 = np.abs((-1/self.theoric_p1)-(-1/self.b3))
        
        # print(-1/self.theoric_p1)
        # print(-1/self.b3)
        # print(self.Dif_b1)
        
        self.Dif_b2 = np.abs((-1/self.theoric_p2)-(-1/self.b4))
        
        self.B1 = (w_1*self.Dif_b1*(self.W/Units))
        self.B2 = (w_2*self.Dif_b2*(self.W/Units))
        
        self.PL_fun = w_1*self.B1**2 + w_2*self.B2**2
        
        ###########################################################################################
       
        # Initialize the Pupil calculation for the system
        self.Pup = Kos.PupilCalc(self.System, sup, W, AperType, AperVal)

        # Configure the sampling for the pupil and its field representation:
        self.Pup.Samp = 7         # Integer number for pupil ray sampling
        self.Pup.FieldType = "angle" # Field type, this in terms of object height and distance from the plane.
        self.Pup.FieldX = 0.07062161665871489 # Field value in degrees on the X-axis
        
        # Calculate Seidel aberrations
        self.AB = Kos.Seidel(self.Pup)
        self.AB.Wf = 0.35                    # Set the first adjacent design wavelength (shorter) 
        self.AB.Wd = W                       # Set the center design wavelength
        self.AB.Wc = 0.5499996               # Set the second adjacent design wavelength (longer)
        # self.AB.calculate()
        
        # Compute each aberration term
        self.Sph = self.AB.SAC_TOTAL[0]*(Units/self.W)
        self.Coma = self.AB.SAC_TOTAL[1]*(Units/self.W)
        self.Ast = self.AB.SAC_TOTAL[2]*(Units/self.W)
        self.CLon = np.sum(self.AB.CL)*(Units/self.W)
        
        # Aberration function calculation
        self.Aberration_fun = w_3*(self.Sph)**2 + w_4*(self.Coma)**2 + w_5*(self.Ast)**2 + w_6*(self.CLon)**2

        ###########################################################################################

        # Compute paraxial data
        self.Prx = np.array(self.System.Parax(self.W)[0])        
        self.d = w_7*(self.Prx[1][1]*(Units/self.W))
        
        ###########################################################################################
        
        # EFFL deviation from the target
        self.D_EFFL = w_8*((self.System.EFFL - self.EFFL_Tr)*(Units/self.W))
        

        
        ###########################################################################################
        
        #Possible merit function
        self.Merit_fun = (self.PL_fun + self.Aberration_fun + self.D_EFFL**2 + self.d**2)
        
        ###########################################################################################
        # --- Safe History ---
    
        self._iter_counter += 1
        self.history['iter'].append(self._iter_counter)
        self.history['B_1'].append(float((self.B1)))
        self.history['B_2'].append(float((self.B2)))
        self.history['F_1'].append(float((1/self.B1)))
        self.history['F_2'].append(float((1/self.B2)))
        self.history['Aberration_fun'].append(float(self.Aberration_fun))
        self.history['D_EFFL'].append(float(self.D_EFFL))
        self.history['d'].append(float(self.d))
        self.history['Merit_fun'].append(float(self.Merit_fun))
        
        ###########################################################################################

        # Restore the original system state
        self.System.RestoreData()
        
        
        return  [self.B1, self.B2, self.Aberration_fun, self.D_EFFL, self.d]