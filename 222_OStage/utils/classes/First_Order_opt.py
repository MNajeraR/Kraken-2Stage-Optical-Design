# -*- coding: utf-8 -*-
"""
Created on Mon Oct 13 16:00:10 2025

@author: MORGANRHAINAJERAROA
"""

import numpy as np

#--------------------------#
# First Order Optimization Class
#--------------------------#

"""
======================================
  Class: Paraxial_Cal
======================================

This class encapsulates the paraxial calculations for a telescope system 
and its associated focal reducer, considering a specified reduction 
percentage. It computes the ABCD matrices for the telescope and the 
focal reducer, as well as solving the paraxial equations to achieve 
desired optical properties.

Initialization Parameters:
- percent (float): Reduction percentage for the telescope focal length.

Attributes:
- EFFL_Tel (float): Effective focal length of the telescope.
- red_per (float): Reduction percentage for the focal length.
- b_1, b_2 (float): Optical power of the primary and secondary mirrors.
- d_1, d_2, d_3, d_4 (float): Optical distances between components.
- d_Obj (float): Hypothetical object distance.
- Tr_per (float): Transmission percentage of the optical system.
- EFFL_Tr (float): Effective focal length of the system after reduction.

======================================
"""

class Paraxial_Cal:
    
    def __init__(self, percent):
        """
        Initializes the Paraxial_Cal class with the specified reduction percentage 
        and calculates the effective focal length, transmission percentage, and 
        optical distances for the telescope system.

        Parameters:
        - percent (float): Reduction percentage for the telescope focal length.
        """
        
        # -------------------------------------
        #   System Constants and Configurations
        # -------------------------------------
        self.EFFL_Tel = 18273.877041856547    # Effective focal length of the telescope
        self.red_per = percent                # Reduction percentage
        
        # -------------------------------------
        #   Optical Power Calculations
        # -------------------------------------
        self.b_1 = -1 / (1.118E004 / 2)       # Optical power of the primary mirror
        self.b_2 = -1 / (-4430. / 2)          # Optical power of the secondary mirror
        
        # -------------------------------------
        #    Distance Definitions
        # -------------------------------------
        self.d_1 = 4052.571043                # Distance between primary and secondary mirrors
        self.d_2 = 622                        # Distance to the focal reducer
        self.d_3 = 12.1858 + 10 + 7           # Distance for optical correction
        self.d_4 = 111.5998648                # Distance to the final image plane
        self.d_Obj = 1.0                      # Hypothetical object distance

        # -------------------------------------
        #   Transmission Percentage Calculation
        # -------------------------------------
        self.Tr_per = 100 - self.red_per
        self.EFFL_Tr = self.EFFL_Tel * (self.Tr_per / 100)

    """
    ======================================
      Method: MS_Telescope
    ======================================
    """
    def MS_Telescope(self, d):
        """
        Constructs the ABCD matrix for the telescope system, incorporating 
        the distances and optical powers of the primary and secondary mirrors.

        Parameters:
        - d (float): Distance variation for the optical configuration.

        Returns:
        - MSJS (np.array): The resulting ABCD matrix representing the 
                           telescope optical path.
        """
        
        # -------------------------------------
        #   Matrix Construction for Telescope
        # -------------------------------------
        # The system is represented by a sequence of ABCD matrices:
        # - M6: Propagation through distance d_1 + d
        # - M7: Reflection from secondary mirror (optical power b_2)
        # - M8: Propagation through distance d_1
        # - M9: Reflection from primary mirror (optical power b_1)
        # - M10: Final propagation to the focal plane
        # -------------------------------------
        self.M6 = np.array([1, 0, self.d_1 + d, 1]).reshape(2, 2)
        self.M7 = np.array([1, self.b_2, 0, 1]).reshape(2, 2)
        self.M8 = np.array([1, 0, self.d_1, 1]).reshape(2, 2)
        self.M9 = np.array([1, self.b_1, 0, 1]).reshape(2, 2)
        self.M10 = np.array([1, 0, 1, 1]).reshape(2, 2)
        
        # Compute the total ABCD matrix for the telescope system
        self.MSJS = self.M6 @ self.M7 @ self.M8 @ self.M9 @ self.M10
        
        return self.MSJS

    """
    ======================================
      Method: MS_FocalReducer
    ======================================
    """
    def MS_FocalReducer(self, x_1, x_2):
        """
        Constructs the ABCD matrix for the focal reducer, combined with the 
        telescope system, to achieve the desired focal reduction and image 
        formation.

        Parameters:
        - x_1 (float): Optical power adjustment for the first lens.
        - x_2 (float): Optical power adjustment for the second lens.

        Returns:
        - MSC1 (np.array): The ABCD matrix for the optical instrument.
        - MSC (np.array): The complete ABCD matrix including the telescope system.
        """
        
        # -------------------------------------
        #   Generate Telescope Matrix
        # -------------------------------------
        self.Matrix_Telescope = self.MS_Telescope(self.d_2)
        
        # -------------------------------------
        #   Matrix Construction for Focal Reducer
        # -------------------------------------
        # M0 -> Propagation to the final image plane (d_4)
        # M1 -> Lens 2 with optical power x_2
        # M2 -> Propagation to Lens 1 (d_3)
        # M3 -> Lens 1 with optical power x_1
        # -------------------------------------
        self.M0 = np.array([1, 0, self.d_4, 1]).reshape(2, 2)
        self.M1 = np.array([1, x_2, 0, 1]).reshape(2, 2)
        self.M2 = np.array([1, 0, self.d_3, 1]).reshape(2, 2)
        self.M3 = np.array([1, x_1, 0, 1]).reshape(2, 2)
        
        # Compute the total ABCD matrix for the focal reducer and telescope
        self.MSC1 = self.M2 @ self.M3 @ self.Matrix_Telescope
        self.MSC2 = self.M1 @ self.MSC1
        self.MSC = self.M0 @ self.MSC2
        
        return self.MSC1, self.MSC

    """
    ======================================
      Method: Prx_equation
    ======================================
    """
    def Prx_equation(self, variables):
        """
        Solves the paraxial equations for the focal reducer by comparing 
        the calculated optical power with the desired target.

        Parameters:
        - variables (tuple): Contains the optical powers (x_1, x_2) of the lenses.

        Returns:
        - b (float): Difference between the actual and desired optical power.
        - d (float): D term of the ABCD matrix for system stability.
        """
        
        # Unpack lens powers
        self.b3, self.b4 = variables
        
        # Generate the ABCD matrix for the focal reducer
        self.Ma, self.Mt = self.MS_FocalReducer(self.b3, self.b4)
        
        # Desired optical power based on the effective focal length
        self.b_t = -1 / self.EFFL_Tr
        
        # Compute deviation from the target power and the determinant term
        self.b = self.Mt[0][1] - self.b_t
        self.d = self.Mt[1][1]
        
        return self.b, self.d