# -*- coding: utf-8 -*-
"""
Created on Mon Oct 13 17:16:17 2025

@author: MORGANRHAINAJERAROA
"""


import numpy as np
from ..equations.MOS_equation import BestRMS
from ..classes.Gaussian_Quadrature import Gaussian_Quadrature


#--------------------------#
# RMS Optimization Class
#--------------------------#


class Function2Optimize:
    
    """
    ======================================
      Class: Function2Optimize
    ======================================

    This class defines the optimization process for an optical system across 
    multiple wavelengths and field positions using Gaussian Quadrature ray 
    tracing.

    Attributes:
    - fun_properties (list): List containing the optical system, raykeeper, 
    and pupil instance.
    - system (object): The optical system instance.
    - raykeeper (object): Instance to manage ray data.
    - P (object): Pupil object instance for ray tracing.
    - nodes (int): Number of nodes for Gaussian Quadrature.
    - arms (int): Number of arms for the circular sampling.
    - Fx (list): List of field positions in X.
    - Fy (float): Field position in Y (fixed at 0.0).
    - w1, w2, w3 (float): Wavelengths for evaluation (0.35, design wavelength 
                                                      W, 0.55).
    - effl (float): Expected effective focal length for the system.
    - result (list): Stores the optimization results.
    ======================================
    """
    
    def __init__(self, fun_info, W):
        """
        Initializes the Function2Optimize class with the optical system information and design wavelength.
        
        Parameters:
        - fun_info (list): Contains the system, raykeeper, and pupil instances.
        - W (float): Design wavelength.
        """
        self.fun_properties = fun_info 
        self.system = self.fun_properties[0]
        self.raykeeper = self.fun_properties[1]
        self.P = self.fun_properties[2]
        
        # Configuration parameters
        self.nodes = 3
        self.arms = 6
        self.Fx = [0.0, 0.07062161665871489]
        self.Fy = 0.0
        
        self.Units = 1000.
        # Wavelength definitions
        self.w1 = 0.35
        self.w2 = W
        self.w3 = 0.55
        
        # Effective focal length target
        self.effl =  9127.198583362275
        self.result = []
        
    def EFFL_3W(self, V): 
        """
        Optimizes the Effective Focal Length (EFFL) across three wavelengths 
        and two field positions, using Gaussian Quadrature for ray sampling.

        Parameters:
        - V (list): List of radii of curvature to be optimized.

        Returns:
        - list: Contains differences in EFFL, wavelength spread, and RMS 
        results 
          for both fields.
        """

        # Update system radii of curvature with the new parameters
        self.system.SDT[3].Rc = V[0]
        self.system.SDT[4].Rc = V[1]
        self.system.SDT[5].Rc = V[2]
        self.system.SDT[6].Rc = V[3]
        print(V[0], V[1], V[2], V[3])
        # Apply the changes to the optical system
        self.system.SetData()
        
        # Bundle system information
        self.InfSystem = [self.system, self.raykeeper, self.P]
        
        ########################################################################
        #                             First Field                              #
        ########################################################################
        
        # Perform Gaussian Quadrature ray tracing for three wavelengths
        self.gqa = Gaussian_Quadrature(self.InfSystem, self.w1)
        self.gqb = Gaussian_Quadrature(self.InfSystem, self.w2)
        self.gqc = Gaussian_Quadrature(self.InfSystem, self.w3)
        
        # Compute the coordinates for the first field
        self.gqa.Coordinates_GQ(self.nodes, self.arms, self.Fx[0], 0.0, 0)
        self.gqb.Coordinates_GQ(self.nodes, self.arms, self.Fx[0], 0.0, 0)
        self.gqc.Coordinates_GQ(self.nodes, self.arms, self.Fx[0], 0.0, 0)
        
        # # Get the effective focal lengths for each wavelength
        # EFFL1 = self.gqa.EFFL_GQ
        EFFL0 = self.gqb.EFFL_GQ
        # EFFL2 = self.gqc.EFFL_GQ
        
        # print(EFFL1, EFFL0, EFFL2)
        
        # # Calculate spread in focal lengths
        # ra_f1, rb_f1 = (EFFL0 - EFFL1), (EFFL0 - EFFL2)
        # r_f1 = np.sqrt(ra_f1**2 + rb_f1**2)
        
        # print(r_f1)
        
        # Compute the deviation from the target effective focal length
        D_EFFL = np.abs(EFFL0 - self.effl)
        
        # Extract coordinates
        xa, ya, za, la, ma, na = self.gqa.Coordinates_GQ(self.nodes, self.arms, self.Fx[0], 0.0, 0)
        xb, yb, zb, lb, mb, nb = self.gqb.Coordinates_GQ(self.nodes, self.arms, self.Fx[0], 0.0, 0)
        xc, yc, zc, lc, mc, nc = self.gqc.Coordinates_GQ(self.nodes, self.arms, self.Fx[0], 0.0, 0)

        # Concatenate all results
        all_points_x = np.concatenate((xa, xb, xc))
        all_points_y = np.concatenate((ya, yb, yc))
        all_points_z = np.concatenate((za, zb, zc))
        all_points_l = np.concatenate((la, lb, lc))
        all_points_m = np.concatenate((ma, mb, mc))
        all_points_n = np.concatenate((na, nb, nc))
        
        setsep_coscoor = [all_points_x, all_points_y, all_points_z, all_points_l, all_points_m,
                          all_points_n] 
        
        
        
        Ba_f0 = np.array(BestRMS(setsep_coscoor, self.system))
        
        RMS_0 = Ba_f0
        
        
        # Extract coordinates
        xa, ya, za, la, ma, na = self.gqa.Coordinates_GQ(self.nodes, self.arms, self.Fx[1], 0.0, 0)
        xb, yb, zb, lb, mb, nb = self.gqb.Coordinates_GQ(self.nodes, self.arms, self.Fx[1], 0.0, 0)
        xc, yc, zc, lc, mc, nc = self.gqc.Coordinates_GQ(self.nodes, self.arms, self.Fx[1], 0.0, 0)

        # Concatenate all results
        all_points_x = np.concatenate((xa, xb, xc))
        all_points_y = np.concatenate((ya, yb, yc))
        all_points_z = np.concatenate((za, zb, zc))
        all_points_l = np.concatenate((la, lb, lc))
        all_points_m = np.concatenate((ma, mb, mc))
        all_points_n = np.concatenate((na, nb, nc))
        
        setsep_coscoor = [all_points_x, all_points_y, all_points_z, all_points_l, all_points_m,
                          all_points_n] 
        
        
        
        Ba_f1 = np.array(BestRMS(setsep_coscoor, self.system))
        
        RMS_1 = Ba_f1
        
       
        # ########################################################################
        #                             Second Field                             #
        ########################################################################
        
        # Perform Gaussian Quadrature ray tracing for three wavelengths in the second field
        xa, ya, za, la, ma, na = self.gqa.Coordinates_GQ(self.nodes, self.arms, self.Fx[1], -self.Fx[1], 0)
        xb, yb, zb, lb, mb, nb = self.gqb.Coordinates_GQ(self.nodes, self.arms, self.Fx[1], -self.Fx[1], 0)
        xc, yc, zc, lc, mc, nc = self.gqc.Coordinates_GQ(self.nodes, self.arms, self.Fx[1], -self.Fx[1], 0)
        
        
        # Concatenate all results
        all_points_x = np.concatenate((xa, xb, xc))
        all_points_y = np.concatenate((ya, yb, yc))
        all_points_z = np.concatenate((za, zb, zc))
        all_points_l = np.concatenate((la, lb, lc))
        all_points_m = np.concatenate((ma, mb, mc))
        all_points_n = np.concatenate((na, nb, nc))
         
        setsep_coscoor = [all_points_x, all_points_y, all_points_z, all_points_l, all_points_m,
                           all_points_n] 
         
        Ba_f2 = np.array(BestRMS(setsep_coscoor, self.system))
        RMS_2 = Ba_f2
        
        
        print(RMS_2*1000, RMS_1*1000, RMS_0*1000)
        
        self.system.RestoreData()
        
        return [ RMS_2, RMS_1, RMS_0, D_EFFL ]
