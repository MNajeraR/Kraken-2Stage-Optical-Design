# -*- coding: utf-8 -*-
"""
Created on Mon Oct 13 17:24:33 2025

@author: MORGANRHAINAJERAROA
"""

import numpy as np

from ..equations.tracing import ProcessPattern2Field

from ..equations.generating import (generate_radios, generate_angles, 
                                    generate_radiosc, generate_anglesc, 
                                    generate_radiosa, generate_anglesa) 
from ..equations.MOS_equation import (Sphere_subtract, Astigmatism_Substraction, 
                                      Coma_Substraction)
from ..equations.ops import calculate_average


#--------------------------#
# Physical Grounded Aberration Class
#--------------------------#

"""
======================================
  Class: Aberration_Info
======================================

This class performs the calculation of optical aberrations based on physical 
principles and ray tracing data. The aberrations calculated include:

    - Chromatic Aberration: Quantifies axial chromatic displacement.
    - Spherical Aberration: Measures deviations in the optical path 
                            between marginal and chief rays.
    - Coma Aberration: Analyzes off-axis rays to identify deviations from 
                       the ideal optical axis.
    - Astigmatism Aberration: Evaluates the difference in focal length 
                              between sagittal and tangential planes.

The calculations are grounded in classical optical theory, including Fermat's 
Principle and ray propagation laws. These are critical for evaluating image 
quality and optimizing the design of optical systems.

Initialization Parameters:
- fun_info (tuple): Contains:
    - Optical system instance.
    - Raykeeper for managing traced rays.
    - Pupil instance for sampling the entrance aperture.
- W (float): Design wavelength for ray tracing.

Attributes:
- w (float): Design wavelength.
- dw_1, dw_2 (float): Secondary wavelengths for chromatic analysis.
======================================
"""

class Aberration_Info:
    
    def __init__(self, fun_info, W):
        """
        Initializes the Aberration_Info class with the optical system, 
        raykeeper, and pupil information.

        Parameters:
        - fun_info (tuple): Contains the optical system, raykeeper, and pupil.
        - W (float): Design wavelength for ray tracing.
        """
        self.fun_propeties = fun_info 
        self.system = self.fun_propeties[0]
        self.raykeeper = self.fun_propeties[1]
        self.P = self.fun_propeties[2]
        
        # Design and secondary wavelengths for chromatic analysis
        self.w = W
        self.dw_1 = 0.35
        self.dw_2 = 0.55
        
        # print('AB_info')
        # print(self.dw_1, self.w, self.dw_2)
    """
    ======================================
      Method: Chromatic
    ======================================
    """
    def Chromatic(self, Sample, Campo):
        """
        Calculates the chromatic aberration for a given field.
        
        The chromatic aberration is quantified by evaluating the 
        Total Optical Path (TOP) for marginal and chief rays at three wavelengths. 
        It uses the principle of chromatic path equalization to minimize 
        axial displacement over the spectrum.

        Parameters:
        - Sample (int): Number of samples for the analysis.
        - Campo (tuple): Coordinates of the field to analyze.

        Returns:
        - r (list): Chromatic aberration values for each ray.
        - r_prom (float): Average chromatic aberration across samples.
        """
        # Initialize the field coordinates and chief ray
        self.clean = 1
        self.CampoX, self.CampoY = np.rad2deg(Campo[0]), np.rad2deg(Campo[1])
        
        # Chief ray (reference for TOP)
        self.PTsChief = ['chief']
        self.PRsChief = [np.nan]
        self.PtsChief = [np.nan]
        self.SCRadio = np.array([0])

        # Generate marginal rays
        self.PTsrtheta = ['rtheta'] * Sample
        self.PRsrtheta = generate_radios(Sample)
        self.Ptsrtheta = generate_angles(Sample)
        self.SMRadios = np.array(self.PRsrtheta) * self.P.RadPupInp

        # Combine chief and marginal rays
        self.SRadios = np.concatenate((self.SCRadio, self.SMRadios))  
        self.ChromType = self.PTsChief + self.PTsrtheta
        self.ChromRad = self.PRsChief + self.PRsrtheta
        self.ChromTh = self.PtsChief + self.Ptsrtheta
        self.ChromFX = [self.CampoX] * (Sample + 1)
        self.ChromFY = [self.CampoY] * (Sample + 1)

        # Initialize arrays for Total Optical Path (TOP)
        self.Rset_TOP_1 = np.ones(len(self.ChromType))
        self.Rset_TOP_2 = np.ones(len(self.ChromType))
        self.Rset_TOP_3 = np.ones(len(self.ChromType))

        # Perform ray tracing for three wavelengths
        if self.clean == 1:
            self.raykeeper.clean()
            self.system = self.raykeeper.SYSTEM
        
        print(self.dw_1, self.w, self.dw_2)
        for i, (ptype, prad, ptheta, fx, fy) in enumerate(zip(self.ChromType,
                                                              self.ChromRad,
                                                              self.ChromTh,
                                                              self.ChromFX,
                                                              self.ChromFY)):
            self.P.Ptype, self.P.rad, self.P.theta, self.P.FieldX, self.P.FieldY = ptype, prad, ptheta, fx, fy
           
            ProcessPattern2Field(self.w, self.fun_propeties)
            self.Rset_TOP_1[i] = self.system.TOP
            self.raykeeper.clean()
            
            ProcessPattern2Field(self.dw_1, self.fun_propeties)
            self.Rset_TOP_2[i] = self.system.TOP
            self.raykeeper.clean()
           
            ProcessPattern2Field(self.dw_2, self.fun_propeties)
            self.Rset_TOP_3[i] = self.system.TOP
            self.raykeeper.clean()

        # Compute TOP differences
        self.result_chrom_1 = Sphere_subtract(self.Rset_TOP_1)[1:]
        self.prom_chorm_1 = calculate_average(self.result_chrom_1) 
        
        self.result_chrom_2 = Sphere_subtract(self.Rset_TOP_2)[1:]
        self.prom_chorm_2 = calculate_average(self.result_chrom_2) 

        self.result_chrom_3 = Sphere_subtract(self.Rset_TOP_3)[1:]
        self.prom_chorm_3 = calculate_average(self.result_chrom_3) 
 
        # Compute chromatic deviations
        self.ra, self.rb, self.r = [], [], []
        for v1, v2, v3 in zip(self.result_chrom_1, self.result_chrom_2, self.result_chrom_3):
            self.ra.append(v1 - v2)
            self.rb.append(v1 - v3)
            self.r.append(np.sqrt((self.ra[-1]**2) + (self.rb[-1]**2)))

        # Calculate average spread
        self.ra_prom = self.prom_chorm_1 - self.prom_chorm_2
        self.rb_prom = self.prom_chorm_1 - self.prom_chorm_3
        self.r_prom = np.sqrt((self.ra_prom**2) + (self.rb_prom**2))
        
        return self.r, self.r_prom
    
        
    """
    ======================================
      Method: Spheric
    ======================================
    """
    def Spheric(self, Sample, Campo):
        """
        Calculates the spherical aberration for a specified field.

        Spherical aberration is quantified by analyzing the deviation of 
        marginal rays from the chief ray in terms of the Total Optical Path (TOP).

        Parameters:
        - Sample (int): Number of samples for the analysis.
        - Campo (tuple): Coordinates of the field to analyze.

        Returns:
        - result_sph (list): Spherical aberration values for each ray.
        - prom_sph (float): Average spherical aberration across samples.
        """
        # Initialize the field coordinates and chief ray
        self.clean = 1
        self.CampoX, self.CampoY = np.rad2deg(Campo[0]), np.rad2deg(Campo[1])
        
        # Chief ray (reference for TOP)
        self.PTsChief = ['chief']
        self.PRsChief = [np.nan]
        self.PtsChief = [np.nan]
        self.SCRadio = np.array([0])

        # Generate marginal rays
        self.PTsrtheta = ['rtheta'] * Sample
        self.PRsrtheta = generate_radios(Sample)
        self.Ptsrtheta = generate_angles(Sample)
        self.SMRadios = np.array(self.PRsrtheta) * self.P.RadPupInp

        # Combine chief and marginal rays
        self.SRadios = np.concatenate((self.SCRadio, self.SMRadios))  
        self.SpherType = self.PTsChief + self.PTsrtheta
        self.SpherRad = self.PRsChief + self.PRsrtheta
        self.SpherTh = self.PtsChief + self.Ptsrtheta
        self.SpherFX = [self.CampoX] * (Sample + 1)
        self.SpherFY = [self.CampoY] * (Sample + 1)

        # Initialize arrays for Total Optical Path (TOP)
        self.Rset_TOP = np.ones(len(self.SpherType))

        # Perform ray tracing for the design wavelength
        if self.clean == 1:
            self.raykeeper.clean()
            self.system = self.raykeeper.SYSTEM
        
        for i, (ptype, prad, ptheta, fx, fy) in enumerate(zip(self.SpherType,
                                                              self.SpherRad,
                                                              self.SpherTh,
                                                              self.SpherFX,
                                                              self.SpherFY)):
            # Define the ray properties and perform ray tracing
            self.P.Ptype, self.P.rad, self.P.theta, self.P.FieldX, self.P.FieldY = ptype, prad, ptheta, fx, fy
            ProcessPattern2Field(self.w, self.fun_propeties)
            self.Rset_TOP[i] = self.system.TOP
            self.raykeeper.clean()

        # Compute TOP differences for spherical aberration
        self.result_sph = Sphere_subtract(self.Rset_TOP)[1:]
        self.prom_sph = calculate_average(self.result_sph) 

        # Return the spherical aberration analysis
        return self.result_sph, self.prom_sph
    

    """
    ======================================
      Method: Coma
    ======================================
    """
    
    def Coma(self, Sample, Campo):
        """
        Calculates the coma aberration for a specified field.

        Coma aberration is analyzed by finding the minimal intersection points 
        of marginal rays with respect to the optical axis and comparing their 
        radial distances to the chief ray.

        Parameters:
        - Sample (int): Number of samples for the analysis.
        - Campo (float): Coordinate of the field to analyze.

        Returns:
        - result_coma (list): Coma aberration values for each ray.
        - prom_coma (float): Average coma aberration across samples.
        """
        # Initialize the field coordinates and chief ray
        self.clean = 1
        # print(Campo)
        self.CampoX, self.CampoY = np.rad2deg(Campo[0]), np.rad2deg(Campo[1])
        
        # Chief ray (reference for TOP)
        self.PTcChief = ['chief']
        self.PRcChief = [np.nan]
        self.PtcChief = [np.nan]
        self.CCRadio = np.array([0])

        # Generate marginal rays
        self.PTcrtheta = ['rtheta'] * (2 * Sample)
        self.PRcrtheta, self.randomcradios = generate_radiosc(Sample)
        self.Ptcrtheta = generate_anglesc(Sample)

        # Apply pupil scaling
        self.randomcradios = np.array(self.randomcradios)
        self.CMRadios = self.randomcradios * self.P.RadPupInp

        # Combine chief and marginal rays
        self.CRadios = np.concatenate((self.CCRadio, self.CMRadios))
        self.ComaType = self.PTcChief + self.PTcrtheta
        self.ComaRad = self.PRcChief + self.PRcrtheta
        self.ComaTh = self.PtcChief + self.Ptcrtheta
        self.ComaFX = [self.CampoX] * (2 * Sample + 1)
        self.ComaFY = [self.CampoY] * (2 * Sample + 1)

        # Perform ray tracing
        if self.clean == 1:
            self.raykeeper.clean()
            self.system = self.raykeeper.SYSTEM
        
        for i, (ptype, prad, ptheta, fx, fy) in enumerate(zip(self.ComaType,
                                                              self.ComaRad,
                                                              self.ComaTh,
                                                              self.ComaFX,
                                                              self.ComaFY)):
            # Define the ray properties and perform ray tracing
           
            self.P.Ptype, self.P.rad, self.P.theta, self.P.FieldX, self.P.FieldY = ptype, prad, ptheta, fx, fy
            # print(fx, fy)
            ProcessPattern2Field(self.w, self.fun_propeties)
            self.X, self.Y, self.Z, self.L, self.M, self.N = self.raykeeper.pick(-1)
            self.XYZ0, self.LMN0 = (self.X, self.Y, self.Z), (self.L, self.M, self.N)
            self.Inf_COMA = (self.XYZ0, self.LMN0)
            
        # Compute Coma aberration
        if len(self.X) == 3:
            self.result_coma = Coma_Substraction(Sample, self.Inf_COMA)
            self.prom_coma = calculate_average(self.result_coma)
        else:
            self.result_coma = [np.nan] * Sample
            self.prom_coma = np.nan

        # Return the coma aberration analysis
        return self.result_coma, self.prom_coma
    
        
    """
    ======================================
      Method: Astigmatism
    ======================================
    """
    def Astigmatism(self, Sample, Campo):
        """
        Calculates the astigmatism aberration for a specified field.
    
        Astigmatism is evaluated by analyzing the absolute difference 
        between the sagittal and tangential rays at their minimal 
        intersection points. The separation between these focal lines 
        represents the level of astigmatism.
    
        Parameters:
        - Sample (int): Number of samples for the analysis.
        - Campo (float): Coordinate of the field to analyze.
    
        Returns:
        - result_ast (list): Astigmatism aberration values for each ray.
        - prom_ast (float): Average astigmatism aberration across samples.
        """
        
        # Initialize the field coordinates and chief ray
        self.clean = 1
        self.CampoX, self.CampoY = np.rad2deg(Campo[0]), np.rad2deg(Campo[1])
        
        # Scaling factor for sagittal and tangential rays
        c = 10000 
        
        # Sagittal and Tangential Rays Definition
        # Astigmatism is characterized by the difference in focal points 
        # in the sagittal (vertical plane) and tangential (horizontal plane).
        self.PTartheta = ['rtheta'] * (4 * Sample)
        self.PRartheta, self.randomaradios = generate_radiosa(Sample, c)
        self.Ptartheta = generate_anglesa(Sample)
    
        # Field coordinates for astigmatic analysis
        self.AstFX = [self.CampoX] * (4 * Sample)
        self.AstFY = [self.CampoY] * (4 * Sample)
    
        # Apply pupil scaling to the generated radii
        self.randomaradios = np.array(self.randomaradios)
        self.AMRadios = (self.randomaradios / c) * self.P.RadPupInp
        
        # Initialize Ray Tracing for the specified field configuration
        if self.clean == 1:
            self.raykeeper.clean()
            self.system = self.raykeeper.SYSTEM
    
        # Loop through all the sagittal and tangential rays
        for i, (ptype, prad, ptheta, fx, fy) in enumerate(zip(self.PTartheta, 
                                                              self.PRartheta, 
                                                              self.Ptartheta, 
                                                              self.AstFX, 
                                                              self.AstFY)):
            # Define the ray properties and perform ray tracing
            self.P.Ptype = ptype
            self.P.rad = prad
            self.P.theta = ptheta
            self.P.FieldX = fx
            self.P.FieldY = fy
            ProcessPattern2Field(self.w, self.fun_propeties)
            
            # Store the traced ray's coordinates and direction cosines
            self.X, self.Y, self.Z, self.L, self.M, self.N = self.raykeeper.pick(3)
            self.XYZ0, self.LMN0 = (self.X, self.Y, self.Z), (self.L, self.M, self.N)
            self.Inf_Ast = (self.XYZ0, self.LMN0)
            
            # print(self.Inf_Ast)
    
        # Compute Astigmatism Aberration
        # The function `Astigmatism_Substraction` calculates the minimum distance 
        # intersection points for sagittal and tangential rays.
        if len(self.X) == 4:
            # Calculate the deviation between sagittal and tangential rays
            self.result_ast = Astigmatism_Substraction(Sample, self.Inf_Ast)
            # Compute the average of the astigmatism aberration values
            self.prom_ast = calculate_average(self.result_ast)
        
        else:
            # If the rays are not properly traced, return nans
            self.result_ast = [np.nan] * Sample
            self.prom_ast = np.nan
        
        # Return the Results
        # - result_ast: Absolute differences for each sampled ray in the astigmatism analysis.
        # - prom_ast: Average astigmatism aberration across all samples.
        
        return self.result_ast, self.prom_ast
       
