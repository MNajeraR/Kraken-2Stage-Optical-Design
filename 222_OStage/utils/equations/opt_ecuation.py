# -*- coding: utf-8 -*-
"""
Created on Fri Oct 17 11:38:47 2025

@author: MORGANRHAINAJERAROA
"""

import numpy as np



def Lens_Maker(n_1, Ra, Rb, d_l):
    
    """
    Applies the lens maker's formula to compute the effective radii 
    of curvature for two lenses.

    Parameters:
    - R1a, R1b (float): Radii of curvature for the first lens.
    - R2a, R2b (float): Radii of curvature for the second lens.

    Returns:
    - (float, float): Effective radii for each lens.
    """
    
    # Apply lens maker's formula
    
    power_numeric_L1a = (n_1 - 1)/Ra
    power_numeric_L1b = (1 - n_1)/Rb
    
    cons_1 = d_l / n_1 
    cons_2 = cons_1 * power_numeric_L1a * power_numeric_L1b
    
    power_numeric_L1 = - (power_numeric_L1a + power_numeric_L1b - cons_2)

    
    return power_numeric_L1


def matrix_refraction(n1,n2,R):
    RR = np.matrix([[(n1 / n2), ((n1 - n2) / (n2 * R))], [0.0, 1]])
    return RR
 
def matrix_trasmition(dd):
    TT = np.matrix([[1.0, 0.0], [dd, 1.0]])
    return TT   

def Prin_Plane(n_1, Ra, Rb, d):
    
    M_1 = matrix_refraction(1.0, n_1, Ra)
    M_2 = matrix_trasmition(d)
    M_3 = matrix_refraction(n_1, 1., Rb)
    
    MS = M_3@M_2@M_1
    a = MS[0, 0]
    b = MS[0, 1]
    d = MS[1, 1]
    
    APP = (1-d)/-b
    PPP = (a-1)/-b
    
    return APP, PPP
    
def seidel_terms(AB, W):
    Sph  = AB.SAC_TOTAL[0]*(1000/W)
    Coma = AB.SAC_TOTAL[1]*(1000/W)
    Ast  = AB.SAC_TOTAL[2]*(1000/W)
    CLon = np.sum(AB.CL)*(1000/W)
    return Sph, Coma, Ast, CLon

def seidel_terms_20(AB, W):
    Sph  = AB.SAC_TOTAL[0]*(1000/W)
    Coma = AB.SAC_TOTAL[1]*(1000/W)
    Ast  = AB.SAC_TOTAL[2]*(1000/W)
    FCur = AB.SAC_TOTAL[3]*(1000/W)
    Dis  = AB.SAC_TOTAL[4]*(1000/W)
    CLon = np.sum(AB.CL)*(1000/W)
    return Sph, Coma, Ast, FCur, Dis, CLon


def airy_data(system, W, sup):
    
    #Parameters:
    # W: # Reference wavelength
    # Telescope_f85_FR.EFFL: Effective Focal Length
    # Ap_Diameter: Diameter of the pupil
    
    Ap_Diameter = system.SDT[1].Diameter
    AiryPrx = system.Parax(W)
    NA = (Ap_Diameter)/(2*AiryPrx[7])
    Rairy = 1.22*((W/1000)/(2*NA))
    num_segmentos = 100
    angulo = np.linspace(0, 2*np.pi, num_segmentos + 1)
    xairy = Rairy * np.cos(angulo)
    yairy = Rairy * np.sin(angulo)
    
    return Rairy, xairy, yairy


def configure_pupil_and_ab(system, Kos, sup, W_Set, AperType, AperVal, Field_ccd, samp=7):
    Pup = Kos.PupilCalc(system, sup, W_Set[1], AperType, AperVal)
    Pup.Samp = samp
    Pup.FieldType = "angle"
    Pup.FieldX = np.rad2deg(Field_ccd)
    AB = Kos.Seidel(Pup)
    AB.Wf = W_Set[0]
    AB.Wd = W_Set[1]
    AB.Wc = W_Set[2]
    return Pup, AB




 

