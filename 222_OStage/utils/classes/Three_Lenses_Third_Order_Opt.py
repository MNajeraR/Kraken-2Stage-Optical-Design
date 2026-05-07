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

class ThirdLens_3O_20:
    
    def __init__(self, System):
        """
        Initializes the ThirdOrder_Cal class with necessary parameters.

        Parameters:
        - SeidelTool: Instance to calculate Seidel aberrations.
        - info_lens (tuple): Contains lens thickness, bending parameters, 
                             and refractive indices.
        """
        
        self.System = System
        
        # Wavelength and system distances
        self.W   = 0.43032015
        self.W_1 = 0.35
        self.W_2 = 0.5499996 
        
        self.d_2 = self.System.SDT[2].Thickness
        self.d_3 = self.System.SDT[4].Thickness
        self.d_4 = self.System.SDT[6].Thickness
       

        self.EFFL_Tr =  9127.198583362275         # Target Effective Focal Length        
        
        Initial_parax = self.System.Parax(self.W)
        Initial_n = Initial_parax[11]
        
        self.d_L1 = self.System.SDT[3].Thickness  # Lens thicknesses
        self.d_L2 = self.System.SDT[5].Thickness  # Lens thicknesses
        self.d_L3 = self.System.SDT[7].Thickness  # Lens thicknesses
        
        
        
        self.n_L1 = Initial_n[3]                  # Refractive indices
        self.n_L2 = Initial_n[5]                  # Refractive indices
        self.n_L3 = Initial_n[7]                  # Refractive indices
        # d
       # --- History---
        self.history = {
                'iter'   : [],
                'Sph_F1' : [],
                'Coma_F1': [],
                'Ast_F1' : [],
                'Dis_F1' : [],
                'FCur_F1': [],
                'CLon_F1' : [],
                'D_EFFL' : [],
                'Coma_F2': [],
                'Ast_F2' : [],
                'Merit_fun' : []
                }
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
        Units = 1000.
        
        # Define the parameters for the pupil calculation:
        sup = 1              # Number of the surface representing the opening of the system
        AperType = "EPD"     # AperType sets the aperture ("STOP") or entrance pupil diameter ("EPD").
        AperVal = 2152.      # Diameter of the entrance pupil
        
        
        #Define constrain 
        w_1 =  (1.)**2
        w_2 =  (1.)**2
        w_3 =  (1.)**2
        w_4 =  (1.)**2
        w_5 =  (1.)**2
        w_6 =  (1.)**2
        w_7 =  (1.)**2
        w_8 =  (1.)**2
        w_9 =  (1.)**2
        w_10 = (1.)**2
          
        # Unpack the variables

        self.R1, self.R2, self.R3, self.R4, self.R5, self.R6, self.d5 = variables
        # print(self.R1, self.R2, self.R3, self.R4, self.R5, self.R6, self.d5)
        ###########################################################################################
        
        # Set radii of curvature in the optical system
        
        self.System.SDT[3].Rc = self.R1
        self.System.SDT[4].Rc = self.R2
        self.System.SDT[5].Rc = self.R3
        self.System.SDT[6].Rc = self.R4
        self.System.SDT[7].Rc = self.R5
        self.System.SDT[8].Rc = self.R6
        
        self.System.SDT[8].Thickness = self.d5  
        
        # Update the system's internal parameters
        self.System.SetData()
        
        ###########################################################################################
       
        # Initialize the Pupil calculation for the system
        self.Pup = Kos.PupilCalc(self.System, sup, self.W, AperType, AperVal)

        # Configure the sampling for the pupil and its field representation:
        self.Pup.Samp = 7         # Integer number for pupil ray sampling
        self.Pup.FieldType = "angle" # Field type, this in terms of object height and distance from the plane.
        self.Pup.FieldX = 0.07062161665871489 # Field value in degrees on the X-axis
        
        
        # Calculate Seidel aberrations
        self.AB = Kos.Seidel(self.Pup)
        self.AB.Wf = self.W_1                    # Set the first adjacent design wavelength (shorter) 
        self.AB.Wd = self.W                       # Set the center design wavelength
        self.AB.Wc = self.W_2               # Set the second adjacent design wavelength (longer)
        # self.AB.calculate()
        
        # Compute each aberration term
        self.Sph_F1  = w_1*(self.AB.SAC_TOTAL[0]*(Units/self.W))
        self.Coma_F1 = w_2*(self.AB.SAC_TOTAL[1]*(Units/self.W))
        self.Ast_F1  = w_3*(self.AB.SAC_TOTAL[2]*(Units/self.W))
        self.FCur_F1 = w_4*(self.AB.SAC_TOTAL[3]*(Units/self.W))
        self.Dis_F1  = w_5*(self.AB.SAC_TOTAL[4]*(Units/self.W))
        self.CLon_F1 = w_6*(np.sum(self.AB.CL)*(Units/self.W))
        
        
        ###########################################################################################
        
        # Initialize the Pupil calculation for the system
        self.Pup = Kos.PupilCalc(self.System, sup, self.W, AperType, AperVal)

        # Configure the sampling for the pupil and its field representation:
        self.Pup.Samp = 7                 # Integer number for pupil ray sampling
        self.Pup.FieldType = "angle"      # Field type, this in terms of object height and distance from the plane.
        self.Pup.FieldX =  0.07062161665871489     # Field value in degrees on the X-axis
        self.Pup.FieldY =  -0.07062161665871489    # Field value in degrees on the X-axis
        
        # Calculate Seidel aberrations
        self.AB = Kos.Seidel(self.Pup)
        self.AB.Wf = self.W_1                    # Set the first adjacent design wavelength (shorter) 
        self.AB.Wd = self.W                       # Set the center design wavelength
        self.AB.Wc = self.W_2               # Set the second adjacent design wavelength (longer)
        # self.AB.calculate()
        
        self.Sph_F2 =  w_7*(self.AB.SAC_TOTAL[0]*(Units/self.W))
        self.Coma_F2 =  w_7*(self.AB.SAC_TOTAL[1]*(Units/self.W))
        self.Ast_F2  =  w_8*(self.AB.SAC_TOTAL[2]*(Units/self.W))
        
        ###########################################################################################
        
        
        # Compute paraxial data
        self.Prx = np.array(self.System.Parax(self.W)[0])        
        self.d = w_9*(self.Prx[1][1]*(Units/self.W))
        
        
        ###########################################################################################
        
        # EFFL deviation from the target
        
        self.D_EFFL = w_10*((self.System.EFFL - self.EFFL_Tr))

        if np.abs(self.D_EFFL) > 300:
            
            self.Sph_F1  = self.Sph_F1  * 100000000.
            self.Coma_F1 = self.Coma_F1 * 100000000.
            self.Ast_F1  = self.Ast_F1  * 100000000.
            self.FCur_F1 = self.FCur_F1 * 100000000.
            self.Dis_F1  = self.Dis_F1  * 100000000.
            self.CLon_F1 = self.CLon_F1 * 100000000.
            self.Coma_F2 = self.Coma_F2 * 100000000.
            
        ###########################################################################################
        
        #Possible merit function
        self.Aberration_fun = sum(x**2 for x in [
                self.Sph_F1, self.Coma_F1, self.Ast_F1, self.Dis_F1,
                self.FCur_F1, self.CLon_F1, self.Coma_F2, self.Ast_F2
                ])
        self.Merit_fun = (self.Aberration_fun + self.D_EFFL**2)
        
        # print(self.Sph_F1, self.Coma_F1, self.Ast_F1, self.Dis_F1, self.FCur_F1)
        # print(self.CLon_F1, self.Coma_F2, self.Ast_F2, self.d, self.D_EFFL)
        ###########################################################################################
        
        # --- Safe History ---
        
        self._iter_counter += 1
        self.history['iter'].append(self._iter_counter)
        self.history['Sph_F1'].append(float(self.Sph_F1))
        self.history['Coma_F1'].append(float(self.Coma_F1))
        self.history['Ast_F1'].append(float(self.Ast_F1))
        self.history['Dis_F1'].append(float(self.Dis_F1))
        self.history['FCur_F1'].append(float(self.FCur_F1))
        self.history['CLon_F1'].append(float(self.CLon_F1))
        self.history['D_EFFL'].append(float(self.D_EFFL))
        self.history['Coma_F2'].append(float(self.Coma_F2))
        self.history['Ast_F2'].append(float(self.Ast_F2))
        self.history['Merit_fun'].append(float(self.Merit_fun))
        
        ###########################################################################################

        # Restore the original system state
        self.System.RestoreData()
        
        return  [self.D_EFFL, self.Sph_F1,  self.Coma_F1, self.Ast_F1,  self.Dis_F1, 
                  self.FCur_F1, self.CLon_F1, self.Coma_F2, self.Ast_F2, 
                  self.d]
        