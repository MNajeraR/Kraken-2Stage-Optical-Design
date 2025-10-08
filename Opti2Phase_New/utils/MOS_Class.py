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

# Now you can import safely
from MOS_equation import (
    ProcessPattern2Field,
    BestRMS,
    generate_radios,
    generate_angles,
    Sphere_subtract,
    calculate_average,
    generate_radiosc,
    generate_anglesc,
    Coma_Substraction,
    generate_radiosa,
    generate_anglesa,
    Astigmatism_Substraction
)
import numpy as np
from scipy.special import erfinv
from numpy.polynomial.legendre import leggauss
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from mpl_toolkits.axes_grid1.inset_locator import inset_axes




class Gaussian_Quadrature:
    """
    ======================================
      Class: Gaussian_Quadrature
    ======================================
    
    This class implements Gaussian quadrature ray sampling over a circular aperture 
    to evaluate the optical performance of a system across multiple wavelengths and 
    field positions. The sampled rays are used to calculate physically interpretable 
    metrics such as RMS radius and image quality.
    
    Attributes:
    -----------
    - system: Optical system to be evaluated.
    - raykeeper: Ray tracing manager instance.
    - P: Pupil configuration object.
    - w (float): Design wavelength.
    - Prx_Cal_EFFL: Effective focal length estimated from paraxial analysis.
    - EFFL_GQ: Effective focal length after Gaussian quadrature evaluation.
    
    ======================================
    """
    def __init__(self, fun_info, W):
        """
        Initializes the Gaussian_Quadrature class with the system, ray container, and pupil.
    
        Parameters:
        - fun_info (list): Contains the optical system, raykeeper, and pupil configuration.
        - W (float): Working wavelength for Gaussian quadrature sampling.
        """
        # Store the input system configuration
        self.fun_properties = fun_info 
    
        # Assign the working wavelength
        self.w = W
    
        # Extract the optical system from the input tuple
        self.system = self.fun_properties[0]
    
        # Extract the ray container object
        self.raykeeper = self.fun_properties[1]
    
        # Extract the pupil configuration object
        self.P = self.fun_properties[2]
        
    def circular_aperture_gaussian_quadrature_corrected(self, n):
        """
        Generate nodes and weights for Gaussian quadrature adapted for a circular aperture, corrected for area.

        Parameters:
        n (int): Number of radial nodes (and weights).

        Returns:
        tuple: Two numpy arrays containing radial nodes and properly adjusted weights respectively.
        """
        # Get the nodes and weights for Gaussian quadrature in the interval [-1, 1]
        self.nodes, self.weights = leggauss(n)

        # Transform the nodes to be in the interval [0, 1]
        self.radial_nodes = (0.5 * (self.nodes + 1))**0.5

         # Transform nodes from [-1, 1] to [0, 1] and apply square root for radial mapping
        self.radial_weights = self.weights * self.radial_nodes
        
        # Return the radial nodes and the normalized weights scaled by pi/2
        
        return self.radial_nodes, self.weights * np.pi / 2
    
    def Coordinates_GQ(self, n_nodes, n_arms, fieldx, fieldy, resp):
       
        """
        Computes ray coordinates for Gaussian Quadrature sampling across a circular pupil.
    
        Parameters:
        - n_nodes (int): Number of radial nodes for Gaussian sampling.
        - n_arms (int): Number of angular arms (azimuthal rays per node).
        - fieldx (float): Field position in X direction.
        - fieldy (float): Field position in Y direction.
        - resp (int): If set to 1, plots the pupil sampling.
        
        Returns:
        - tuple: (x, y, z, l, m, n) coordinates of traced rays at image plane.
        """
        # Generate the Gaussian quadrature radial nodes and weights
        self.radial_nodes, self.radial_weights = self.circular_aperture_gaussian_quadrature_corrected(n_nodes)
        # Initialize angle distribution and cleaning flag
        self.theta_n = []
        self.clean = 1
        # Normalize the weights by the number of angular arms
        self.radial_weights = self.radial_weights/n_arms
        # Generate evenly distributed angles depending on arm count
        if n_arms == 12 or n_arms == 8:
            self.theta_n = [(180 / n_arms) * (2 * i + 1) for i in range(n_arms)]
            
        elif n_arms == 10 or n_arms == 6:
            self.theta_n = [(180 / n_arms) * (2 * i) for i in range(n_arms)]
        
        else:
            raise ValueError('The number of arms can only be 6, 8, 10, or 12')
        # Assign ray types for each radial node (same for all angles)
        self.RayType = ['rtheta'] * len(self.radial_nodes) 
        
        
        # Initialize arrays for pupil plane and image plane coordinates
        self.xpup_values, self.ypup_values, self.zpup_values = [], [], []
        self.lpup_values, self.mpup_values, self.npup_values = [], [], []
    
        self.x_values, self.y_values, self.z_values = [], [], []
        self.l_values, self.m_values, self.n_values = [], [], []
            
        
       
        for self.i_nrad in range(n_nodes):
            
            if self.clean == 1:
                
                self.raykeeper.clean()
                self.system = self.raykeeper.SYSTEM
            
            for self.i_nang in range(n_arms):
                # Define ray type, radial distance, and angle
                self.P.Ptype = self.RayType[self.i_nrad]
                self.P.rad = self.radial_nodes[self.i_nrad]
                self.P.theta = self.theta_n[self.i_nang]
                self.P.FieldX = fieldx
                self.P.FieldY = fieldy
                # Perform ray tracing
                ProcessPattern2Field(self.w, self.fun_properties)
                self.Xpup, self.Ypup, self.Zpup, self.Lpup, self.Mpup, self.Npup = self.raykeeper.pick(1)
                self.X, self.Y, self.Z, self.L, self.M, self.N = self.raykeeper.pick(-1)
              
            
            # Append ray data to corresponding lists
            self.xpup_values.append(self.Xpup)
            self.ypup_values.append(self.Ypup)
            self.zpup_values.append(self.Zpup)
            self.lpup_values.append(self.Lpup)
            self.mpup_values.append(self.Mpup)
            self.npup_values.append(self.Npup)
             
            self.x_values.append(self.X)
            self.y_values.append(self.Y)
            self.z_values.append(self.Z)
            self.l_values.append(self.L)
            self.m_values.append(self.M)
            self.n_values.append(self.N)
         
        
        # Convert pupil plane coordinate lists to NumPy arrays
        self.xpup_values = np.array(self.xpup_values)
        self.ypup_values = np.array(self.ypup_values)
        self.zpup_values = np.array(self.zpup_values)
        self.lpup_values = np.array(self.lpup_values)
        self.mpup_values = np.array(self.mpup_values)
        self.npup_values = np.array(self.npup_values)
        
        
        # Reshape to 1D arrays of length (n_nodes * n_arms)
        self.xpup_values = self.xpup_values.reshape(1, n_nodes*n_arms)[0]
        self.ypup_values = self.ypup_values.reshape(1, n_nodes*n_arms)[0]
        self.zpup_values = self.zpup_values.reshape(1, n_nodes*n_arms)[0]
        self.lpup_values = self.lpup_values.reshape(1, n_nodes*n_arms)[0]
        self.mpup_values = self.mpup_values.reshape(1, n_nodes*n_arms)[0]
        self.npup_values = self.npup_values.reshape(1, n_nodes*n_arms)[0]    
        
        
        # Repeat the process for the image plane coordinates
        self.x_values = np.array(self.x_values)
        self.y_values = np.array(self.y_values)
        self.z_values = np.array(self.z_values)
        self.l_values = np.array(self.l_values)
        self.m_values = np.array(self.m_values)
        self.n_values = np.array(self.n_values)
        
        self.x_values = self.x_values.reshape(1, n_nodes*n_arms)[0]
        self.y_values = self.y_values.reshape(1, n_nodes*n_arms)[0]
        self.z_values = self.z_values.reshape(1, n_nodes*n_arms)[0]
        self.l_values = self.l_values.reshape(1, n_nodes*n_arms)[0]
        self.m_values = self.m_values.reshape(1, n_nodes*n_arms)[0]
        self.n_values = self.n_values.reshape(1, n_nodes*n_arms)[0]
        
    
        # Calculate the normalized radial distance of each ray with respect to the entrance pupil radius
        self.r_values = np.sqrt(self.xpup_values**2+self.ypup_values**2)/self.P.RadPupInp
        
        # Create a dictionary to store boolean masks per radial node
        self.masks = {}
        # Initialize lists to hold separated coordinates per radial segment (node)
        self.coordinatespupil_x = []
        self.coordinatespupil_y = []
        self.coordinatespupil_z = []
        self.coordinatespupil_l = []
        self.coordinatespupil_m = []
        self.coordinatespupil_n = []
        
        self.coordinates_x = []
        self.coordinates_y = []
        self.coordinates_z = []
        self.coordinates_l = []
        self.coordinates_m = []
        self.coordinates_n = []
        
        
        for i in range(n_nodes):
            # Create a dynamic mask name for storage
            self.mask_name = f"mask_{i}"
            self.coordinate_name = f"radius_{i}"
            # Define mask for rays belonging to the current radial node
            if i == 0:
                # First ring includes all rays with r < first node radius
                self.masks[self.mask_name] = self.r_values < self.radial_nodes[i] 
                
                
            else:
                # Other rings include rays within radial bounds of the annular segment
                self.masks[self.mask_name] =  (self.r_values >= self.radial_nodes[i-1]) & (self.r_values < self.radial_nodes[i])
                
            
            # Store entrance pupil coordinates corresponding to current radial node
            self.coordinatespupil_x.append(self.xpup_values[self.masks[self.mask_name]])
            self.coordinatespupil_y.append( self.ypup_values[self.masks[self.mask_name]])
            self.coordinatespupil_z.append(self.zpup_values[self.masks[self.mask_name]])
            self.coordinatespupil_l.append(self.lpup_values[self.masks[self.mask_name]])
            self.coordinatespupil_m.append(self.mpup_values[self.masks[self.mask_name]])
            self.coordinatespupil_n.append(self.npup_values[self.masks[self.mask_name]])
            # Store final image plane coordinates corresponding to current radial node
            self.coordinates_x.append(self.x_values[self.masks[self.mask_name]])
            self.coordinates_y.append(self.y_values[self.masks[self.mask_name]])
            self.coordinates_z.append(self.z_values[self.masks[self.mask_name]])
            self.coordinates_l.append(self.l_values[self.masks[self.mask_name]])
            self.coordinates_m.append(self.m_values[self.masks[self.mask_name]])
            self.coordinates_n.append(self.n_values[self.masks[self.mask_name]])
    
        # Convert entrance pupil coordinates to NumPy arrays
        self.coordinatespupil_x = np.array(self.coordinatespupil_x)
        self.coordinatespupil_y = np.array(self.coordinatespupil_y)
        self.coordinatespupil_z = np.array(self.coordinatespupil_z)
        self.coordinatespupil_l = np.array(self.coordinatespupil_l)
        self.coordinatespupil_m = np.array(self.coordinatespupil_m)
        self.coordinatespupil_n = np.array(self.coordinatespupil_n)
        # Convert image plane coordinates to NumPy arrays
        self.coordinates_x = np.array(self.coordinates_x)
        self.coordinates_y = np.array(self.coordinates_y)
        self.coordinates_z = np.array(self.coordinates_z)
        self.coordinates_l = np.array(self.coordinates_l)
        self.coordinates_m = np.array(self.coordinates_m)
        self.coordinates_n = np.array(self.coordinates_n)
        # Package entrance pupil direction cosines and coordinates
        self.setsep_pupilcoscoor = [self.coordinatespupil_x, self.coordinatespupil_y, self.coordinatespupil_z,
                               self.coordinatespupil_l, self.coordinatespupil_m, self.coordinatespupil_n]
        # Package final image plane direction cosines and coordinates
        self.setsep_coscoor = [self.coordinates_x, self.coordinates_y, self.coordinates_z,
                               self.coordinates_l, self.coordinates_m, self.coordinates_n]
        # Retrieve paraxial EFFL from the system at current wavelength
        self.Prx_Cal_EFFL= self.system.Parax(self.w)[7]
        # Retrieve the exact system EFFL from ray-based calculation
        self.EFFL_GQ = self.system.EFFL
        # If response flag is set to 1, generate a plot of the sampling points on the entrance pupil
        if resp == 1:
            
           # Create a new figure
            fig, ax = plt.subplots(figsize=(6, 6))
            
            # Plot the points
            ax.scatter(self.xpup_values, self.ypup_values, color='red')
            
            # Plot the unit circle for reference
            circle = plt.Circle((0, 0), self.P.RadPupInp, color='black', fill=False, linestyle='--')
            ax.add_patch(circle)
            
            # Set equal scaling
            ax.set_aspect('equal', adjustable='box')
            
            # Add labels and title
            ax.set_xlabel('X-axis')
            ax.set_ylabel('Y-axis')
            ax.set_title('Points on the Entrance Pupil')
            ax.legend()
            ax.grid(True)
            
            # Show the plot
            plt.show()
            
        else:
            pass
            
        
        # Return the final spatial and directional coordinates of all sampled rays
        return self.x_values, self.y_values, self.z_values, self.l_values, self.m_values, self.n_values


#########################################################################################################

class Function2Optimize:
    """
    ======================================
      Class: Function2Optimize
    ======================================

    This class defines the optimization process for an optical system across 
    multiple wavelengths and field positions using Gaussian Quadrature ray tracing.

    Attributes:
    - fun_properties (list): List containing the optical system, raykeeper, and pupil instance.
    - system (object): The optical system instance.
    - raykeeper (object): Instance to manage ray data.
    - P (object): Pupil object instance for ray tracing.
    - nodes (int): Number of nodes for Gaussian Quadrature.
    - arms (int): Number of arms for the circular sampling.
    - Fx (list): List of field positions in X.
    - Fy (float): Field position in Y (fixed at 0.0).
    - w1, w2, w3 (float): Wavelengths for evaluation (0.35, design wavelength W, 0.55).
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
        self.Fx = [0.0, 0.07]
        self.Fy = 0.0
        
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
        - list: Contains differences in EFFL, wavelength spread, and RMS results 
          for both fields.
        """
        # Update system radii of curvature with the new parameters
        self.system.SDT[3].Rc = V[0]
        self.system.SDT[4].Rc = V[1]
        self.system.SDT[5].Rc = V[2]
        self.system.SDT[6].Rc = V[3]

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
        
        # Get the effective focal lengths for each wavelength
        EFFL1 = self.gqa.EFFL_GQ
        EFFL0 = self.gqb.EFFL_GQ
        EFFL2 = self.gqc.EFFL_GQ
        
        # Calculate spread in focal lengths
        ra_f1, rb_f1 = EFFL0 - EFFL1, EFFL0 - EFFL2
        r_f1 = np.sqrt(ra_f1**2 + rb_f1**2)
        
        # Compute the deviation from the target effective focal length
        D_EFFL = np.abs(EFFL0 - self.effl)

        # Perform RMS calculations for each wavelength (vectorized approach)
        Ba_f1 = np.array(BestRMS(self.gqa.setsep_coscoor, self.system))
        Bb_f1 = np.array(BestRMS(self.gqb.setsep_coscoor, self.system))
        Bc_f1 = np.array(BestRMS(self.gqc.setsep_coscoor, self.system))
        
        # Vectorized computation for RMS summation
        weights_a = np.array(self.gqa.radial_weights)
        weights_b = np.array(self.gqb.radial_weights)
        weights_c = np.array(self.gqc.radial_weights)
        
        # Vectorized summation of RMS weighted by Gaussian quadrature weights
        self.result_1 = np.sqrt(
            (np.sum(weights_a * Ba_f1**2) + 
             np.sum(weights_b * Bb_f1**2) + 
             np.sum(weights_c * Bc_f1**2)) / len(weights_a)
        )

        ########################################################################
        #                             Second Field                             #
        ########################################################################
        
        # Perform Gaussian Quadrature ray tracing for three wavelengths in the second field
        self.gqa.Coordinates_GQ(self.nodes, self.arms, self.Fx[1], 0.0, 0)
        self.gqb.Coordinates_GQ(self.nodes, self.arms, self.Fx[1], 0.0, 0)
        self.gqc.Coordinates_GQ(self.nodes, self.arms, self.Fx[1], 0.0, 0)

        # Perform RMS calculations for each wavelength (vectorized approach)
        Ba_f2 = np.array(BestRMS(self.gqa.setsep_coscoor, self.system))
        Bb_f2 = np.array(BestRMS(self.gqb.setsep_coscoor, self.system))
        Bc_f2 = np.array(BestRMS(self.gqc.setsep_coscoor, self.system))
        
        # Vectorized summation of RMS weighted by Gaussian quadrature weights
        self.result_2 = np.sqrt(
            (np.sum(weights_a * Ba_f2**2) + 
             np.sum(weights_b * Bb_f2**2) + 
             np.sum(weights_c * Bc_f2**2)) / len(weights_a)
        )

        # Restore the original state of the optical system
        self.system.RestoreData()
        
        # Return the evaluation metrics
        return [D_EFFL, r_f1, self.result_1, self.result_2]




#########################################################################################################

"""
======================================
  Class: Random_RO
======================================

This class implements a randomized optimization strategy using Gaussian 
distribution for parameter sampling. It allows multiple iterations to 
refine the solution, progressively moving closer to the optimal result.

Parameters:
- num_range (int): The number of random values to generate for each center.
- centers (list): A list of initial center points for the Gaussian distribution.

Methods:
1. Gaussian_distribution:
    Generates a set of random values around specified centers using 
    a Gaussian distribution.

2. Gaussian_optimization:
    Iteratively optimizes the solution by evaluating a specified function 
    and adjusting the distribution of samples based on the best result.

3. _process_iteration:
    Processes a single iteration of the optimization, evaluating the function 
    at each sampled point and selecting the best solution.

4. _process_new_iteration:
    Processes subsequent iterations, refining the sample space around 
    the best solution found so far.

======================================
"""

# ======================================
#    Class Initialization
# ======================================

class Random_RO:
    def __init__(self, num_range, centers):
        """
        Initializes the Random_RO class.

        Parameters:
        - num_range (int): The number of random values to generate per center.
        - centers (list): A list of center values for the Gaussian distribution.
        """
        self.num_range = num_range   # Number of random values to generate
        self.centers = centers       # List of initial center points

    # ======================================
    #    Gaussian Distribution Generation
    # ======================================

    def Gaussian_distribution(self, centers_distribution):
        """
        Generates a set of Gaussian-distributed random values centered 
        around specified points.

        Parameters:
        - centers_distribution (list): A list of desired centers for the distribution.

        Returns:
        - dim_coor (list): A list of arrays containing the sampled values 
                           for each center, adjusted by the distribution.
        """
        
        # Store the target centers
        self.desired_centers = centers_distribution
        
        # Initialize index and list to store coordinates
        self.i = 0
        self.dim_coor = []

        # Loop through each center and generate samples
        while self.i < len(self.desired_centers):
            
            # Generate uniformly distributed values between -1 and 1
            self.uniform_values = np.random.uniform(-1, 1, self.num_range).astype(np.float64)
            
            # Apply the inverse error function to achieve a Gaussian distribution
            self.x = erfinv(self.uniform_values)
            
            # Shift the generated values by the desired center
            self.dim_coor.append(self.x + self.desired_centers[self.i])
            
            # Move to the next center
            self.i += 1
        
        # Return the list of coordinates
        return self.dim_coor   

    # ======================================
    #    Gaussian Optimization Loop
    # ======================================

    def Gaussian_optimization(self, iterations, function):
        """
        Performs the optimization process through multiple iterations, 
        refining the Gaussian sampling distribution around the best solution.

        Parameters:
        - iterations (int): The number of optimization iterations.
        - function (callable): The objective function to minimize.

        Returns:
        - best_solution (list): The best solution found during the optimization process.
        """
        
        # Validate the number of iterations
        self.iterations = iterations
        
        if self.iterations < 1:
            print('Number of iterations must be greater than 1')
            return None

        # Generate the initial set of Gaussian-distributed values
        self.set_values = self.Gaussian_distribution(self.centers)
       
        # Initialize lists to store the evaluation results
        self.vector_2ev = []
        self.results = []
        self.num_vectors = len(self.set_values)
       
        # Perform the first iteration
        self._process_iteration(function)
        
        # If there is only one iteration, return the best solution
        if self.iterations == 1:
            return self.best_solution
        
        # Loop through the remaining iterations, refining around the best solution
        for _ in range(self.iterations - 1):
            self.set_new_values = self.Gaussian_distribution(self.best_solution)
            self._process_new_iteration(function)
        
        return self.best_solution

    # ======================================
    #    Process First Iteration
    # ======================================

    def _process_iteration(self, function):
        """
        Processes the first iteration of the optimization by evaluating 
        the function at each generated sample.

        Parameters:
        - function (callable): The objective function to minimize.

        Steps:
        1. Combine the sampled values into vectors.
        2. Evaluate the function for each vector.
        3. Find the minimum value and its corresponding vector.
        4. Store the best solution found.
        """
        
        # Collect the generated values into evaluation vectors
        for i in range(self.num_range):
            self.value_2keep = [self.set_values[j][i] for j in range(self.num_vectors)]
            self.vector_2ev.append(self.value_2keep)
        
        # Evaluate the function for each vector
        self.results = [function(vec) for vec in self.vector_2ev]
        
        # Identify the minimum result and its index
        self.min_index = np.argmin(self.results)
        self.min_result = self.results[self.min_index]
        self.best_solution = self.vector_2ev[self.min_index]

    # ======================================
    #    Process Subsequent Iterations
    # ======================================

    def _process_new_iteration(self, function):
        """
        Processes additional iterations, refining the sampling around the 
        best solution found in the previous iteration.

        Parameters:
        - function (callable): The objective function to minimize.

        Steps:
        1. Generate new samples around the best solution.
        2. Evaluate the function for each new vector.
        3. Update the best solution if a better one is found.
        """
        
        # Initialize lists to store new evaluation vectors and results
        self.new_vector_2ev = []
        self.new_results = []
       
        # Collect the generated values into evaluation vectors
        for i in range(self.num_range):
            self.new_value_2keep = [self.set_new_values[j][i] for j in range(self.num_vectors)]
            self.new_vector_2ev.append(self.new_value_2keep)
        
        # Evaluate the function for each new vector
        self.new_results = [function(vec) for vec in self.new_vector_2ev]
     
        # Identify the minimum result and its index
        self.new_min_index = np.argmin(self.new_results)
        self.new_min_result = self.new_results[self.new_min_index]
        self.best_solution = self.new_vector_2ev[self.new_min_index]
        
#########################################################################################################
        

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
    
    def __init__(self, SeidelTool, info_lens):
        """
        Initializes the ThirdOrder_Cal class with necessary parameters.

        Parameters:
        - SeidelTool: Instance to calculate Seidel aberrations.
        - info_lens (tuple): Contains lens thickness, bending parameters, 
                             and refractive indices.
        """
        self.SeidelInfo = SeidelTool
        self.d_l1, self.d_l2 = info_lens[0]     # Lens thicknesses
        self.b3, self.b4 = info_lens[1]         # Bending parameters
        self.n_L1, self.n_L2 = info_lens[2]     # Refractive indices
        
        
        # Wavelength and system distances
        self.W = 0.43032015 
        self.d_1  = 4052.571043
        self.d_2 = 622
        self.d_3 = 12.1858 + 10 + 7
        self.EFFL_Tr =  9127.198583362275         # Target Effective Focal Length
        
        
        # --- En __init__ ---
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
      Method: Lens_Maker
    ======================================
    """
    def Lens_Maker(self, R1a, R1b, R2a, R2b):
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
        
        
        
        power_theoric_1 = self.b3 
        
        power_numeric_L1a = (self.n_L1 - 1)/R1a
        power_numeric_L1b = (1 - self.n_L1)/R1b
        
        cons_1 = self.d_l1 / power_theoric_1 
        cons_2 = cons_1 * power_numeric_L1a * power_numeric_L1b
        
        power_numeric_L1 = (power_numeric_L1a + power_numeric_L1b - cons_2)
        
        theoric_F_L1 = 1/self.b3
        numeric_F_L1 = 1/ power_numeric_L1
        # print(R1a,R1b,R2a,R2b)
        
        # print(power_theoric_1, power_numeric_L1)
        # print(theoric_F_L1, numeric_F_L1)
        
        self.Re_1 = np.abs(-power_theoric_1 + power_numeric_L1)
        
        # print(1/self.Re_1)
        # print(self.Re_1)
        
        power_theoric_2 = self.b4 
        power_numeric_L2a = (self.n_L2 - 1)/R2a
        power_numeric_L2b = (1 - self.n_L2)/R2b
        
        cons_3 = self.d_l2 / self.n_L2 
        cons_4 = cons_3*power_numeric_L2a*power_numeric_L2b
        
        power_numeric_L2 = (power_numeric_L2a + power_numeric_L2b - cons_4)
        
        theoric_F_L2 = 1/power_theoric_2
        numeric_F_L2 = 1/power_numeric_L2 
        
        
        # print(power_theoric_2,  power_numeric_L2)
        # print(theoric_F_L2, numeric_F_L2)
        
        self.Re_2 =  np.abs(-power_theoric_2 + power_numeric_L2)
        
        # print(1/self.Re_2)
        # print(self.Re_2)
        
        return self.Re_1, self.Re_2
    
    """
    ======================================
      Method: Prin_Plane
    ======================================
    """
    def Prin_Plane(self, R1a, R1b, R2a, R2b):
        """
        Calculates the principal planes for two lenses.

        Parameters:
        - R1a, R1b (float): Radii of curvature for the first lens.
        - R2a, R2b (float): Radii of curvature for the second lens.

        Returns:
        - (float, float, float, float): Positions of the principal planes.
        """
        
        # Principal plane calculations
        self.h_1a = ((self.n_L1 - 1) * self.d_l1) / (R1b * self.b3 * self.n_L1)
        self.h_2a = ((self.n_L1 - 1) * self.d_l1) / (R1a * self.b3 * self.n_L1)
        
        self.h_1b = ((self.n_L2 - 1) * self.d_l2) / (R2b * self.b4 * self.n_L2)
        self.h_2b = ((self.n_L2 - 1) * self.d_l2) / (R2a * self.b4 * self.n_L2)
        
        return self.h_1a, self.h_2a, self.h_1b, self.h_2b
    
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
        
        Units = 1000.0
        w_1 = (1.0)**2
        w_2 = (1.0)**2
        w_3 = (0.3043)**2
        w_4 = (0.3043)**2
        w_5 = (0.3043)**2
        w_6 = (1.)**2
        w_7 = (1.)**2
        w_8 = (1.)**2
        
        
        
        # Unpack the variables
        self.R1, self.R2, self.R3, self.R4, self.d4 = variables
        
        # Access the optical system
        self.System = self.SeidelInfo.SYSTEM
        
        # Set radii of curvature in the optical system
        self.System.SDT[3].Rc = self.R1
        self.System.SDT[4].Rc = self.R2
        self.System.SDT[5].Rc = self.R3
        self.System.SDT[6].Rc = self.R4
        
        # Calculate the principal planes
        self.H1_a, self.H2_a, self.H1_b, self.H2_b = self.Prin_Plane(self.R1, self.R2, self.R3, self.R4)
        
        # Set the thickness of each lens in the optical train
        self.System.SDT[2].Thickness = self.d_1 + self.d_2 - self.H1_a
        self.System.SDT[4].Thickness = self.d_3 + self.H2_a - self.H1_b
        self.System.SDT[6].Thickness = self.d4 + self.H2_b
        
        # Update the system's internal parameters
        self.System.SetData()
        
        # Calculate the effective lens radii
        self.Sis1, self.Sis2 = self.Lens_Maker(self.R1, self.R2, self.R3, self.R4)
        
        self.B1 = (w_1*self.Sis1*(self.W/Units))
        self.B2 = (w_2*self.Sis2*(self.W/Units))
        
        print(self.B1, self.B2)
        
        self.PL_fun = self.B1**2 + self.B2**2
        
        # Calculate Seidel aberrations
        self.SeidelInfo.calculate()
        print(self.SeidelInfo.Wf, self.SeidelInfo.Wc)
        
        # Compute each aberration term
        self.Sph = self.SeidelInfo.SAC_TOTAL[0]*(Units/self.W)
        self.Coma = self.SeidelInfo.SAC_TOTAL[1]*(Units/self.W)
        self.Ast = self.SeidelInfo.SAC_TOTAL[2]*(Units/self.W)
        self.CLon = np.sum(self.SeidelInfo.CL)*(Units/self.W)
        
        # Merit function calculation
        self.Aberration_fun = w_3*(self.Sph)**2 + w_4*(self.Coma)**2 + w_5*(self.Ast)**2 + w_6*(self.CLon)**2
        print(w_3*(self.Sph)**2)
        print(w_4*(self.Coma)**2)
        print(w_5*(self.Ast)**2)
        print(w_6*(self.CLon)**2)
        print(self.Aberration_fun)


        # Compute paraxial data
        self.Prx = np.array(self.System.Parax(self.W)[0])
        
        self.d = w_7*(self.Prx[1][1]*(Units/self.W))
        print((self.Prx[1][1]*(Units/self.W)))

        
        
        # EFFL deviation from the target
        self.D_EFFL = w_8*((self.System.EFFL - self.EFFL_Tr)*(Units/self.W))
        print(self.D_EFFL)
        
        
        self.Merit_fun = (self.PL_fun + self.Aberration_fun + self.D_EFFL**2 + self.d**2)
        
        # --- Guardar historial ---
    
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
        
        print('')


        # Restore the original system state
        self.System.RestoreData()
        
        
        return  [self.B1, self.B2, self.Aberration_fun, self.D_EFFL, self.d]
    
    """
    ======================================
      Method: SeedPar_OLD
    ======================================
    """
    def SeedPar_OLD(self, variables):
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
        
        # Unpack the variables
        self.R1, self.R2, self.R3, self.R4, self.d4 = variables
        
        # Access the optical system
        self.System = self.SeidelInfo.SYSTEM
        
        # Set radii of curvature in the optical system
        self.System.SDT[3].Rc = self.R1
        self.System.SDT[4].Rc = self.R2
        self.System.SDT[5].Rc = self.R3
        self.System.SDT[6].Rc = self.R4
        
        # Calculate the principal planes
        self.H1_a, self.H2_a, self.H1_b, self.H2_b = self.Prin_Plane(self.R1, self.R2, self.R3, self.R4)
        
        # Set the thickness of each lens in the optical train
        self.System.SDT[2].Thickness = self.d_1 + self.d_2 - self.H1_a
        self.System.SDT[4].Thickness = self.d_3 + self.H2_a - self.H1_b
        self.System.SDT[6].Thickness = self.d4 + self.H2_b
        
        # Update the system's internal parameters
        self.System.SetData()
        
        # Calculate the effective lens radii
        self.Sis1, self.Sis2 = self.Lens_Maker(self.R1, self.R2, self.R3, self.R4)
        
        print(self.Sis1, self.Sis2)
        
        # Compute paraxial data
        self.Prx = np.array(self.System.Parax(self.W)[0])
        self.d = self.Prx[1][1]
        print(self.d)

        # Calculate Seidel aberrations
        self.SeidelInfo.calculate()
        
        # Compute each aberration term
        self.Sph = np.abs(self.SeidelInfo.SAC_TOTAL[0]) * 1000 * self.W
        self.Coma = np.abs(self.SeidelInfo.SAC_TOTAL[1]) * 1000 * self.W 
        self.Ast = np.abs(self.SeidelInfo.SAC_TOTAL[2]) * 1000 * self.W 
        self.CLon = np.abs(np.sum(self.SeidelInfo.CL)) * 1000 * self.W
        
        # Merit function calculation
        self.Merit_fun = 0.5 * (self.Sph)**2 + 0.5 * (self.Coma)**2 + 0.5 * (self.Ast)**2 + (self.CLon)**2
        
        print(0.5 * (self.Sph)**2)
        print(0.5 * (self.Coma)**2)
        print(0.5 * (self.Ast)**2)
        print((self.CLon)**2)
        print(self.Merit_fun)

        
        # EFFL deviation from the target
        self.D_EFFL = np.abs(self.System.EFFL - self.EFFL_Tr)
        print(self.D_EFFL)
        
        # Restore the original system state
        
        # --- Guardar historial ---
    
        self._iter_counter += 1
        self.history['iter'].append(self._iter_counter)
        self.history['B_1'].append(float((self.Sis1)))
        self.history['B_2'].append(float((self.Sis2)))
        self.history['F_1'].append(float((1/self.Sis1)))
        self.history['F_2'].append(float((1/self.Sis2)))
        self.history['D_EFFL'].append(float(self.D_EFFL))
        self.history['d'].append(float(self.d))
        self.history['Merit_fun'].append(float(self.Merit_fun))
        
        self.System.RestoreData()
        
        return [self.Sis1, self.Sis2, self.Merit_fun, self.D_EFFL, self.d]
    
    
#########################################################################################################

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
        self.d_2 = 622                       # Distance to the focal reducer
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
    
    
    
#########################################################################################################

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
        self.dw_1 = 0.5876
        self.dw_2 = 0.6563
        
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
        self.CampoX, self.CampoY = Campo
        
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
        self.CampoX, self.CampoY = Campo
        
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
        self.CampoX, self.CampoY = 0.0, Campo
        
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
        self.CampoX, self.CampoY = 0.0, Campo
        
        # Scaling factor for sagittal and tangential rays
        c = 100  
        
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


        

#########################################################################################################

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
        
            