# -*- coding: utf-8 -*-
"""
Created on Mon Oct 13 17:05:24 2025

@author: MORGANRHAINAJERAROA
"""

import numpy as np
from numpy.polynomial.legendre import leggauss
import matplotlib.pyplot as plt
from ..equations.tracing import ProcessPattern2Field


################################################################################

class Gaussian_Quadrature:
    """
    ======================================
      Class: Gaussian_Quadrature
    ======================================
    
    This class implements Gaussian quadrature ray sampling over a circular 
    aperture to evaluate the optical performance of a system across multiple 
    wavelengths and field positions. The sampled rays are used to calculate 
    physically interpretable metrics such as RMS radius and image quality.
    
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
        - fun_info (list): Contains the optical system, raykeeper, and pupil 
        configuration.
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
        Generate nodes and weights for Gaussian quadrature adapted for a 
        circular aperture, corrected for area.

        Parameters:
        n (int): Number of radial nodes (and weights).

        Returns:
        tuple: Two numpy arrays containing radial nodes and properly 
        adjusted weights respectively.
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
        Computes ray coordinates for Gaussian Quadrature sampling across a 
        circular pupil.
    
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