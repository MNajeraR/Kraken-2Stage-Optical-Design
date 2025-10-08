# -*- coding: utf-8 -*-
"""
Created on Tue May 16 11:25:13 2023

@author: MORGANRHAINAJERAROA
"""

import numpy as np
import random
import csv
import scipy
import matplotlib.pyplot as plt
import os

import pkg_resources
required = {'KrakenOS'}
installed = {pkg.key for pkg in pkg_resources.working_set}
missing = required - installed

if missing:
    print("No instalado")
    import sys
    sys.path.append("../..")


import KrakenOS as Kos
import matplotlib.patches as patches






#########################################################################################################

def configure_and_trace(Pup, fx, fy, pattern_type, wavelength, clean=1):
    """
    Configure pupil parameters, perform ray tracing, and return traced rays.
    
    Parameters:
    - Pup (object): Pupil object that handles the sampling and tracing of rays.
    - fx (float): X-coordinate of the field in degrees, representing the angular 
                  displacement of the field with respect to the optical axis.
    - fy (float): Y-coordinate of the field in degrees, representing the angular 
                  displacement in the vertical plane.
    - pattern_type (str): The sampling pattern for rays (e.g., "hexapolar").
    - wavelength (float): Wavelength of light for the ray tracing process.
    - clean (int, optional): Flag indicating whether the system should be reset 
                             before tracing (default is 1, which means "yes").
    
    Process:
    1. Configure the pupil sampling settings:
        - Set the number of sampling points (1 point in this case).
        - Assign the field coordinates (`fx`, `fy`) to the pupil.
        - Set the type of sampling pattern (e.g., "hexapolar").
    
    2. Perform the ray tracing by invoking `Pattern2Field()`, which projects 
       the sampled rays from the pupil onto the field plane according to the 
       specified parameters.
    
    3. Return the traced coordinates and direction cosines of the rays.
    
    Returns:
    - tuple: Contains the traced coordinates (`x`, `y`, `z`) and direction 
             cosines (`L`, `M`, `N`) of the rays.
    """
    
    # Configure the pupil object with the specified parameters
    Pup.Samp = 1                # Single sampling point for tracing
    Pup.FieldX = fx             # Assign X-coordinate of the field
    Pup.FieldY = fy             # Assign Y-coordinate of the field
    Pup.Ptype = pattern_type    # Set the pattern type (e.g., "hexapolar")
    
    # Perform the ray tracing and return the traced data
    
    return Pup.Pattern2Field()


# ======================================
#    Plotting Functions
# ======================================
   
"""
======================================
  Function: calculate_geometrical_center
======================================

This function calculates the geometrical center (centroid) of a set of 
coordinates in the X and Y planes. It finds the extreme values (max and min) 
for both X and Y coordinates and computes the average to determine the central 
point. This is particularly useful for centering spot diagrams or optical 
fields in lens design and analysis.

Parameters:
- X_all (list or np.array): List or array of x coordinates.
- Y_all (list or np.array): List or array of y coordinates.

Steps:
1. Compute the maximum and minimum values for both X and Y coordinates.
2. Store these values in lists for maximum and minimum.
3. Calculate the geometrical center by averaging the maximum and minimum 
   for each axis (X and Y).
4. Return the calculated center coordinates (h, k).

Returns:
- h (float): Geometrical center coordinate for the X-axis.
- k (float): Geometrical center coordinate for the Y-axis.

======================================
"""

def calculate_geometrical_center(X_all, Y_all):
    
    # -------------------------------------------------
    # Step 1: Compute the maximum and minimum values
    #         for the X and Y coordinates.
    # -------------------------------------------------
    
    x_setmax = [max(X_all)]  # Find the maximum X value
    x_setmin = [min(X_all)]  # Find the minimum X value
    y_setmax = [max(Y_all)]  # Find the maximum Y value
    y_setmin = [min(Y_all)]  # Find the minimum Y value
    
    # -------------------------------------------------
    # Step 2: Calculate the geometrical center (centroid)
    #         by averaging the maximum and minimum values.
    # -------------------------------------------------
    
    h = (max(x_setmax) + min(x_setmin)) / 2  # X center
    k = (max(y_setmax) + min(y_setmin)) / 2  # Y center
    
    # -------------------------------------------------
    # Step 3: Return the calculated center coordinates.
    # -------------------------------------------------
    
    return h, k

#########################################################################################################

"""
======================================
  Function: calculate_radius
======================================

This function calculates two important radius metrics for a set of spot points:
1. Geometrical Radius (GEO_Radius): The maximum Euclidean distance from the 
   center of the spot to the furthest point.
2. Root Mean Square Radius (RMS_Radius): The square root of the mean squared 
   distances of all points to the center.

Parameters:
- X_lists (list or np.array): List or array of x coordinates of the spot points.
- Y_lists (list or np.array): List or array of y coordinates of the spot points.
- center_x (float): The x-coordinate of the geometrical center.
- center_y (float): The y-coordinate of the geometrical center.

Steps:
1. Compute the Euclidean distance of each point to the center.
2. Find the maximum distance (Geometrical Radius).
3. Compute the root mean square of the distances (RMS Radius).
4. Return the center coordinates along with the two calculated radii.

Returns:
- (center_x, center_y) (tuple): The center coordinates of the spot diagram.
- GEO_Radius (float): The maximum distance from the center.
- RMS_Radius (float): The root mean square of all distances from the center.

======================================
"""

def calculate_radius(X_lists, Y_lists, center_x, center_y):
    
    # Compute Euclidean distances from each point
    # to the geometrical center (center_x, center_y).
    distances = np.sqrt((X_lists - center_x)**2 + (Y_lists - center_y)**2)
    
    # Calculate the Geometrical Radius (maximum distance).
    GEO_Radius = np.max(distances)  # Furthest point from the center
    
    # Calculate the RMS Radius (root of the mean squared distances).
    RMS_Radius = np.sqrt(np.mean(distances ** 2))  # Root Mean Square of distances
    
    # Return the center coordinates and the two radius metrics.
    return (center_x, center_y), GEO_Radius, RMS_Radius

#########################################################################################################

"""
======================================
  Function: plot_spot_diagram
======================================

This function generates and displays the spot diagram for a specified 
field configuration in an optical system. It represents the distribution 
of rays at the image plane for three different wavelengths, along with 
the Airy disk, the geometrical radius, and the RMS radius.

Parameters:
- Xa, Ya, Xb, Yb, Xc, Yc (list): Lists of coordinates representing 
  the intersection of rays with the image plane for three wavelengths:
  - Xa, Ya -> Wavelength 0.35 μm (blue)
  - Xb, Yb -> Wavelength 0.43 μm (green)
  - Xc, Yc -> Wavelength 0.55 μm (red)
- x_airy, y_airy (list): Coordinates for the Airy disk representation.
- field_name (str): Identifier for the field being analyzed. It determines 
  which labels (X, Y) are displayed in the plot.

Steps:
1. Concatenate the coordinates for all wavelengths into single lists.
2. Compute the geometrical center of the spot diagram.
3. Calculate the geometrical (GEO) radius and the root-mean-square (RMS) radius.
4. Recenter the spots to the origin (0, 0) based on the computed center.
5. Generate the spot diagram plot:
   - Plot the three wavelengths with different colors.
   - Plot the Airy disk as a dashed circle.
   - Plot the GEO and RMS radii as dashed and dotted circles, respectively.
6. Configure the plot aesthetics:
   - Increase label sizes for better readability.
   - Conditionally add axis labels based on the field being plotted.
7. Save the plot as a PNG file named after the field identifier.
8. Display the plot.

Returns:
- Geo_r (float): The geometrical radius of the spot diagram.
- Rms_r (float): The root-mean-square radius of the spot diagram.

======================================
"""

def plot_spot_diagram(Coordinates, fields, x_airy, y_airy, field_name, custom_name=""):
    # ==============================
    # Define fixed output path 
    # ==============================
    script_dir = os.path.abspath(os.path.dirname(__file__))
    base_path = os.path.abspath(os.path.join(script_dir, '..', '..'))  # Go two levels up from current script
    output_folder = os.path.join(base_path, 'Aux_Material', 'Images')

    # Create the output folder if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
        print(f"Folder created at: {output_folder}")
    else:
        print(f"Folder already exists: {output_folder}")
        
    Xa = Coordinates[0][0]
    Ya = Coordinates[0][1]
    
    Xb = Coordinates[1][0]
    Yb = Coordinates[1][1]
    
    Xc = Coordinates[2][0]
    Yc = Coordinates[2][1]
    
    field_x = fields[0]
    field_y = fields[1]
    
    # Concatenate coordinates for all wavelengths
    X_all = np.concatenate([Xa, Xb, Xc])
    Y_all = np.concatenate([Ya, Yb, Yc])

    # Compute the geometrical center
    h, k = calculate_geometrical_center(X_all, Y_all)

    # Calculate the geometrical (GEO) radius and RMS radius
    _, Geo_r, Rms_r = calculate_radius(X_all, Y_all, h, k)

    # Recenter the spots to the origin (0, 0)
    Xa -= h
    Ya -= k
    Xb -= h
    Yb -= k
    Xc -= h
    Yc -= k

    # Generate the spot diagram plot
    plt.figure()

    # Plot each wavelength in different colors
    plt.plot(Xa, Ya, 'x', color='blue', label='Wavelength 0.35')
    plt.plot(Xb, Yb, 'x', color='green', label='Wavelength 0.43')
    plt.plot(Xc, Yc, 'x', color='red', label='Wavelength 0.55')

    # Plot the Airy disk representation as a dashed line
    plt.plot(x_airy, y_airy, color="k", linestyle='dashed')
    
    
    # Plot the geometrical radius as a dashed circle
    circle = patches.Circle((0.0, 0.0), radius=Geo_r,
                            edgecolor='k', linestyle='-.', linewidth=1.5,
                            facecolor='none')
    plt.gca().add_patch(circle)

    # Plot the RMS radius as a dotted circle
    circle_rms = patches.Circle((0.0, 0.0), radius=Rms_r,
                                edgecolor='k', linestyle=':', linewidth=2.0,
                                facecolor='none')
    plt.gca().add_patch(circle_rms)

    # Configure the plot aesthetics
    plt.tick_params(axis='both', which='major', labelsize=12, length=6, width=1)
    plt.tick_params(axis='both', which='minor', labelsize=10, length=4, width=0.8)
    plt.gca().xaxis.set_tick_params(pad=2)
    plt.gca().yaxis.set_tick_params(pad=2)
    
    # Título del campo arriba
    plt.title(f'OBJ: {field_x:.3f}, {field_y:.3f} deg', fontsize=15)
    
    # Subtítulo con coordenadas abajo del gráfico
    plt.xlabel(f'IMA: {h:.3f}, {k:.3f} mm', fontsize=15, color='black', labelpad=0.8)

   

    # Ensure the plot has a square aspect ratio
    plt.axis('square')
    plt.xticks([])
    plt.yticks([])
    
    ax = plt.gca()

    # Tamaño de la caja (en mm) a partir de los límites actuales
    x_min, x_max = ax.get_xlim()
    box_size = x_max - x_min  # como es cuadrado, x y y tienen el mismo tamaño
    
    # --- Barra vertical de escala al costado derecho ---
    bar_x = 1.00       # 2% fuera del borde derecho del eje
    tick_w = 0.02      # ancho de las marquitas superior e inferior (en fracción del eje)
    txt_x = 1.02       # texto un poco a la derecha de la barra
    
    # Línea vertical (|------| versión vertical)
    ax.plot([bar_x, bar_x], [0, 1],
            transform=ax.transAxes, color='black', linewidth=1.2,
            clip_on=False, zorder=10)
    
    # Marcas en los extremos (arriba y abajo)
    ax.plot([bar_x - tick_w, bar_x + tick_w], [0, 0],
            transform=ax.transAxes, color='black', linewidth=1.2,
            clip_on=False, zorder=10)
    ax.plot([bar_x - tick_w, bar_x + tick_w], [1, 1],
            transform=ax.transAxes, color='black', linewidth=1.2,
            clip_on=False, zorder=10)    # marca arriba
    
    # Texto vertical con el tamaño de la caja
    ax.text(txt_x, 0.5, f'{box_size*1000:.2f} mm',
        transform=ax.transAxes, rotation=90,
        va='center', ha='left', fontsize=15, color='black',
        clip_on=False)

    # # Colocar el texto centrado en el costado
    # plt.text(x_pos + 0.02*box_size, (y_start + y_end)/2,
    #          f'{box_size*1000:.3f} mm',
    #          ha='left', va='center', fontsize=12, color='black', rotation=90)

    # ==============================
    # Save as EPS
    # ==============================
    base_filename = f'spot_diagram_{field_name}_{custom_name}' if custom_name else f'spot_diagram_{field_name}'
    pdf_path = os.path.join(output_folder, base_filename + '.pdf')
    plt.savefig(pdf_path, format='pdf', bbox_inches='tight', pad_inches=0.01)
    print(f"EPS image saved at: {pdf_path}")
    
    
    # Display the plot
    plt.show()
    
    # Return the calculated radii
    return Geo_r, Rms_r
    
    

#########################################################################################################

def trace_rays(Pup, wavelengths, Rays):
    
    """
    Traces rays for a given pupil configuration and list of wavelengths.

    Parameters:
    - Pup: The pupil object.
    - wavelengths: List of wavelengths to trace.
    - Rays: The ray keeper object.

    Returns:
    - A list of tuples with traced coordinates for each wavelength.
    """
    
    # The array of rays: transfer from the unit pupil to the real pupil
    x, y, z, L, M, N = Pup.Pattern2Field()
    results = []

    # Loop through each wavelength and perform the trace
    for w in wavelengths:
        Kos.TraceLoop(x * 0.99, y * 0.99, z * 0.99, L, M, N, w, Rays, clean=1)
        # Collect the traced rays from the image plane
        results.append(Rays.pick(-1))

    return results

#########################################################################################################

"""
======================================
  Function: ProcessPattern2Field
======================================

This function processes the ray tracing for a specific wavelength (wv) 
using the provided system information (sys_info), which includes the 
optical system, the raykeeper instance, and the pupil calculation.

Parameters:
- wv: The wavelength for the ray tracing process.
- sys_info: A tuple containing:
    - system: The optical system instance.
    - raykeeper: The raykeeper instance for tracking rays.
    - P: The pupil calculation instance for field representation.

Steps:
1. Extract the ray coordinates and direction cosines from the pupil sampling.
2. Define the starting point for the rays, slightly adjusted (0.99 factor).
3. Perform ray tracing through the optical system at the specified wavelength.
4. Store the traced ray data in the raykeeper instance.
5. Extract the traced data from the third surface for further processing.

Returns:
- X, Y, Z, L, M, N: Coordinates and direction cosines of the rays after tracing.

======================================
"""

def ProcessPattern2Field(wv, sys_info):
    
    # Unpack the information of the entire system:
    system = sys_info[0]    # Optical system with the defined elements and configuration 
    raykeeper = sys_info[1] # RayKeeper instance for storing and accessing ray tracing results.
    P = sys_info[2]         # Pupil calculation for the system
    
    # The array of rays: transfer from the unit pupil to the real pupil
    xR, yR, zR, LR, MR, NR = P.Pattern2Field()
    # Set of origin coordinates of the rays
    pSource_0 = [xR[0]*0.99, yR[0]*0.99, zR[0]*0.99]
    # Set of origin director cosines of the rays
    dCos = [LR[0], MR[0], NR[0]]
    
    # Trace of the rays in a sequential way through every surface taking into consideration
    # the set of origin coordinates and director cosines and the wavelength
    system.Trace(pSource_0, dCos, wv)
    
    # Store the information of the traced ray
    raykeeper.push()
    
    # Access the information of the surface
    X, Y, Z, L, M, N = raykeeper.pick(3)
  
    return X, Y, Z, L, M, N


#########################################################################################################

                
"""
======================================
  Function: MS_Telescope
======================================

This function computes the ABCD matrix (paraxial matrix) for a telescope 
configuration, considering a deistanc shift 'd' and a set of constants.

Parameters:
- d: Distance shift applied to the telescope's optical path.
- cons: A tuple containing the coefficients:
    - b_1: Power coefficient for the primary mirror.
    - b_2: Power coefficient for the secondary mirror.
    - d_1: Separation distance between the primary and secondary mirrors.

Steps:
1. Define the ABCD matrix for each segment of the telescope path:
   - M6 to M10 represent different sections of the optical path.
2. Multiply the matrices to obtain the global paraxial matrix (MSJS).

Returns:
- MSJS: The final paraxial matrix representing the telescope configuration.

======================================
"""
def MS_Telescope(d, cons):
    
    # Unpack the constants
    b_1, b_2, d_1 = cons
    
    # Define the ABCD matrices for each optical segment
    M6  = np.array([1, 0, d_1 + d, 1]).reshape(2, 2)
    M7  = np.array([1, b_2, 0, 1]).reshape(2, 2)
    M8  = np.array([1, 0, d_1, 1]).reshape(2, 2)
    M9  = np.array([1, b_1, 0, 1]).reshape(2, 2)
    M10 = np.array([1, 0, 1, 1]).reshape(2, 2)

    # Compute the global paraxial matrix
    MSJS = M6 @ M7 @ M8 @ M9 @ M10
    return MSJS


#########################################################################################################

"""
======================================
  Function: R_RMS_delta
======================================

This function calculates the Root Mean Square (RMS) radius for the 
intersection coordinates of rays with a surface or plane.

Parameters:
- Z1: Distance along the optical axis to the surface or plane.
- L, M, N: Direction cosines of the rays.
- X0, Y0: Initial coordinates of the rays.

Steps:
1. Prevent division by zero by replacing zeros in N with NaN.
2. Compute the intersection coordinates (X1, Y1) on the plane at Z1.
3. Calculate the mean position (cenX, cenY) of the intersection points.
4. Compute the shift of each point from the mean position.
5. Calculate the squared radius for each ray from the mean position.
6. Compute the RMS value of these distances.

Returns:
- R_RMS: The Root Mean Square (RMS) radius, representing the average 
  spread of the rays around their mean intersection point.

======================================
"""

def R_RMS_delta(Z1, L, M, N, X0, Y0):
    
    # Avoid a crash for division by zero
    N = np.where(N == 0, np.nan, N)
    
    # Calculation for the intersection coordinates of rays with a surface or plane 
    X1 = ((L / N) * Z1) + X0
    Y1 = ((M / N) * Z1) + Y0
    
    # Calculate the mean of the intersection coordinates
    cenX = np.mean(X1)
    cenY = np.mean(Y1)
    
    # Determine the shift from the mean position
    x1 = (X1 - cenX)
    y1 = (Y1 - cenY)
    
    # Calculate the squared radius for each ray
    R2 = ((x1 * x1) + (y1 * y1))
    
    # Compute the RMS radius
    R_RMS = np.sqrt(np.mean(R2))
    
    return R_RMS

#########################################################################################################

"""
======================================
  Function: BestFocus
======================================

This function optimizes the position of the last optical surface 
in the system to minimize the RMS radius of the spot diagram, 
representing the best focus position.

Parameters:
- X, Y, Z: Coordinates of the rays.
- L, M, N: Direction cosines of the rays.
- system: The optical system configuration.
- mod: (optional) If set to 1, updates the system with the best focus position.

Steps:
1. Initialize the starting point for the focus shift (delta_Z).
2. Define a tuple with the ray data (direction cosines and coordinates).
3. Use a solver to find the Z displacement that minimizes the RMS radius.
4. If `mod` is set to 1, update the system's thickness for the last surface.
5. Return the modified system and the optimized displacement value.

Returns:
- system: The updated optical system if `mod` is 1.
- v[0]: The optimal displacement value for best focus.

======================================
"""

def BestFocus(X, Y, Z, L, M, N, system, mod=1):

    # Initialize the starting point for the Z-axis shift (delta_Z)
    delta_Z = 0
    
    # Prepare the data for optimization: direction cosines and initial positions
    ZZ = (L, M, N, X, Y)
    
    # Solve for the optimal Z-axis shift that minimizes the RMS radius
    v = scipy.optimize.fsolve(R_RMS_delta, delta_Z, args=ZZ)
    
    # If mod is set to 1, update the optical system with the new focus position
    if mod == 1:
        
        # Adjust the thickness of the last surface in the optical system
        system.SDT[-2].Thickness = system.SDT[-2].Thickness + v[0]
        
        # Apply the changes to the system configuration
        system.SetData()
        system.SetSolid()
        
    # Return the modified system and the optimal displacement
    return system, v[0]

#########################################################################################################

"""
======================================
  Function: BestRMS
======================================

This function calculates the RMS radius for a set of 
coordinates and direction cosines for multiple nodes and arms following the 
Gaussian Quadrature in the optical system. It determines the optimal focus 
position and calculates the RMS value for each node in the original configuration.

Parameters:
- info_coordinates: A tuple containing:
    - set_x, set_y, set_z: Coordinates of the rays.
    - set_l, set_m, set_n: Direction cosines of the rays.
- system: The optical system configuration.

Steps:
1. Unpack the coordinate information from the tuple.
2. Determine the number of nodes and arms in the configuration.
3. Reshape the coordinates into a single list of elements for processing.
4. Optimize the focus position using `BestFocus`.
5. Iterate through each node and calculate the RMS radius.
6. Store the RMS value for each node in a list.

Returns:
- rms: A list of RMS values for each node.

======================================
"""

def BestRMS(info_coordinates, system):
    
    # Unpack coordinates information
    set_x = info_coordinates[0]
    set_y = info_coordinates[1]
    set_z = info_coordinates[2]
    set_l = info_coordinates[3]
    set_m = info_coordinates[4]
    set_n = info_coordinates[5]
    
    # Obtain the number of nodes and arms
    n_nodes = len(set_x)
    n_arms = len(set_x[0])
    
    # Reshape the coordinates of information into a single list with n_nodes * n_arms elements
    x_reshaped = set_x.reshape(1, n_nodes * n_arms)
    y_reshaped = set_y.reshape(1, n_nodes * n_arms)
    z_reshaped = set_z.reshape(1, n_nodes * n_arms)
    l_reshaped = set_l.reshape(1, n_nodes * n_arms)
    m_reshaped = set_m.reshape(1, n_nodes * n_arms)
    n_reshaped = set_n.reshape(1, n_nodes * n_arms)
    
    # Initialize the counter for iteration
    i_ray = 0
    
    # Optimize the position of the best focus without modifying the system (mod = 0)
    system, deltaZ = BestFocus(x_reshaped, y_reshaped, z_reshaped, l_reshaped, m_reshaped, n_reshaped, system, mod=0)
    
    # Initialize a list to keep track of the RMS values for each node
    rms = []
    
    # Iterate through each node and calculate the RMS radius
    while i_ray < n_nodes:
        rms.append(R_RMS_delta(deltaZ, set_l[i_ray], set_m[i_ray], set_n[i_ray], set_x[i_ray], set_y[i_ray]))
        i_ray += 1
    
    return rms

#########################################################################################################

"""
======================================
  Function: Min_Dis_point
======================================

This function calculates the coordinates of the point where the distance 
is minimal between two lines defined by the initial coordinates and 
direction cosines. It solves the geometric problem of finding the closest
approach between two skew rays.

Parameters:
- x1, y1, z1 (float): Initial coordinates of the first ray.
- x2, y2, z2 (float): Initial coordinates of the second ray.
- L1, M1, N1 (float): Direction cosines of the first ray.
- L2, M2, N2 (float): Direction cosines of the second ray.

Steps:
1. Compute the cross product of the direction vectors to obtain the normal vector.
2. Calculate the parameter 't' that determines the intersection in the minimum distance.
3. Compute the coordinates of the minimal distance point using parameter 't'.

Returns:
- x, y, z (float): Coordinates of the point where the distance is minimal.

======================================
"""

def Min_Dis_point(x1, y1, z1, x2, y2, z2, L1, M1, N1, L2, M2, N2):

    # Compute the vector perpendicular to the two direction vectors
    Vx = M1 * N2 - N1 * M2
    Vy = N1 * L2 - L1 * N2
    Vz = L1 * M2 - M1 * L2

    # Calculate the parameter 't' for the line of intersection
    t = (x2 - x1) / Vx

    # Coordinates of the point where the distance is minimal
    x = x1 + t * Vx
    y = y1 + t * Vy
    z = z1 + t * Vz

    return x, y, z

#########################################################################################################

"""
======================================
  Function: calculate_average
======================================

This function calculates the average of a list of numbers.

Parameters:
- lst (list): A list of numeric values.

Returns:
- average (float): The average of the numbers in the list. 
  Returns 0 for an empty list.

======================================
"""

def calculate_average(lst):

    if not lst:
        return 0  # Return 0 for an empty list
    
    total = sum(lst)
    average = total / len(lst)
    
    return average

# ======================================
#  Section for sphere aberration
# ======================================

"""
======================================
  Function: generate_radios
======================================

This function generates a list of random radii values, with the maximum 
radius of 1.0 always included in the list. The generated radii values 
are sorted in descending order.

Parameters:
- count (int): The number of random radii values to generate.

Steps:
1. If the count is 1, return [1.0] directly as there is only the maximum radius.
2. Generate `count - 1` random float values between 0 and 1.
3. Append the maximum radius value (1.0) to the list.
4. Sort the list in descending order.
5. Return the sorted list.

Returns:
- list: A sorted list of radii in descending order. If the count is 1, 
  the list contains only the value [1.0].
  
======================================
"""
 
def generate_radios(count):
    
    if count == 1:
        return [1.0]
    
    # Generate the list of random numbers and append 1.0
    random_numbers = sorted([random.random() for _ in range(count - 1)] + [1.0], reverse=True)
    
    return random_numbers


#########################################################################################################

"""
======================================
  Function: generate_angles
======================================

This function generates a list of random angles between 0 and 90 degrees, 
including 0. The generated angles are sorted in ascending order.

Parameters:
- count (int): The number of random angles to generate.

Steps:
1. If the count is 1, return [0] directly as the only angle.
2. Generate `count - 1` random float values between 0 and 90.
3. Append 0 to the list.
4. Sort the list in ascending order.
5. Return the sorted list.

Returns:
- list: A sorted list of angles in ascending order. If the count is 1, 
  the list contains only the value [0].

======================================
"""

def generate_angles(count):
    if count == 1:
        return [0]
    
    # Generate the list of random angles and append 0
    random_angles = sorted([random.uniform(0, 90) for _ in range(count - 1)] + [0])
    
    return random_angles

#########################################################################################################

"""
======================================
  Function: Sphere_subtract
======================================

This function computes the absolute differences between each element 
in a list and the first element of that list. It is specifically used 
to calculate the deviation of marginal rays from the chief ray in 
total optical path analysis.

Parameters:
- lst (list): A list of numeric values representing total optical path lengths 
             for different rays.
             
Steps:
1. Initialize an empty list to store results.
2. Iterate over each element in the input list.
3. Compute the absolute difference between the chief ray (lst[0]) 
   and the current element.
4. Append the difference to the result list.
5. Return the populated result list.
             

Returns:
- result (list): A list containing the absolute differences between 
                 each element and the first element (chief ray).
    
======================================
"""

def Sphere_subtract(lst):

    result = []
    
    for i in range(len(lst)):
        subtraction = np.abs(lst[0] - lst[i])
        result.append(subtraction)
        
    return result

# ======================================
#  Section for coma aberration
# ======================================

"""
======================================
  Function: generate_radiosc
======================================

This function generates two lists based on a sample size:
1. A list of random numbers reshaped as a 1D list representing 
   pupil radii.
2. A list of original random numbers.

Parameters:
- Sample (int): The number of random numbers to generate.

Steps:
1. Initialize two empty lists:
   - `PRcrtheta`: To store the reshaped list of tuples (r, r).
   - `PRcrtheta_i`: To temporarily store each tuple.
2. Generate `Sample` random numbers in the range [0, 1).
3. Force the first element to be 1.0 to ensure maximum radius.
4. Loop through the generated random numbers:
   - Create tuples of the form (r, r) and store them in `PRcrtheta_i`.
   - Append these tuples to `PRcrtheta`.
5. Reshape `PRcrtheta` as a 1D list (flattened).
6. Convert the NumPy array back to a list format.
7. Return the reshaped list and the original list of random numbers.

Returns:
- PRcrtheta (list): A list of random numbers, reshaped as a 
  1D list, each represented as (r, r).
- random_numbers (list): A list of random numbers, including the 
  extreme value of 1.0 as the first element.



======================================
"""

def generate_radiosc(Sample):
    # Initialize lists for storing radii and formatted tuples
    PRcrtheta = []
    PRcrtheta_i = []

    # Generate random numbers and set the first one to 1.0
    random_numbers = [random.random() for _ in range(Sample)]
    random_numbers[0] = 1.0
    
    # Loop through the list of random numbers to create (r, r) tuples
    for i in range(len(random_numbers)):
        PRcrtheta_i.append((random_numbers[i], random_numbers[i]))
        PRcrtheta.append(PRcrtheta_i[i])
    
    # Reshape the list to a 1D array and convert back to a list
    PRcrtheta = np.array(PRcrtheta).reshape((2 * Sample,))
    PRcrtheta = PRcrtheta.tolist()  

    # Return the formatted list and the original list
    return PRcrtheta, random_numbers

#########################################################################################################


"""
======================================
  Function: generate_anglesc
======================================

This function generates a list of angles in the form (90, -90) to be used
for pupil-based ray generation in the coma aberration analysis.

Parameters:
- Sample (int): The number of angles to generate.

Steps:
1. Initialize two empty lists to store the angles.
2. Loop through the sample size to create tuples (90, -90).
3. Store each tuple in the list.
4. Reshape the list into a 1D array for easy manipulation.
5. Convert back to a Python list and return.

Returns:
- Ptcrtheta (list): A list of angles, reshaped as a 1D list to be used in
  ray tracing simulations.



======================================
"""

def generate_anglesc(Sample):
    
    # Initialize lists for storing angles and formatted tuples
    Ptcrtheta = []
    Ptcrtheta_i = []
    angle = 90.
    
    # Loop through the list of random numbers to create (90, -90) tuples
    for i in range(Sample):
        Ptcrtheta_i.append((angle, -angle))
        Ptcrtheta.append(Ptcrtheta_i[i])
    
    # Reshape the list to a 1D array and convert back to a list
    Ptcrtheta = np.array(Ptcrtheta)
    Ptcrtheta = Ptcrtheta.reshape((2*Sample,))
    Ptcrtheta = Ptcrtheta.tolist()
    
    return Ptcrtheta

#########################################################################################################

"""
======================================
  Function: Coma_Substraction
======================================

This function calculates the radial deviation of marginal rays from the 
principal (chief) ray in the context of coma aberration analysis. It finds 
the minimal intersection points for pairs of marginal rays and compares their 
radial distances to the chief ray.

Parameters:
- Sample (int): The number of samples or marginal rays to analyze.
- Inf_Rays (tuple): A tuple containing:
    - Siscoor (list): A list of coordinates (x, y, z) for all the rays.
    - Dircos (list): A list of direction cosines (L, M, N) for all the rays.

Steps:
1. Extract the (x, y) coordinates of the chief ray (first element of Siscoor).
2. Initialize lists to store coordinates and direction cosines for the 
   marginal rays.
3. Loop through each sample, and for each pair of marginal rays:
   - Extract their coordinates and direction cosines.
   - Calculate the minimal intersection point using `Min_Dis_point`.
   - Compute the radial distances (r) of the intersection points for:
     a) The marginal rays.
     b) The chief ray.
   - Compute the absolute difference in radii and store it in `diff`.
   
4. The difference represents the displacement of the marginal rays 
   with respect to the principal ray in the image plane.

Returns:
- diff (list): A list containing the absolute differences in radial 
               distances between the minimal intersection points of 
               the marginal rays and the principal ray.

======================================
"""

def Coma_Substraction(Sample, Inf_Rays):
    
    # Extract coordinates and direction cosines
    Siscoor = Inf_Rays[0]
    Dircos = Inf_Rays[1]
    
    # Chief ray information
    xp0 = Siscoor[0][0]
    yp0 = Siscoor[1][0]

    # Initialize lists for marginal ray information
    xma, xmb, yma, ymb, zma, zmb = [], [], [], [], [], []
    Lma, Lmb, Mma, Mmb, Nma, Nmb = [], [], [], [], [], []
    diff = []
    
    # Loop through each sample
    for i in range(Sample):
        
        # Extract marginal ray information
        xma.append(Siscoor[0][2*i+1])
        xmb.append(Siscoor[0][2*i+2])
        yma.append(Siscoor[1][2*i+1])
        ymb.append(Siscoor[1][2*i+2])
        zma.append(Siscoor[2][2*i+1])
        zmb.append(Siscoor[2][2*i+2])
        
        Lma.append(Dircos[0][2*i+1])
        Lmb.append(Dircos[0][2*i+2])
        Mma.append(Dircos[1][2*i+1])
        Mmb.append(Dircos[1][2*i+2])
        Nma.append(Dircos[2][2*i+1])
        Nmb.append(Dircos[2][2*i+2])
        
        # Find the minimal intersection point
        
        xm, ym, zm = Min_Dis_point(xma[i], yma[i], zma[i], Lma[i], Mma[i],
                                   Nma[i], xmb[i], ymb[i], zmb[i], Lmb[i], 
                                   Mmb[i], Nmb[i])
        
        # Calculate the radial distances and their difference
        
        rm = np.sqrt(xm**2 + ym**2)
        rp = np.sqrt(xp0**2 + yp0**2)
        diff.append(np.abs(rm - rp))
        
    return diff

# ======================================
#  Section for astigmatism aberration
# ======================================

"""
======================================
  Function: generate_radiosa
======================================

This function generates a list of random radii and arranges them in 
a specific format for astigmatism analysis. The radii are scaled by a 
constant value (`cons`) and formatted as tuples of the form 
(r/c, -r/c, r/c, -r/c) for each randomly generated radius.

Parameters:
- Sample (int): The number of radii to generate and the number of times 
                the list of radii will be stored.
- cons (float): The constant value used to scale the radii.

Steps:
1. Initialize two empty lists for storing formatted radii.
2. Generate random radii values between 0 and 1 for the given sample size.
3. Set the first radius to 1, representing the maximum edge of the pupil.
4. Loop through the list of generated radii:
   - Format each radius as a tuple (r/c, -r/c, r/c, -r/c).
   - Append the formatted tuple to the main list.
5. Reshape the list into a 1D array for easy manipulation.
6. Return the reshaped list and the original random numbers.

Returns:
- PRartheta (list): A list of scaled radii, reshaped as a 1D list.
- random_numbers (list): The original list of randomly generated radii.

======================================
"""

def generate_radiosa(Sample, cons):
    
    # Initialize lists for storing radii and formatted tuples
    PRartheta = []
    PRartheta_i = []
    
    # Generate random radii values between 0 and 1 for the given sample size
    random_numbers = [random.random() for _ in range(Sample)]
    
    # Ensure the maximum edge is represented
    random_numbers[0] = 1
    
    # Loop through the list of generated radii
    for i in range(len(random_numbers)):
        
        # Create tuples of the form (r/c, -r/c, r/c, -r/c)
        PRartheta_i.append((random_numbers[i]/cons, -random_numbers[i]/cons,
                            random_numbers[i]/cons, -random_numbers[i]/cons))
        
        # Append the formatted tuple to the main list
        PRartheta.append(PRartheta_i[i])
    
    # Reshape the list into a 1D array for easy manipulation
    PRartheta = np.array(PRartheta)
    PRartheta = PRartheta.reshape((4*Sample,))
    
    # Return the reshaped list and the original random numbers
    return PRartheta, random_numbers

#########################################################################################################

"""
======================================
  Function: generate_anglesa
======================================

This function generates and stores a list of angles in the form 
(90, -90, 0, 0) a specified number of times based on the sample size.
These angles are used for pupil-based ray generation in astigmatism 
analysis.

Parameters:
- Sample (int): The number of times the list of angles will be 
                generated and stored.

Steps:
1. Initialize two empty lists for storing formatted angle tuples.
2. Define two specific angles: 90 degrees for the first two components 
   and 0 degrees for the last two components.
3. Loop through the sample size:
   - Create a tuple of the form (90, -90, 0, 0).
   - Append the tuple to the main list.
4. Reshape the list into a 1D array for easy manipulation.
5. Return the reshaped list.

Returns:
- Ptartheta (list): A list of angles, reshaped as a 1D list to be used 
                    in ray tracing simulations for astigmatism analysis.

======================================
"""

def generate_anglesa(Sample):
    
    # Initialize lists for storing angles and formatted tuples
    Ptartheta = []
    Ptartheta_i = []
    
    # Define the angle values
    angle_a = 90
    angle_b = 0
    
    # Loop through the sample size to generate tuples
    for i in range(Sample):
        
        # Create a tuple with the structure (90, -90, 0, 0)
        Ptartheta_i.append((angle_a, angle_a, angle_b, angle_b))
        
        # Append the formatted tuple to the main list
        Ptartheta.append(Ptartheta_i[i])
        
    # Reshape the list into a 1D array for easy manipulation
    Ptartheta = np.array(Ptartheta)
    Ptartheta = Ptartheta.reshape((4 * Sample,))
    
    # Return the reshaped list
    return Ptartheta

#########################################################################################################

"""
======================================
  Function: Astigmatism_Substraction
======================================

This function performs the subtraction of astigmatism between sagittal and 
tangential rays for a given sample size. It calculates the minimum distance 
points for both sagittal and tangential planes and computes the absolute 
difference between these distances.

Parameters:
- Sample (int): The number of samples to process.
- Inf_Rays (tuple): A tuple containing the spatial coordinates and direction 
                    cosines of the rays:
    - Inf_Rays[0]: Spatial coordinates of the rays.
    - Inf_Rays[1]: Direction cosines of the rays.

Steps:
1. Initialize lists to store coordinates and direction cosines for:
    - Sagittal plane (xms, yms, zms, Lms, Mms, Nms)
    - Tangential plane (xmt, ymt, zmt, Lmt, Mmt, Nmt)
    
2. Loop through the sample size:
    - Extract information from `Inf_Rays` for each ray in sagittal and 
      tangential planes.
    - Store the information in their corresponding lists.
    
3. For each sample:
    - Calculate the minimum distance point for the sagittal rays using 
      `Min_Dis_point`.
    - Calculate the minimum distance point for the tangential rays.
    - Compute the Euclidean distance for both sagittal and tangential points.
    - Subtract these distances and store the absolute difference.

4. Return the list of differences representing the astigmatism aberration.

Returns:
- drs (list): A list of the absolute differences in astigmatism for each sample.

======================================
"""

def Astigmatism_Substraction(Sample, Inf_Rays):
    
    # Extract spatial coordinates and direction cosines from the tuple
    Siscoor = Inf_Rays[0]
    Dircos = Inf_Rays[1]
 
    # Initialize lists for Sagittal Plane
    xmsa, xmsb, ymsa, ymsb, zmsa, zmsb = [], [], [], [], [], []
    Lmsa, Lmsb, Mmsa, Mmsb, Nmsa, Nmsb = [], [], [], [], [], []
    
    # Initialize lists for Tangential Plane
    xmta, xmtb, ymta, ymtb, zmta, zmtb = [], [], [], [], [], []
    Lmta, Lmtb, Mmta, Mmtb, Nmta, Nmtb = [], [], [], [], [], []
    
    # List to store the differences between sagittal and tangential distances
    drs = []
    
    # Loop through each sample to extract information
    for i in range(Sample):
        
        # --------------------------
        # Sagittal Plane Extraction
        # --------------------------
        
        # Coordinates
        xmsa.append(Siscoor[0][4 * i])
        xmsb.append(Siscoor[0][4 * i + 1])
        ymsa.append(Siscoor[1][4 * i])
        ymsb.append(Siscoor[1][4 * i + 1])
        zmsa.append(Siscoor[2][4 * i])
        zmsb.append(Siscoor[2][4 * i + 1])
        
        # Direction Cosines
        Lmsa.append(Dircos[0][4 * i])
        Lmsb.append(Dircos[0][4 * i + 1])
        Mmsa.append(Dircos[1][4 * i])
        Mmsb.append(Dircos[1][4 * i + 1])
        Nmsa.append(Dircos[2][4 * i])
        Nmsb.append(Dircos[2][4 * i + 1])
        
        # --------------------------
        # Tangential Plane Extraction
        # --------------------------
        
        # Coordinates
        xmta.append(Siscoor[0][4 * i + 2])
        xmtb.append(Siscoor[0][4 * i + 3])
        ymta.append(Siscoor[1][4 * i + 2])
        ymtb.append(Siscoor[1][4 * i + 3])
        zmta.append(Siscoor[2][4 * i + 2])
        zmtb.append(Siscoor[2][4 * i + 3])
        
        # Direction Cosines
        Lmta.append(Dircos[0][4 * i + 2])
        Lmtb.append(Dircos[0][4 * i + 3])
        Mmta.append(Dircos[1][4 * i + 2])
        Mmtb.append(Dircos[1][4 * i + 3])
        Nmta.append(Dircos[2][4 * i + 2])
        Nmtb.append(Dircos[2][4 * i + 3])
        
        # --------------------------
        # Distance Calculation
        # --------------------------
        
        # Calculate the minimum distance points for the sagittal rays
        xs, ys, zs = Min_Dis_point(
            xmsa[i], ymsa[i], zmsa[i], 
            Lmsa[i], Mmsa[i], Nmsa[i], 
            xmsb[i], ymsb[i], zmsb[i], 
            Lmsb[i], Mmsb[i], Nmsb[i]
        )
        
        # Calculate the minimum distance points for the tangential rays
        xt, yt, zt = Min_Dis_point(
            xmta[i], ymta[i], zmta[i], 
            Lmta[i], Mmta[i], Nmta[i], 
            xmtb[i], ymtb[i], zmtb[i], 
            Lmtb[i], Mmtb[i], Nmtb[i]
        )
        
        # Calculate the Euclidean distances
        rs = np.sqrt(xs**2 + ys**2 + zs**2)   # Sagittal radius
        rt = np.sqrt(xt**2 + yt**2 + zt**2)   # Tangential radius
        
        # Append the absolute difference to the results
        drs.append(np.abs(rs - rt))
        
    # Return the list of differences
    return drs

#########################################################################################################

def guardar_lista_en_csv(lista, nombre_archivo):
    
    with open(nombre_archivo, 'w', newline='') as archivo_csv:
        
        writer = csv.writer(archivo_csv)
        writer.writerow(lista)
        
    print(f"La lista se ha guardado en el archivo {nombre_archivo}.")


#########################################################################################################
                
def data_exists(data):
    
    with open('aberration_data.csv', 'r', newline='') as csvfile:
        csv_reader = csv.reader(csvfile)
        for row in csv_reader:
            if row == data:
                return True
    return False



























