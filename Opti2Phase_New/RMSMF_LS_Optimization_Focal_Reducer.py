
"""
======================================
  Section: RMS-Based Refinement Stage
======================================

Description:
------------
After optimizing the optical system using a physically grounded merit function (MF), 
a second optimization stage is performed to refine the solution. This refinement 
focuses on minimizing the RMS spot size at the image plane—a metric widely adopted 
in commercial optical design due to its sensitivity to residual aberrations and 
its practical relevance to image quality.

Starting from the configuration obtained via the classical Least Squares (LS) method, 
this stage adjusts curvatures and spacings to reduce the RMS radius across various 
field positions and wavelengths. The process uses Gaussian quadrature sampling 
to ensure accurate and efficient ray tracing.

Dependencies:
-------------
- NumPy: For numerical operations.
- SciPy: For optimization routines.
- KrakenOS (Kos): Optical simulation environment.
- MOS_Class: Custom module with physically grounded MF and RMS evaluators.
- MOS_equation: Custom module for ray tracing and RMS radius calculation.
"""

# ===============================
#      Library Imports
# ===============================
import time
import numpy as np
import pkg_resources
import scipy

# Import KrakenOS and custom modules
from utils import MOS_Class as MOSCL
from utils import MOS_equation as Meq

# Check if KrakenOS is installed, otherwise append its path
required = {'KrakenOS'}
installed = {pkg.key for pkg in pkg_resources.working_set}
missing = required - installed

if missing:
    print("KrakenOS not installed")
    import sys
    sys.path.append("../..")

import KrakenOS as Kos



# ===============================
#    Constants Definition
# ===============================


per_red = 50.05329978714254      
Prx_data = MOSCL.Paraxial_Cal(per_red)
Target_EFFL = Prx_data.EFFL_Tr

# First order    
d_lens1 = 23.                        # Thickness of Lens L1
d_lens2 = 9.2                             # Thickness of Lens L2

F_Diameter = 105.                    # Firts Lens Diameter
S_Diameter = 88.0                         # Second Lens Diameter


# physically grounded merit function LS optimization
R_1 = 238.83472330021036
R_2 = -101987.90085807789
R_3 = 9998.050739254013
R_4 = 495.26909047059485

d_2 = 4674.5630261730985
d_3 = 27.792111342523572
d_4 = 185.83839257849027

# Calculate the field angle for the CCD
ccd_co = 16.1
Field_ccd = ccd_co / Prx_data.EFFL_Tr

# Initialize initial height 
h_i  = 1076.00


# ______________________________________#

# ======================================
#    Optical Element Initialization
# ======================================


# Object surface configuration

P_Obj = Kos.surf()
P_Obj.Rc = 0
P_Obj.Thickness = 1000 + Prx_data.d_1
P_Obj.Glass = "AIR"
P_Obj.Diameter = h_i * 2.0

# ______________________________________#

# ======================================
#  Telescope Initialization
# ======================================

# Mirror M1 configuration

M1 = Kos.surf()
M1.Rc = -11.176E+003
M1.Thickness = - Prx_data.d_1
M1.k = -1.070110000000E+000
M1.Glass = "MIRROR"
M1.Diameter = h_i * 2.0

# ______________________________________#

# Mirror M2 configuration

M2 = Kos.surf()
M2.Rc = -4.4300E+003
M2.Thickness = d_2
M2.k = -4.32070000000000E+000
M2.Glass = "MIRROR"
M2.Diameter = 3.175E+002 * 2.0


# ______________________________________#

# ======================================
#    Lens Initialization
# ======================================

# Lens L1a configuration

L1a = Kos.surf()
L1a.Rc = R_1
L1a.Thickness = d_lens1
L1a.Glass = "S-FPL51"
L1a.Diameter = F_Diameter

# ______________________________________#

# Lens L1b configuration

L1b = Kos.surf()
L1b.Rc = R_2
L1b.Thickness = d_3
L1b.Glass = "AIR"
L1b.Diameter = F_Diameter

# ______________________________________#

# Lens L2a configuration

L2a = Kos.surf()
L2a.Rc = R_3
L2a.Thickness = d_lens2
L2a.Glass = "F2HT"
L2a.Diameter = S_Diameter + 0.1*S_Diameter

# ______________________________________#

# Lens L2b configuration

L2b = Kos.surf()
L2b.Rc = R_4
L2b.Thickness = d_4
L2b.Glass = "AIR"
L2b.Diameter = S_Diameter + 0.1*S_Diameter

# ======================================
#  Image Plane Initialization
# ======================================

P_Ima = Kos.surf()
P_Ima.Rc = 0.0
P_Ima.Thickness = 0.0
P_Ima.Glass = "AIR"
P_Ima.Diameter = 100.0
P_Ima.Name = "Plano imagen"

#______________________________________#

# ======================================
#  System Configuration Setup
# ======================================

A = [P_Obj, M1, M2, L1a, L1b, L2a, L2b, P_Ima]

config_1 = Kos.Setup()

# ======================================
#  System Initialization
# ======================================


# Initialize the optical system with the defined elements and configuration
Telescope_f85_FR = Kos.system(A, config_1)
# Create a RayKeeper instance for storing and accessing ray tracing results.
Rays = Kos.raykeeper(Telescope_f85_FR)


# ______________________________________#

# ======================================
#  Pupil Calculation Setup
# ======================================

# Define the parameters for the pupil calculation:
W = 0.43032015             # Reference wavelength in micrometers
sup = 1                    # Number of the surface representing the opening of the system
AperType = "EPD"           # AperType sets the aperture ("STOP") or entrance pupil diameter ("EPD").
AperVal = M1.Diameter      # Diameter of the entrance pupil

# Initialize the Pupil calculation for the system
Pup = Kos.PupilCalc(Telescope_f85_FR , sup, W, AperType, AperVal)

# Configure the sampling for the pupil and its field representation:
Pup.Samp = 7              # Integer number for pupil ray sampling
Pup.FieldType = "angle"   # Field type, this in terms of object height and distance from the plane.
Pup.FieldX = np.rad2deg(Field_ccd) # Field value in degrees on the X-axis

# Display surface radii

print("\nSurface Radii Configuration and working distance:")
print(f"R1: {R_1:.2f} mm")
print(f"R2: {R_2:.2f} mm")
print(f"R3: {R_3:.2f} mm")
print(f"R4: {R_4:.2f} mm")
print(f"Working distance (d4): {d_4:.2f} mm")
print(" ")


# Third Order EFFL
EFFL_PGMF_LSO = Telescope_f85_FR.EFFL 


# Initializes the optical system for aberration analysis
InfSystem = [Telescope_f85_FR, Rays, Pup]

# ======================================
#  Gaussian Quadrature Sampling Setup 
# ======================================

# Instantiate Gaussian Quadrature objects for each wavelength
gqa = MOSCL.Gaussian_Quadrature(InfSystem, 0.35)
gqb = MOSCL.Gaussian_Quadrature(InfSystem, W)
gqc = MOSCL.Gaussian_Quadrature(InfSystem, 0.55)

# Extract coordinates
xa, ya, za, la, ma, na = gqa.Coordinates_GQ(n_nodes=3, n_arms=6, fieldx=0.0, fieldy=0.0, resp=0)
xb, yb, zb, lb, mb, nb = gqb.Coordinates_GQ(n_nodes=3, n_arms=6, fieldx=0.0, fieldy=0.0, resp=0)
xc, yc, zc, lc, mc, nc = gqc.Coordinates_GQ(n_nodes=3, n_arms=6, fieldx=0.0, fieldy=0.0, resp=0)

# Concatenate all results
all_points_x = np.concatenate((xa, xb, xc))
all_points_y = np.concatenate((ya, yb, yc))
all_points_z = np.concatenate((za, zb, zc))
all_points_l = np.concatenate((la, lb, lc))
all_points_m = np.concatenate((ma, mb, mc))
all_points_n = np.concatenate((na, nb, nc))

# Compute RMS using deltaZ correction
deltaZRMS = Meq.R_RMS_delta(all_points_z, all_points_l, all_points_m, all_points_n,
                      all_points_x, all_points_y)


cen_h, cen_k = Meq.calculate_geometrical_center(all_points_x, all_points_y)
coor_res, G_R, R_R = Meq.calculate_radius(all_points_x, all_points_y, cen_h, cen_k)


# Display initial spot size performance
print("======================================")
print("Initial RMS radius from Gaussian quadrature sampling:")
print(f"RMS (using deltaZ correction): {deltaZRMS:.2f}")
print(f"RMS radius: {R_R*1000:.2f} µm")
print(f"Effective Focal Length: {EFFL_PGMF_LSO:.2f}")
print("======================================")
print('')


# ===============================
#    RMS Optimization Process
# ===============================
print("\nStarting RMS Optimization...")
start_time = time.time()

# Initialize optimization function using Gaussian Quadrature and RMS-based metric
MyFun = MOSCL.Function2Optimize(InfSystem, W)
MyFun.Fx[1] = np.rad2deg(Field_ccd)  # Set field angle for RMS evaluation

# Extract current curvature values as starting point
R1, R2, R3, R4 = [L1a.Rc, L1b.Rc, L2a.Rc, L2b.Rc]

# Define bounds for optimization
LimInf = [-1e7] * 4
LimSup = [ 1e7] * 4
bounds = (LimInf, LimSup)

# Run least-squares optimization to minimize RMS radius
R = scipy.optimize.least_squares(MyFun.EFFL_3W, [R1, R2, R3, R4], bounds=bounds, verbose=0)
R1, R2, R3, R4 = R.x

# Update optical system with optimized curvature values
Telescope_f85_FR.SDT[3].Rc = R1
Telescope_f85_FR.SDT[4].Rc = R2
Telescope_f85_FR.SDT[5].Rc = R3
Telescope_f85_FR.SDT[6].Rc = R4
Telescope_f85_FR.SetData()
Telescope_f85_FR.SetSolid()

elapsed_time = time.time() - start_time

# ===============================
#    Gaussian Quadrature Sampling
# ===============================

# Update system configuration
InfSystem = [Telescope_f85_FR, Rays, Pup]

# Define function to avoid repetition of sampling + coordinate extraction
def sample_gaussian_rays(wavelengths, n_nodes=3, n_arms=6, fx=0.0, fy=0.0, resp=0):
    samples = [MOSCL.Gaussian_Quadrature(InfSystem, wl).Coordinates_GQ(n_nodes, n_arms, fx, fy, resp) for wl in wavelengths]
    return [np.concatenate(items) for items in zip(*samples)]

# Perform ray sampling at three wavelengths
wavelengths = [0.35, W, 0.55]
all_points_x, all_points_y, all_points_z, all_points_l, all_points_m, all_points_n = sample_gaussian_rays(wavelengths)

# Compute RMS using deltaZ correction
deltaZRMS = Meq.R_RMS_delta(all_points_z, all_points_l, all_points_m, all_points_n,
                      all_points_x, all_points_y)


# Apply best focus correction
system, deltaZ = Meq.BestFocus(all_points_x, all_points_y, all_points_z,
                               all_points_l, all_points_m, all_points_n, Telescope_f85_FR)


print("======================================")
print("Optimization Complete")
print("Elapsed time:", elapsed_time, "seconds")
print("Optimized Parameters:")
print(f"R1: {R1:.2f} mm")
print(f"R2: {R2:.2f} mm")
print(f"R3: {R3:.2f} mm")
print(f"R4: {R4:.2f} mm")
print(f"Working distance (d4): {d_4+deltaZ:.2f} mm")
print("======================================")


print("======================================")
print("Comparison with Initial Parameters:")
print(f"ΔR1: {np.abs(R1 - R_1):.2f} mm")
print(f"ΔR2: {np.abs(R2 - R_2):.2f} mm")
print(f"ΔR3: {np.abs(R3 - R_3):.2f} mm")
print(f"ΔR4: {np.abs(R4 - R_4):.2f} mm")
print(f"ΔWorking distance (d4): {np.abs(deltaZ):.2f} mm")
print("======================================")


# Re-sample after refocus
all_points_x, all_points_y, all_points_z, all_points_l, all_points_m, all_points_n = sample_gaussian_rays(wavelengths)


# Compute final geometrical center and RMS/GEO radii
cen_h, cen_k = Meq.calculate_geometrical_center(all_points_x, all_points_y)
coor_res, G_R, R_R = Meq.calculate_radius(all_points_x, all_points_y, cen_h, cen_k)

#Calculates the Airy disk radius and its coordinates for plotting.
    
#Parameters:
# W: # Reference wavelength
# Telescope_f85_FR.EFFL: Effective Focal Length
# M1.Diameter: Diameter of the primary mirror

Rairy = (1.22 * (0.55 / 1000) * Telescope_f85_FR.EFFL) / (M1.Diameter)
num_segmentos = 100
angulo = np.linspace(0, 2 * np.pi, num_segmentos + 1)
xairy = Rairy * np.cos(angulo)
yairy = Rairy * np.sin(angulo)


# ======================================
#    Plot Generation for All Fields
# ======================================

fields = [
    (0.0, 0.0, 'Field_0'),
    (np.rad2deg(Field_ccd), 0.0, 'Field_1'),
    (0.0, -np.rad2deg(Field_ccd), 'Field_2'),
    (0.035, -0.035, 'Field_3')
]

wavelengths = [0.35, W, 0.55]

List_Radius = [] 

for fx, fy, name in fields:
    
    # Configure the pattern sampling for the pupil and fields:
    Pup.Ptype = "hexapolar"     
    Pup.FieldX = fx
    Pup.FieldY = fy
    
    # Perform ray tracing for all wavelengths
    traced_data = Meq.trace_rays(Pup, wavelengths, Rays)
    
    # Extract the data
    (Xa, Ya, Za, La, Ma, Na), (Xb, Yb, Zb, Lb, Mb, Nb), (Xc, Yc, Zc, Lc, Mc, Nc) = traced_data
    
    # Plot the diagram
    geo_r, rms_r = Meq.plot_spot_diagram(Xa, Ya, Xb, Yb, Xc, Yc, xairy, yairy, name, "RMS_Method")
    
    # Append to the list of radii
    List_Radius.append((geo_r, rms_r))

# Convert to NumPy array for easy manipulation
List_Radius = np.array(List_Radius)

# Geo_radius in mm 
GEO_R = List_Radius[:, 0] * 1000
GEO_R_average = np.average(GEO_R)

# RMS_radius in mm 
RMS_R = List_Radius[:, 1] * 1000
RMS_R_average = np.average(RMS_R)


# ======================================
#    Plot for All the System
# ======================================

# ------------------------------------------------
# Field Configuration for Hexapolar Sampling
# ------------------------------------------------

field_configs = [
    (np.rad2deg(Field_ccd), 0.0, "hexapolar", 1),
    (0.0, 0.0, "hexapolar", 0),
    (-np.rad2deg(Field_ccd), 0.0, "hexapolar", 0)
]

traced_fields = []

for fx, fy, pattern, clean in field_configs:
    x, y, z, L, M, N = Meq.configure_and_trace(Pup, fx, fy, pattern, W, clean)
    Kos.TraceLoop(x * 0.99, y * 0.99, z * 0.99, 
                  L * 0.99, M * 0.99, N * 0.99, 
                  W, Rays, clean)
    traced_fields.append((x, y, z, L, M, N))

# ------------------------------------------------
# Display the 2D Optical System Layout
# ------------------------------------------------

Kos.display2d(Telescope_f85_FR, Rays, 1, arrow=0)

# Reinitialize the Pupil calculation
W = 0.43032015 
Pup = Kos.PupilCalc(Telescope_f85_FR, sup, W, AperType, AperVal)

# ======================================
#  Display Final Results
# ======================================

print("======================================")
print("\nFinal RMS analisys Analysis:")
print("Gaussian quadrature Analysis")
print(f"RMS radius (using deltaZ correction): {deltaZRMS:.2f}")
print(f"Final RMS radius after refocus: {R_R*1000:.2f} µm")
print("Hexapolar array Analysis")
print(f"Effective Focal Length: {Telescope_f85_FR.EFFL:.2f} mm")
print(f"EFFL deviation to Target: {np.abs(Telescope_f85_FR.EFFL - Target_EFFL):.2f} mm")
print(f"Airy disk radius: {Rairy * 1000:.2f}")
print(f"Average GEO radius: {GEO_R_average:.2f}")
print(f"Average RMS radius: {RMS_R_average:.2f}")
print(f"Airy's disk comparison with GEO and RMS: "
      f"GEO/Airy = {(GEO_R_average / (Rairy * 1000)):.2f}, "
      f"RMS/Airy = {(RMS_R_average / (Rairy * 1000)):.2f}")
print("======================================")
print('')