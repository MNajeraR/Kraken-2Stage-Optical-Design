import _222_path
import KrakenOS as Kos
import numpy as np

# ======================================
#    Optical Element Initialization
# ======================================

# Import KrakenOS and custom modules
from utils import (run_spots_for_fields, Prin_Plane,
                   plot_all_EE_for_fields, airy_data, 
                   configure_pupil_and_ab, seidel_terms) 



# Principal plane calculation
H_1, H_2 = Prin_Plane(1.5050514774681767, 86.85057940198467, -640.5824088487474, 30.)
H_3, H_4 = Prin_Plane(1.6447434449603227, -172.97479690009752, 220.44109983675503, 8.5)

# Object surface configuration

P_Obj = Kos.surf()
P_Obj.Rc = 0
P_Obj.Thickness = 1000 + 4052.571043
P_Obj.Glass = "AIR"
P_Obj.Diameter = 1076.0 * 2.0

# ______________________________________#

# ======================================
#  Telescope Initialization
# ======================================

# Mirror M1 configuration

M1 = Kos.surf()
M1.Rc = -11.176E+003
M1.Thickness = - 4052.571043
M1.k = -1.070110000000E+000
M1.Glass = "MIRROR"
M1.Diameter = 1076.0 * 2.0


# ______________________________________#

# Mirror M2 configuration

M2 = Kos.surf()
M2.Rc = -4.4300E+003
M2.Thickness = 4656.771691
M2.k = -4.32070000000000E+000 
M2.Glass = "MIRROR"
M2.Diameter = 3.175E+002 * 2.0

# ______________________________________#

# ======================================
#    Lens Initialization
# ======================================

# Lens L1a configuration

L1a = Kos.surf()
L1a.Rc = 86.85057940
L1a.Thickness = 30.
L1a.Glass = 'M-FCD1'
L1a.Diameter = 105.0

# ______________________________________#

# Lens L1b configuration

L1b = Kos.surf()
L1b.Rc = -640.582409
L1b.Thickness = 28.72761486
L1b.Glass = "AIR"
L1b.Diameter = 105.0

# ______________________________________#


# Lens L2a configuration

L2a = Kos.surf()
L2a.Rc = -172.974797
L2a.Thickness = 8.5
L2a.Glass = 'F11'
L2a.Diameter = 99.0

# ______________________________________#

# Lens L2b configuration

L2b = Kos.surf()
L2b.Rc = 220.4410998
L2b.Thickness = 90.78470159
L2b.Diameter = 99.0

# ______________________________________#

# ======================================
#  Image Plane Initialization
# ======================================

P_Ima = Kos.surf()
P_Ima.Rc = 0.0
P_Ima.Thickness = 0.0
P_Ima.Glass = "AIR"
P_Ima.Diameter = 35.0
P_Ima.Name = "Plano imagen"

# ______________________________________#

# ======================================
#  System Configuration Setup
# ======================================

A = [P_Obj, M1, M2, L1a, L1b, L2a, L2b, P_Ima]

config_1 = Kos.Setup()


# ======================================
#  System Initialization
# ======================================

# Initialize the optical system with the defined elements and configuration
Telescope_FR = Kos.system(A, config_1)

# Create a RayKeeper instance for storing and accessing ray tracing results.
Rays = Kos.raykeeper(Telescope_FR)


# ======================================
#  Pupil Calculation Setup (Recalculation)
# ======================================

# Define the parameters for the pupil calculation:
W = 0.43032015       # Reference wavelength in micrometers
# Define the wavelength range:
RW = [0.35, W, 0.5499996] 
sup = 1              # Number of the surface representing the opening of the system
AperType = "EPD"     # AperType sets the aperture ("STOP") or entrance pupil diameter ("EPD").
AperVal = M1.Diameter # Diameter of the entrance pupil
Field_ccd = 0.0012325797337758514
Pup = Kos.PupilCalc(Telescope_FR, sup, W, AperType, AperVal)
Pup.Samp = 7         
Pup.FieldType = "angle" 
Pup.FieldX = np.rad2deg(Field_ccd)

# ======================================
#    Plot Generation for All Fields
# ======================================

fields = [
    (0.0, 0.0, 'Field_0'),
    (np.rad2deg(Field_ccd), 0.0, 'Field_1'),
    (0.0, -np.rad2deg(Field_ccd), 'Field_2'),
    (0.035, -0.035, 'Field_3')
]


# Perform a paraxial analysis of the system at the specified wavelength (W).
Prx = Telescope_FR.Parax(W)

# Extract the list of refractive indices for each surface in the system.
N = Prx[11]

# Assign the refractive indices of the two main lenses for reference.
n_l1 = N[3]          # Refractive index of the first lens.
n_l2 = N[5]          # Refractive index of the second lens.

# Extract the paraxial matrix (ABCD Matrix) of the optical system.
M_System = np.array(Prx[0]) 

# From the paraxial matrix, extract the D element.
MS_d = M_System[1][1] 

# ======================================
#  Pupil Calculation Setup and Seidel Aberration Calculation
# ======================================

sup = 1              # Number of the surface representing the opening of the system
AperType = "EPD"     # AperType sets the aperture ("STOP") or entrance pupil diameter ("EPD").
AperVal = M1.Diameter # Diameter of the entrance pupil
W = 0.43032015 
# Pupil setup and Seidel computation
Pup, AB = configure_pupil_and_ab(Telescope_FR, Kos, sup, RW, AperType,
                                 AperVal, Field_ccd, samp=7)

# Extract individual aberration dimensionless terms
Sph, Coma, Ast, CLon = seidel_terms(AB, W)

#Calculates the Airy disk radius and its coordinates for plotting.
Rairy, xairy, yairy = airy_data(Telescope_FR, W, AperVal)
    

name_save = f"Example_{L1a.Glass}_{L2a.Glass}"

List_Radius, meta = run_spots_for_fields(
    Pup, Rays, Field_ccd, RW,
    xairy, yairy, name_save=name_save, ptype="hexapolar",
    save=False, show = False, show_geo_circle=False,
    show_rms_circle=False, lock_box_across_fields=True,    
    box_include_airy=False, save_dir = 'Images\SPT_Diagrams_New')


EE_Example_information = plot_all_EE_for_fields(
                            Pup, Rays, Field_ccd, RW, save_dir='Images\EE_Diagrams_New',
                            show_r50 = True, save = True,  show=True,
                            filename=f"EE_Example_{L1a.Glass}_{L2a.Glass}=.pdf",
                            airy_radius_um=Rairy*1000.,
                            multiply_by_diff_limit=True
                            )   

print("======================================")
print("Initial Seidel Aberrations:")
print(f"Spherical Aberration: {Sph:.2f}")
print(f"Coma: {Coma:.2f}")
print(f"Astigmatism: {Ast:.2f}")
print(f"Longitudinal Chromatic Aberration: {CLon:.2f}")
print(f"Matrix Element (D) from Paraxial Analysis: {MS_d:.2f}")
print(f"Effective Focal Length: {Telescope_FR.EFFL:.2f} mm")
print("======================================")
print('')

