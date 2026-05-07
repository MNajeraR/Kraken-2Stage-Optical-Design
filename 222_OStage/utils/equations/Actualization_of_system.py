

from ..equations.opt_ecuation import Prin_Plane
import numpy as np
from ..classes.Gaussian_Quadrature import Gaussian_Quadrature
from ..equations.ops import BestFocus

def apply_system_actualization(
    system,
    R1, R2, R3, R4,
    n_l1, d_lens1,
    n_l2, d_lens2,
    TO_data, d4,
    sup, AperType, AperVal, Field_ccd,
    W_ref,
    Rays, Kos, AB,
    n_nodes=3, n_arms=6, samp=7
):
    """
    Applies radii R1..R4 to the system, recalculates principal planes and thicknesses,
    reconfigures the pupil, samples rays (at three wavelengths),
    and performs BestFocus. Returns (updated_system, deltaZ).
    
    Requires the following functions/classes:
      - Prin_Plane(n, R1, R2, d): -> (H1, H2)
      - Kos.PupilCalc(system, sup, W, AperType, AperVal)
      - Gaussian_Quadrature(InfSystem, wl).Coordinates_GQ(n_nodes, n_arms, fx, fy, resp)
      - BestFocus(...)
      """
      
    # --- 1) Set radios
    system.SDT[3].Rc = R1
    system.SDT[4].Rc = R2
    system.SDT[5].Rc = R3
    system.SDT[6].Rc = R4

    # --- 2) Principal Planes (each lens)
    H1_a, H2_a = Prin_Plane(n_l1, R1, R2, d_lens1)
    H1_b, H2_b = Prin_Plane(n_l2, R3, R4, d_lens2)

    # --- 3) Update optical train thicknesses
    system.SDT[2].Thickness = TO_data.d_2 - H1_a
    system.SDT[4].Thickness = TO_data.d_3 - H2_a - H1_b
    system.SDT[6].Thickness = d4 - H2_b

    # --- 4) Apply changes to the optical system
    system.SetData()
    system.SetSolid()
    
    
     # --- 5) Pupil / Field configuration
    Pup = Kos.PupilCalc(system, sup, W_ref, AperType, AperVal)
    Pup.Samp = samp
    Pup.FieldType = "angle"
    Pup.FieldX = np.rad2deg(Field_ccd)

    # --- 6) Gaussian ray sampling at three wavelengths
    InfSystem = [system, Rays, Pup]

    def sample_gaussian_rays(wavelengths, n_nodes=3, n_arms=6, fx=0.0, fy=-np.rad2deg(Field_ccd), resp=0):
        samples = [Gaussian_Quadrature(InfSystem, wl).Coordinates_GQ(n_nodes, n_arms, fx, fy, resp)
                   for wl in wavelengths]
        return [np.concatenate(items) for items in zip(*samples)]

    wavelengths = [AB.Wf, W_ref, AB.Wc]
    (all_x, all_y, all_z,
     all_l, all_m, all_n) = sample_gaussian_rays(wavelengths, n_nodes=n_nodes, n_arms=n_arms)

    # --- 7) Best focus
    system_focused, deltaZ = BestFocus(all_x, all_y, all_z,
                                       all_l, all_m, all_n, system)
    deltaZ = - H2_b + deltaZ
    
    return system_focused, deltaZ