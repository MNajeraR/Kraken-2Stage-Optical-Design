# -*- coding: utf-8 -*-
"""
Created on Mon Oct 13 11:40:31 2025

@author: MORGANRHAINAJERAROA
"""


import random
import numpy as np


#########################################################################################################
#########################################################################################################


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
#########################################################################################################

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
#########################################################################################################

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