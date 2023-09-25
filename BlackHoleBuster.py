import numpy as np
import matplotlib.pyplot as plt

# Constants
G = 6.67430e-11  
c = 299792458  
M_sun = 1.989e30  

# Black hole mass
M = 10 * M_sun

# Maximum angular momentum
J_max = M**2 * G / c

# Initial and final angular momentum values
J_initial = 0.5 * J_max  
J_final = 1.5 * J_max  # Going beyond J_max for demonstration
delta_J = 0.01 * J_max  

# Angular momentum array
J_values = np.arange(J_initial, J_final, delta_J)

# Validity check for square root
valid_term = 1 - (J_values / J_max)**2
Re = np.where(valid_term >= 0, Rs * np.sqrt(valid_term), np.nan)
r_EH = np.where(valid_term >= 0, Rs * (1 + np.sqrt(valid_term)) / 2, np.nan)

# Angular velocity (omega)
omega_values = c * np.sqrt(G * M) / (2 * J_values)

# Energy associated with angular momentum change
delta_E_values = omega_values * delta_J

# Schwarzschild radius
Rs = 2 * G * M / c**2

# Plotting
plt.figure(figsize=(14, 8))

# Plot of Energy needed to reach J
plt.subplot(3, 1, 1)
plt.plot(J_values, delta_E_values, label='Energy to reach $J$')
plt.axvline(x=J_max, color='r', linestyle='--', label='Extremal Condition $J_{max}$')
plt.xlabel('Angular Momentum $(kg \, m^2/s)$')
plt.ylabel('Energy $(Joules)$')
plt.title('Energy Needed to Increase Angular Momentum of Black Hole')
plt.legend()
plt.grid(True)

# Plot of Schwarzschild radius and Ergosphere radius
plt.subplot(3, 1, 2)
plt.plot(J_values, Re, label='Ergosphere Radius $R_e$')
plt.plot(J_values, r_EH, label='Event Horizon Radius $r_{EH}$')
plt.axhline(y=Rs, color='g', linestyle='--', label='Schwarzschild Radius $R_s$')
plt.axvline(x=J_max, color='r', linestyle='--', label='Extremal Condition $J_{max}$')
plt.xlabel('Angular Momentum $(kg \, m^2/s)$')
plt.ylabel('Radius $(meters)$')
plt.title('Schwarzschild, Ergosphere, and Event Horizon Radii')
plt.legend()
plt.grid(True)

# Show the plots
plt.tight_layout()
plt.show()

print('''
#### Graph Explanations:

1. **Energy Needed to Increase Angular Momentum of Black Hole**:  
    - **What it Shows**: The graph displays the energy required to alter the angular momentum \( J \) of a black hole.
    - **Red Dashed Line**: Represents the extremal condition, \( J_{\text{max}} \), beyond which a black hole cannot gain more angular momentum according to general relativity.
    - **Significance**: Understanding the energy required to influence a black hole's angular momentum is vital for any theoretical attempts to manipulate these cosmic entities.

2. **Schwarzschild, Ergosphere, and Event Horizon Radii**:  
    - **What it Shows**: The graph illustrates how the radius of the ergosphere \( R_e \) and the event horizon \( r_{\text{EH}} \) vary with the black hole's angular momentum \( J \).
    - **Green Dashed Line**: Indicates the Schwarzschild radius \( R_s \), which is constant for a black hole of a given mass.
    - **Red Dashed Line**: Like in the first graph, represents the extremal condition \( J_{\text{max}} \).
    - **Blue Line**: Represents the radius of the event horizon \( r_{\text{EH}} \), which varies depending on the angular momentum of the black hole.
    - **Significance**: The size of the ergosphere has implications for the energy extraction processes around a black hole, such as the Penrose process. The event horizon radius provides insights into the boundary within which nothing can escape the black hole.
    - **Note**: The graph ceases to converge because of mathematical constraints related to square root calculations. Going beyond the maximum angular momentum (J(max)) would give rise to a "naked singularity." According to the cosmic censorship hypothesis, this is generally deemed unphysical. In simpler terms, singularities with infinite densities should be veiled by an event horizon, rendering them unobservable from the outside. If no event horizon is visible, it means you are already inside the black hole's event horizon, cbrwx.
''')
