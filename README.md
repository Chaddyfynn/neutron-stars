# Compact Stars Numerical Calculator
This GIT contains the python code to numerically evaluate the equations of hydrostatic equilibrium for compact stars in Newtonian gravity and general relativity. There are five models available to use: Newtonian Polytropic, Newtonian Fermi Gas of Neutrons, TOV Polytropic, TOV Fermi Gas of Neutrons, and TOV Fermi Gas of Neutrons Protons and Electrons. 

## Main File //  main.py

The main.py file should be where the simulations are run from, and unless changes need to be made to the graphs, or the fundamental workings of the models, all changes necessary to simulate different stars should be made here. The file contains a number of initial global variables that can be used to set the bounds of integration/solving the ODE.

### Global Variables / System Settings

The first section of global variables defines the boundary condition for the ODE and the range over which the ODE should be solved. It does not directly affect the physics of the star per se, and R_F is the only variable that should need changing. Since the boundary conditions for the equations of hydrostatic equilibrium are only known at R = R_0 = 0, R_0 should be as close to 0 as possible without producing runtime or div0 errors. R_F should be greater than the expected total radius for the star so that there is enough data to conclude where the 'edge' of the star is, but not so much that the computation time is massive. R_F can be set using trial and error until an acceptable range of data is attained.

```python
# Numerical Methods / Calculator Settings
R_0 = 0.0001  # Initial Condition Radius, m
R_F = 50_000  # Final Radius, m
```

The second section is a continuation of the first section (setting up the ODE and its BCs), but is focused more on the physics of the star. MIN_PRESSURE and MAX_PRESSURE define the range of central pressures over which iterations of the ODE should be solved for.  When solving for one central pressure only, MIN_PRESSURE must be used to define the central pressure with MAX_PRESSURE being obsolete. Similarly, NUM_STEPS establishes the number of iterations between and including MIN_PRESSURE and MAX_PRESSURE that the ODE should be solved for. In the case that the ODE is only being solved for one central pressure, NUM_STEPS is obsolete. LOGARITHMIC is a boolean variable responsible for choosing whether the central pressures the ODEs are solved for are spaced linearly (False) or logarithmically (True). When True, the central pressures will be spaced logarithmically with a base value of 10.

```python
# System Settings
MIN_PRESSURE = 1e30  # Minimum Central Pressure, Pa
MAX_PRESSURE = 1e32  # Maximum Central Pressure, Pa
NUM_STEPS = 100  # Number of Iterations (Plot Points on Graph)
LOGARITHMIC = True  # Plot and Produce Points Logarithmically? (Boolean)
```

The third section focuses on how the star's radius and mass are calculated from the solved ODEs. The exact meaning of TOLERANCE changes from method to method, but it loosely describes how selective the algorithm must be to choose a max radius and mass. METHOD is currently not used, but will be used in the future to make it easier to select which radii-finding method is to be used. Expected options are: 

"normal": (radius and mass chosen when pressure drops below TOLERANCE * pressure range in output data)

"min": (choose the radius and mass that corresponds with the minimum pressure in the output_data)

"first": (choose the radius and mass when the gradient between the current and last point drops below TOLERANCE * average gradient of points)

"second": (choose the radius and mass when the gradient of the gradient between the current point and point before last drops below TOLERANCE * average of points)

"saturation": (choose the radius and mass when the change in the value of the mass does not exceed TOLERANCE for three successive points)

```python
# Radius Root Finding Tolerance (Changes Meaning According to Root Finding Algorithm)
TOLERANCE = 0.001
METHOD = "normal" # Options are "normal", "min", "first", "second", "saturation"
```

The final section focuses on how the data is generated, visualised, and saved. FILENAME is a string which contains the desired name for the output data text file and graph. The actual filename will vary depending on what is already saved in the 'saves' folder. PLOT_TIME is a legacy variable that was once used to specify whether a second graph should be generated containing the function evaluation times for different iterations of solving the ODE. This was used to test the code and check its efficiency, but since the code has been rewritten from scratch, its functionality has not been reincluded. FULL_COMPUTATION is a boolean variable that established whether or not the ODEs should be solved for one central pressure (False) or over the range of central pressures outlined in the second section (True). PLOT_INDIVIDUAL chooses whether or not to plot a graph of every intermediate solution for the ODE at each central pressure if FULL_COMPUTATION is True. If FULL_COMPUTATION is False, then PLOT_INDIVIDUAL is useless. PLOT_INDIVIDUAL is very useful for seeing whether or not TOLERANCE is set correctly - though this information is also printed to the console. CROP is a variable that selects the left crop on the x-axis for the FULL_COMPUTATION = True graph. Finally, 

```python
# Save and Graph Settings
FILENAME = "Efficient_Star_Tests"  # Graph and Text File Desired Name
PLOT_TIME = False  # Plot Function Evaluation Times vs Pressure? (Boolean) (legacy)
# Compute for a range of central pressures (True), or one (False)
# Compute over central pressure range? (0 to generate e_dens array)
FULL_COMPUTATION = False
PLOT_INDIVIDUAL = False  # Create graphs for each central pressure (False)
CROP = 0  # Left Crop for Full computation, 5e23 for rel
METADATA = [R_0, MIN_PRESSURE, MAX_PRESSURE,
            NUM_STEPS]  # Desired save metadata
```
