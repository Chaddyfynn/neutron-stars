# Compact Stars and Hydrostatic Equilibrium Calculator
This GIT contains the python code to numerically evaluate the equations of hydrostatic equilibrium for compact stars in Newtonian gravity and general relativity.

## Main File //  calculator.py

The calculator.py file cannot run as a standalone file but contains all the necessary functions to evaluate different differential equations.

Here are some notable functions and their intended usage.
You will need to import calculator.py into the model file to use it first. It is recommended to import as solve.

```python
import calculator as solve
```
### RK4
The RK4 function can solve a set of differential equations using the Runge-Kutta 4 method. This function and all other RK4 named functions in the calculator.py file were authored by the University of Manchester,
```python
# Solves set of ODES
solve.rk4(grad, time, state, step_size, n_steps)
# Returns
```

## Model Files // [model_name].py

