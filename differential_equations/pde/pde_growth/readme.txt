Growth PDEs 

Structure
- _solve.m: defines the system and calls the solver
- _coeff_*.m: contain the c, a, and d coefficient specifications that define the system of PDEs 
- script_growth.m: calls the solver and passes the results to pde_visualizer.m

Things to select for a simulation
- BC choice (constant value, constant flux, periodic, infinite or local bounds)
- IC shapes and values (e.g. disk with value u0 interior, 0 elsewhere)
- growth function (in _coeff_a) -- linear or MM
- variable values (e.g. diffusion rate, growth rate)
- simulation duration (t0, t1, number of timesteps)

TODO 
- check BCs and IC values to match mimura paper
- consider infinite (0 at inf) or periodic BCs instead of neumann
- make visualizer bar 'static axis' option
- fix nutrient plot behaviour
