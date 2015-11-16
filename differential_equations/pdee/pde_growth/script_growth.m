% simulation parameters
t0 = 0; 
t1= 6; 
timesteps = 10;

[u, model, tlist] = pde_growth_solve(t0, t1, timesteps);
pde_visualizer(u, model, tlist);
