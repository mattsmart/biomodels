% simulation parameters
t0 = 0; 
t1= 6; 
timesteps = 4;

[u, model, tlist] = pde_conjugation_solve(t0, t1, timesteps);
pde_visualizer(u, model, tlist);
