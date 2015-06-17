% simulation parameters
t0 = 0; 
t1= 1; 
timesteps = 10;

[u, model, tlist] = pde_growth_solve(t0, t1, timesteps);
pde_visualizer(u, model, tlist);

% TODO 
% fix problem of nutrient loss not following bacteria
% check BCs and ICs
% make visualizer bar static axis
% diminesion of c and how to define it arent clear -- fix using defn 1 http://www.mathworks.com/help/pde/ug/c.html
% verify symmetry after picking reasonable constants and functions
