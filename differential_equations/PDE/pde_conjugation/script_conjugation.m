% simulation parameters (hours)
t0 = 0;
t1 = 6;
timesteps = 25;

% solve
[u, model, tlist] = pde_conjugation_solve(t0, t1, timesteps);

% visualization
state_id = char('D','R','T','Dr','Tr','n');
pde_visualizer(u, model, tlist, state_id);
pde_value_plotter(u, model, tlist, state_id);
