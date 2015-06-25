% simulation parameters (hours)
t0 = 0;
t1 = 1;
timesteps = 20;  % was 4

% solve
[u, model, tlist] = pde_conjugation_solve(t0, t1, timesteps);

% visualization
state_id = char('D','R','T','Dr','Tr','n');
pde_visualizer(u, model, tlist, state_id);
pde_value_plotter(u, model, tlist, state_id);
