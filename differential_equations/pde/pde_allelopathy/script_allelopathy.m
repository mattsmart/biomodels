% simulation parameters (hours)
t0 = 0;
t1 = 4;
timesteps = 25;

% solve
[u, model, tlist] = pde_conjugation_solve(t0, t1, timesteps);

% plot settings
title_modifier = sprintf('_%dh', t1-t0);
plot_directory = '.\\plots';
if ~exist(plot_directory,'dir')
    mkdir(plot_directory)
end
state_id = char('D','R','T','Dr','Tr','n');
title_totals = sprintf('%s\\totals%s', plot_directory, title_modifier) ;
title_visualizer = sprintf('%s\\visualizer%s', plot_directory, title_modifier);

% plots
state_totals = pde_value_plotter(u, model, tlist, state_id, title_totals);
pde_visualizer(u, model, tlist, state_id, title_visualizer);
