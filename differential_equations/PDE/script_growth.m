[u, model, tlist] = pde_solve_growth(0, 1, 10);
pde_visualizer(u, model, tlist);


% TODO - fix problem of nutrient loss not following bacteria
% check BCs 
% check equations 