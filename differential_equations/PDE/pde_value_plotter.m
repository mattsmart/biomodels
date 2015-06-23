function [] = pde_visualizer(u, model, tlist, state_id)
% takes any pde u (soln), model, tlist and plots total state value over time

% extract pde solution parameters
timesteps = size(u, 2);
np = size(model.Mesh.Nodes, 2);
N = size(u,1)/np;

assert(size(state_id,1) == N)

% get state totals
state_totals = zeros(N,timesteps);
for state = 1:N
    for tt = 1:timesteps
        state_totals(state,tt) = sum(u((state-1)*np+1:state*np,tt));
    end
    %plot(timesteps,N,(tt-1)*N + state);
end
plot(state_totals)
legend(state_id)
title(['State values over time'])

end
