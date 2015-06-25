function state_totals = pde_value_plotter(u, model, tlist, state_id)
% takes any pde u (soln), model, tlist and plots total state value over time

% extract pde solution parameters
timesteps = size(u, 2);
np = size(model.Mesh.Nodes, 2);
N = size(u,1)/np;

assert(size(state_id,1) == N)

% get state totals
% state_totals = zeros(N+1,timesteps);
% for state = 1:N
%     for tt = 1:timesteps
%         state_totals(state,tt) = sum(u((state-1)*np+1:state*np,tt));
%     end
% end
state_totals = pde_value_integrate(u, model, tlist);

% add column for total bacteria (general)
state_id = char(state_id, 'N');
state_totals(N+1,:) = sum(state_totals(1:N-1,:));  % bacteria sum

% plot sums
%figure
plot(tlist,state_totals')
legend(state_id)
title(['State values over time'])

end
