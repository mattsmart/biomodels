function state_totals = pde_value_plotter(u, model, tlist, state_id, plot_title)
% takes any pde u (soln), model, tlist and plots total state value over time

% extract pde solution parameters
np = size(model.Mesh.Nodes, 2);
N = size(u,1)/np;
assert(size(state_id,1) == N)

% get state totals through integration
state_totals = pde_value_integrate(u, model);

% add column for total bacteria (general)
state_id = char(state_id, 'N');
state_totals(N+1,:) = sum(state_totals(1:N-1,:));  % bacteria sum

% plot sums
fig = figure('visible','off');
for state = 1:(N+1)
    subplot(floor(N/3),floor(N/2)+1,state)
    plot(tlist,state_totals(state,:)')
    title(['Total ' state_id(state,:)])
end

% save figure
print(fig,plot_title,'-djpeg','-r0');

end
