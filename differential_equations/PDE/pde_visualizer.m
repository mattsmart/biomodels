function [] = pde_visualizer(u, model, tlist)
% takes any pde u (soln), model, tlist and visualizes accordingly
% colormap choices: 'jet', 'cool'

% extract pde solution parameters
timesteps = size(u, 2);
np = size(model.Mesh.Nodes, 2);
N = size(u,1)/np;

% plot each state
figure
for tt = 1:timesteps
    for state = 1:N
        subplot(timesteps,N,(tt-1)*N + state)
        pdeplot(model,'xydata',u((state-1)*np+1:state*np,tt));
        title(['State ' num2str(state) ' Step ' num2str(tt)]);
    end
end
