function [] = pde_visualizer(u, model, tlist, state_id)
% takes any pde u (soln), model, tlist and visualizes accordingly
% colormap choices: 'jet', 'cool', 'gray', 'bone', 'copper'
% can also specify contour levels 
% might be useful to have average value at a timepoint labelled

% extract pde solution parameters
timesteps = size(u, 2);
np = size(model.Mesh.Nodes, 2);
N = size(u,1)/np;

assert(size(state_id,1) == N)

% plot each state
figure
for tt = 1:timesteps
    for state = 1:N
        subplot(timesteps,N,(tt-1)*N + state);
        pdeplot(model,'xydata',u((state-1)*np+1:state*np,tt),'colormap','gray');
        title(['State ' state_id(state,:) ' Time ' num2str(tlist(tt)) 'h']);
    end
end
