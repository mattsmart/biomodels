function [] = pde_visualizer(u, model, tlist, state_id)
% takes any pde u (soln), model, tlist and visualizes accordingly
% colormap choices: 'jet', 'cool', 'gray', 'bone', 'copper'
% can also specify contour levels 
% might be useful to have average value at a timepoint labelled

% extract pde solution parameters
np = size(model.Mesh.Nodes, 2);
N = size(u,1)/np;
assert(size(state_id,1) == N)

% choose timepoints for plotting
% DEFAULT - 4 timepoints, t0, ta, tb, t1
timesteps = size(u, 2);
if timesteps > 4
    m = mod(timesteps,3);
    T = (timesteps - m) / 3;    
    ta = 1 + T;
    tb = 1 + 2*T;
    timepoints = [1, ta, tb, timesteps];
else
    timepoints = 1:timesteps;
end

% plot each state
figure
step = 0;
for tt = timepoints
    for state = 1:N
        subplot(length(timepoints), N, step*N + state);
        pdeplot(model, 'xydata', u((state-1)*np+1:state*np, tt), 'colormap', 'gray');
        title(['State ' state_id(state,:) ' Time ' num2str(tlist(tt)) 'h']);
    end
    step = step + 1;
end

end
