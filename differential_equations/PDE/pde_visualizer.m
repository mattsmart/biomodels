function [] = pde_visualizer(u, model, tlist)
% takes any pde u (soln), model, tlist and visualizes accordingly
% colormap choices: 'jet', 'cool'

% extract pde solution parameters
timesteps = size(u, 2);
np = size(model.Mesh.Nodes, 2);
N = size(u,1)/np;

for state = 1:N
    for tt = 1:timesteps
        state_idx_lower = (state-1)*np + 1;
        state_idx_upper = state*np;
        plot_title = ['State ' num2str(state) ' Step ' num2str(tt)];
        pdeplot(model,'xydata',u(state_idx_lower:state_idx_upper,tt));
        title(plot_title)
        drawnow
        pause(.1)
    end
end


% NOTE need to have subplots for each solution based on np row slices
% for tt = 1:timesteps
%     pdeplot(model,'xydata',u(1:np,tt))
%     title(['Step ' num2str(tt)])
%     drawnow
%     pause(.1)
% end
% 
% for tt = 1:timesteps
%     pdeplot(model,'xydata',u(np+1:np*2,tt),'colormap','jet')
%     title(['Step ' num2str(tt)])
%     drawnow
%     pause(.1)
% end
