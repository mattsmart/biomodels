function [] = pde_visualizer(u, model, tlist)
% takes any pde u (soln), model, tlist and visualizes accordingly

% extract pde solution parameters
timesteps = size(u, 2);
np = size(model.Mesh.Nodes, 2);
N = mod(size(u,1), np);

% NOTE need to have subplots for each solution based on np row slices

% V1
% -------------------
for tt = 1:timesteps
    pdeplot(model,'xydata',u(1:np,tt))
    %axis([-1 1 -1/2 1/2 -1.5 1.5 -1.5 1.5]) % use fixed axis
    title(['Step ' num2str(tt)])
    %view(-45,22)  % ???????
    drawnow  % ???????
    pause(.1)
end
for tt = 1:timesteps
    pdeplot(model,'xydata',u(np+1:np*2,tt),'colormap','jet')
    %axis([-1 1 -1/2 1/2 -1.5 1.5 -1.5 1.5]) % use fixed axis
    title(['Step ' num2str(tt)])
    %view(-45,22)  % ???????
    drawnow  % ???????
    pause(.1)
end
% V2
% -------------------
% pdeplot(model,'xydata',u(:np,1));
% axis equal
% figure
% pdeplot(model,'xydata',u(:np,timesteps))
% axis equal

% end
% 'colormap', 'jet'