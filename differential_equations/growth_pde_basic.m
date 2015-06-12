function [B, N, times] = growth_pde_basic()
% Args:
%     none
% Returns:
%     B      -- [matrix] bacteria concentration
%     N      -- [matrix] nutrient concentration
%     times  -- [vector] times


% =======================================================================
% System Setup
% =======================================================================

% number of PDEs in the system
N = 2;

% constants
nutrient_threshold = 1;      % "theta" (Mimura, 2000)
diffusion_rate = 1;          % "d_0" (Mimura, 2000)
growth_rate_malthusian = 1;  % "alpha" (Mimura, 2000)
growth_rate_mm_alpha = 1;    % "MM alpha" (Mimura, 2000)
growth_rate_mm_beta = 1;     % "MM beta" (Mimura, 2000)

% function handles
% NOTE: need to choose one of the two growth functions)
growth_malthusian = @(n) growth_rate_malthusian.*n;
growth_mm = @(n) growth_rate_mm_alpha.*n./(1 + growth_rate_mm_beta.*n);
heaviside_growth_cutoff = @(b) b - nutrient_threshold > 0;

% pde parameters
% http://www.mathworks.com/help/pde/ug/multidimensional-coefficients.html
% parabolic system - 2 states (bacteria u1 and nutrients u2)
a = char('-heaviside_growth_cutoff(u(1)).*growth_malthusian(u(2))',...
         'heaviside_growth_cutoff(u(1)).*growth_malthusian(u(2))')
c = [diffusion_rate, 0.0; 
     0.0, 1.0];
d = [1.0, 0.0; 
     0.0, 1.0];
f = [0.0; 0.0];

%a = char('heaviside(u(1)-1).*1.*u(2)./(1+1.*u(2))', '-heaviside(u(1)-1).*1.*u(2)./(1+1.*u(2))')
a = 1
c = 1
d = 1

% geometry
model = createpde(N);
geometryFromEdges(model,@squareg);
generateMesh(model,'Hmax',0.1);
p = model.Mesh.Nodes
np = size(p,2)

% boundary conditions
applyBoundaryCondition(model,'Edge',1:model.Geometry.NumEdges,'u',zeros(N,1));

% intial conditions
u0_bacteria = zeros(np,1);
u0_nutrients = zeros(np,1);
ix = find(sqrt(p(1,:).^2 + p(2,:).^2) < 0.4);  % circle with value 1
u0_nutrients(ix) = ones(size(ix));
u0 = [u0_bacteria; u0_nutrients]; 
%u0 = 0

% =======================================================================
% System Evaluation
% =======================================================================

% solver parameters
timesteps = 3;
tlist = linspace(0,1,timesteps);

% solve
u = parabolic(u0, tlist, model, c, a, f, d);


% REMOVE
B = model
N = u
times = p

end
%{
% ======================================================================
% Plotting
% ======================================================================
% NOTE need to have subplots for each solution based on np row slices
% for tt = 1:timesteps % number of timesteps
%     pdeplot(model,'xydata',u(:np,tt),'zdata',u(:np,tt),'colormap','jet')
%     axis([-1 1 -1/2 1/2 -1.5 1.5 -1.5 1.5]) % use fixed axis
%     title(['Step ' num2str(tt)])
%     view(-45,22)
%     drawnow
%     pause(.1)
% end
pdeplot(model,'xydata',u(:np,1));
axis equal
figure
pdeplot(model,'xydata',u(:np,timesteps))
axis equal

B = u(1:np,:);
N = u(np+1:np*N,:);
times = tlist;

end
%}