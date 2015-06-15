function [u, model, tlist] = pde_solve_growth(t0, t1, timesteps)
% Args:
%     none
% Returns:
%     u      -- [matrix] bacteria / nutrient concentration solution
%     model  -- [object] contains BCs and geometry
%     tlist  -- [vector] times


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

%TODO FIX
a = char('-1*heaviside(u(1)-0.75).*2.*u(2)', 'heaviside(u(1)-0.75).*2.*u(2)')
%c = [diffusion_rate; diffusion_rate; 1.0; 1.0];
c = 1
d = 1

% geometry
model = createpde(N);
geometryFromEdges(model,@squareg);
generateMesh(model,'Hmax',0.1);
p = model.Mesh.Nodes;
np = size(p,2);

% boundary conditions
% dirichlet (constant value)
%applyBoundaryCondition(model,'Edge',1:model.Geometry.NumEdges,'u',zeros(N,1));
% neumann (constant flux)
applyBoundaryCondition(model,'Edge',1:model.Geometry.NumEdges,'q',zeros(N,N),'g',zeros(N,1));

% intial conditions
u0_bacteria = zeros(np,1);
ix_bacteria = find(sqrt(p(1,:).^2 + p(2,:).^2) < 0.4);  % circle with value 1
u0_bacteria(ix_bacteria) = ones(size(ix_bacteria));
u0_nutrients = ones(np,1);
%ix_nutrients = find(sqrt(p(1,:).^2 + p(2,:).^2) < 0.8);  % circle with value 1
%u0_nutrients(ix_nutrients) = ones(size(ix_nutrients));
u0 = [u0_bacteria; u0_nutrients]; 


% =======================================================================
% System Evaluation
% =======================================================================

% solver parameters
tlist = linspace(t0,t1,timesteps);

% solve
u = parabolic(u0, tlist, model, c, a, f, d);

end
