function [u, model, times] = pde_conjugation_solve(t0,t1,timesteps)
% Args:
%     t0,t1      -- [scalars] start and end times
%     timesteps  -- [scalar] number of steps between t0 and t1
% Returns:
%     u          -- [matrix] system solution (state evolution over time)
%     model      -- [object] contains BCs and geometry
%     tlist      -- [vector] times

% STATES
% u1 - D    donors
% u2 - Dr   refractory donors; just conjugated
% u3 - R    recipients
% u4 - Tr   refractory transconjugants; just received plasmid
% u5 - T    transconjugants
% u6 - N    nutrients


% =======================================================================
% System Setup
% =======================================================================

% number of PDEs in the system
N = 6;

% geometry
model = createpde(N);
geometryFromEdges(model,@squareg);
generateMesh(model,'Hmax',0.1);
p = model.Mesh.Nodes;
np = size(p,2);

% TODO FIX
% boundary conditions
% dirichlet (constant value)
%applyBoundaryCondition(model,'Edge',1:model.Geometry.NumEdges,'u',zeros(N,1));
% neumann (constant flux)
applyBoundaryCondition(model,'Edge',1:model.Geometry.NumEdges,'q',zeros(N,N),'g',zeros(N,1));

% TODO FIX
% intial conditions
u0_bacteria = zeros(np,1);
ix_bacteria = find(sqrt(p(1,:).^2 + p(2,:).^2) < 0.4);  % circle with value 1
u0_bacteria(ix_bacteria) = ones(size(ix_bacteria));
u0_nutrients = ones(np,1);
%ix_nutrients = find(sqrt(p(1,:).^2 + p(2,:).^2) < 0.8);  % circle with value 1
%u0_nutrients(ix_nutrients) = ones(size(ix_nutrients));
u0 = [u0_bacteria; u0_nutrients]; 

% pde parameters (system of equations)
% http://www.mathworks.com/help/pde/ug/multidimensional-coefficients.html
% parabolic system with 6 states
c = @pde_conjugation_coeff_c;    % interpreted as N^2 x  1
a = @pde_conjugation_coeff_a;    % interpreted as  N  x  N
f = zeros(N,1);                  % interpreted as  N  x  1
d = @pde_conjugation_coeff_d;    % interpreted as  N  x  N


% =======================================================================
% System Evaluation
% =======================================================================

% solver parameters
tlist = linspace(t0,t1,timesteps);

% solve
u = parabolic(u0, tlist, model, c, a, f, d);

end
