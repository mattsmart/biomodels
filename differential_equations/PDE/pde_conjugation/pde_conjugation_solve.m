function [u, model, tlist] = pde_conjugation_solve(t0,t1,timesteps)
% Args:
%     t0,t1      -- [scalars] start and end times
%     timesteps  -- [scalar] number of steps between t0 and t1
% Returns:
%     u          -- [matrix] system solution (state evolution over time)
%     model      -- [object] contains BCs and geometry
%     tlist      -- [vector] times

% STATES
% u1 - D    donors
% u2 - R    recipients
% u3 - T    transconjugants
% u4 - Dr   refractory donors; just conjugated
% u5 - Tr   refractory transconjugants; just received plasmid
% u6 - n    nutrients


% =======================================================================
% System Setup
% =======================================================================

% number of PDEs in the system
N = 6;

% geometry
model = createpde(N);
geometryFromEdges(model,@squareg);
generateMesh(model,'Hmax',0.2);
p = model.Mesh.Nodes;
np = size(p,2);

% boundary conditions
% dirichlet (constant value)
%applyBoundaryCondition(model,'Edge',1:model.Geometry.NumEdges,'u',zeros(N,1));
% neumann (constant flux)
applyBoundaryCondition(model,'Edge',1:model.Geometry.NumEdges,'q',zeros(N,N),'g',zeros(N,1));

% intial conditions
D0 = 1e2;  % average donor concentration in IC region
R0 = 1e2;  % average recipient concentration in IC region
n0 = 1.0;  % average nutrient concentration in IC region
% initilize ICs
u0_D  = zeros(np,1);
u0_R  = zeros(np,1);
u0_T  = zeros(np,1);
u0_Dr = zeros(np,1);
u0_Tr = zeros(np,1);
u0_n  = n0 * ones(np,1);
% IC donor
ix_D = find(sqrt(p(1,:).^2 + p(2,:).^2) < 0.05);
u0_D(ix_D) = D0 * ones(size(ix_D));  % set disk value D0
% IC recipient
ix_R = find(sqrt(p(1,:).^2 + p(2,:).^2) < 0.2);
u0_R(ix_R) = R0 * ones(size(ix_R));  % set disk value R0
% finalize ICs
u0 = [u0_D; u0_R; u0_T; u0_Dr; u0_Tr; u0_n]; 

% pde parameters (system of equations)
% http://www.mathworks.com/help/pde/ug/multidimensional-coefficients.html
% parabolic system with 6 states
c = @pde_conjugation_coeff_c;    % interpreted as N^2 x  1  x  mesh points
a = 0;                           % interpreted as  N  x  N  x  mesh points
f = @pde_conjugation_coeff_f;    % interpreted as  N  x  1  x  mesh points
d = 1;                           % interpreted as  N  x  N  x  mesh points

% =======================================================================
% System Evaluation
% =======================================================================

% solver parameters
tlist = linspace(t0,t1,timesteps);

% solve
u = parabolic(u0, tlist, model, c, a, f, d);  % can try setting a and d to scalars

end
