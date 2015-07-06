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
% Solver Options
% =======================================================================
hmax_mesh = 0.05;  % max size of a triangle edge in the mesh (larger is more coarse)
flag_ic_donor_solid = true;  % default true; use solid disk instead of sparse disk
flag_ic_recipient_solid = true;  % default true; use solid disk instead of sparse disk
flag_bc_neumann = true;  % default true; BCs are neumann (0 flux) instead of dirichlet
flag_system_nonlinear_diffusion = true;  % default true; use nonlinear diffusion instead of linear
flag_system_monod_growth = true;  % default true; use monod growth instead of linear
flag_system_nonnegative = false;  % default true; force state variable nonnegative in the eqns


% =======================================================================
% System Setup
% =======================================================================
% (0) Number of PDEs in the system
% (1) Geometry
% (2) Boundary Conditions
% (3) Intial Conditions
% (4) PDE System Coefficients
% =======================================================================

% (0) Number of PDEs in the system
N = 6;

% (1) Geometry
model = createpde(N);
geometryFromEdges(model,@squareg);
generateMesh(model,'Hmax',hmax_mesh);
p = model.Mesh.Nodes;
np = size(p,2);

% (2) Boundary Conditions
if flag_bc_neumann  % neumann (constant flux)
    applyBoundaryCondition(model,'Edge',1:model.Geometry.NumEdges,'q',zeros(N,N),'g',zeros(N,1));
else  % dirichlet (constant value)
    applyBoundaryCondition(model,'Edge',1:model.Geometry.NumEdges,'u',zeros(N,1));
end


% (3) Intial Conditions
% non-zero IC values
D0 = 1.0;  % [g/m^2] average donor concentration in IC region
R0 = 1.0;  % [g/m^2] average recipient concentration in IC region
n0 = 1.0;  % [g/m^3] = [1000g/l] average nutrient concentration in IC region
% initilize ICs
u0_D  = zeros(np,1);
u0_R  = zeros(np,1);
u0_T  = zeros(np,1);
u0_Dr = zeros(np,1);
u0_Tr = zeros(np,1);
u0_n  = n0 * ones(np,1);
% IC donor
ix_D = find(sqrt(p(1,:).^2 + p(2,:).^2) < 0.2);
if ~flag_ic_donor_solid
    ix_D = random_subsample(ix_D, 0.5);  % randomly subsample the disk
end
u0_D(ix_D) = D0 * ones(size(ix_D));  % set IC D0
% IC recipient
ix_R = find(sqrt(p(1,:).^2 + p(2,:).^2) < 0.2);
if ~flag_ic_recipient_solid
    ix_R = random_subsample(ix_R, 0.5);  % randomly subsample the disk
end
u0_R(ix_R) = R0 * ones(size(ix_R));  % set disk value R0
% finalize ICs
u0 = [u0_D; u0_R; u0_T; u0_Dr; u0_Tr; u0_n]; 

% (4) PDE System Coefficients
% http://www.mathworks.com/help/pde/ug/multidimensional-coefficients.html
% parabolic system with 6 states
[c, a, f, d] = pde_conjugation_system(flag_system_nonlinear_diffusion, flag_system_monod_growth,flag_system_nonnegative);


% =======================================================================
% System Evaluation
% =======================================================================

% solver parameters
tlist = linspace(t0,t1,timesteps);

% solve
u = parabolic(u0, tlist, model, c, a, f, d);


end
