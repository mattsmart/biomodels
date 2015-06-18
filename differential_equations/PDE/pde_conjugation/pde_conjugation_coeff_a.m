function a = pde_conjugation_coeff_a(p,t,u,time)

% STATES
% u1 - D    donors
% u2 - R    recipients
% u3 - T    transconjugants
% u4 - Dr   refractory donors; just conjugated
% u5 - Tr   refractory transconjugants; just received plasmid
% u6 - n    nutrients

% constants
nutrient_threshold = 0.25;         % "theta" (Mimura, 2000)
growth_rate_malthusian = 1.0;      % "alpha" (Mimura, 2000)
growth_rate_mm_alpha = 1.0;        % "MM alpha" (Mimura, 2000)
growth_rate_mm_beta = 1.0;         % "MM beta" (Mimura, 2000)
death_rate = 0.01;                 % "mu" (Mimura, 2000)
donor_return_rate = 0.5;           % "k_D"
transconjugant_return_rate = 0.5;  % "k_T"
conjugation_rate = 0.5;            % "gamma"

% function handles
% NOTE: need to choose one of the two growth functions
growth_malthusian = @(n) growth_rate_malthusian.*n;
growth_mm = @(n) growth_rate_mm_alpha.*n./(1 + growth_rate_mm_beta.*n);
heaviside_growth_cutoff = @(b) b - nutrient_threshold > 0;

% http://www.mathworks.com/help/pde/ug/multidimensional-coefficients.html
% parabolic system - 6 states
a = [conjugation_rate.*u(2) - growth_malthusian(u(6));              % u1 - D
     conjugation_rate.*(u(1) + u(3)) - growth_malthusian(u(6));     % u2 - R 
     conjugation_rate.*u(2) - growth_malthusian(u(6));              % u3 - T
     donor_return_rate - growth_malthusian(u(6));                   % u4 - Dr
     transconjugant_return_rate - growth_malthusian(u(6));          % u5 - Tr
     (u(1) + u(2) + u(3) + u(4) + u(5)).*growth_rate_malthusian];   % u6 - n

end
