function a = pde_growth_coeff_a(p,t,u,time)

% constants
nutrient_threshold = 0.25;     % "theta" (Mimura, 2000)
growth_rate_malthusian = 1.0;  % "alpha" (Mimura, 2000)
growth_rate_mm_alpha = 1.0;    % "MM alpha" (Mimura, 2000)
growth_rate_mm_beta = 1.0;     % "MM beta" (Mimura, 2000)
death_rate = 0.01;             % "mu" (Mimura, 2000)

% function handles
% NOTE: need to choose one of the two growth functions
growth_malthusian = @(n) growth_rate_malthusian.*n;
growth_mm = @(n) growth_rate_mm_alpha.*n./(1 + growth_rate_mm_beta.*n);
heaviside_growth_cutoff = @(b) b - nutrient_threshold > 0;

% http://www.mathworks.com/help/pde/ug/multidimensional-coefficients.html
% parabolic system - 2 states (bacteria u1 and nutrients u2)
a = [-heaviside_growth_cutoff(u(1)).*growth_malthusian(u(2)) + death_rate;
     heaviside_growth_cutoff(u(1)).*growth_malthusian(u(2))];

end
