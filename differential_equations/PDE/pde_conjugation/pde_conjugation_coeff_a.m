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

% interpolate function value at centroids
nt = size(t,2);
uintrp = pdeintrp(p,t,u); % size N x nt

% function handles
% NOTE: need to choose one of the two growth functions
growth_malthusian = @(n) growth_rate_malthusian.*n;
growth_mm = @(n) growth_rate_mm_alpha.*n./(1 + growth_rate_mm_beta.*n);

% http://www.mathworks.com/help/pde/ug/multidimensional-coefficients.html
% parabolic system - 6 states
a = zeros(6, nt);  % 6 rows correspond to each state
for pt = 1:nt
    a(:,pt) = [conjugation_rate.*uintrp(2,pt) - growth_malthusian(uintrp(6,pt));                   % u1 - D
               conjugation_rate.*(uintrp(1,pt) + uintrp(3,pt)) - growth_malthusian(uintrp(6,pt));  % u2 - R 
               conjugation_rate.*uintrp(2,pt) - growth_malthusian(uintrp(6,pt));                   % u3 - T
               donor_return_rate - growth_malthusian(uintrp(6,pt));                                % u4 - Dr
               transconjugant_return_rate - growth_malthusian(uintrp(6,pt));                      % u5 - Tr
               sum(uintrp(1:5,pt)).*growth_rate_malthusian];                                         % u6 - n
end

end
