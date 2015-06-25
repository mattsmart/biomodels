function f = pde_conjugation_coeff_f(p,t,u,time)

% STATES
% u1 - D    donors
% u2 - R    recipients
% u3 - T    transconjugants
% u4 - Dr   refractory donors; just conjugated
% u5 - Tr   refractory transconjugants; just received plasmid
% u6 - n    nutrients

% constants
growth_rate_malthusian = 1.0;      % "alpha" (Mimura, 2000)
growth_rate_mm_alpha = 1.0;        % "MM alpha" (Mimura, 2000)
growth_rate_mm_beta = 1.0;         % "MM beta" (Mimura, 2000)
donor_return_rate = 0.25;          % "k_D"
transconjugant_return_rate = 0.25; % "k_T"
conjugation_rate = 9.0;            % "gamma"

% interpolate function value at centroids
nt = size(t,2);
uintrp = pdeintrp(p,t,u); % size N x nt

% function handles
% NOTE: need to choose one of the two growth functions
growth_malthusian = @(n) growth_rate_malthusian.*n;
growth_mm = @(n) growth_rate_mm_alpha.*n./(1 + growth_rate_mm_beta.*n);

f = zeros(6,nt);
for pt = 1:nt
    g_n = growth_malthusian(uintrp(6,pt));
    f(:,nt) = [g_n.*uintrp(1,pt) - conjugation_rate.*uintrp(2,pt).*uintrp(1,pt)                   + donor_return_rate.*uintrp(4,pt);           % D
               g_n.*uintrp(2,pt) - conjugation_rate.*uintrp(2,pt).*(uintrp(1,pt) + uintrp(3,pt));                                              % R 
               g_n.*uintrp(3,pt) - conjugation_rate.*uintrp(2,pt).*uintrp(3,pt)                   + transconjugant_return_rate.*uintrp(5,pt);  % T
               g_n.*uintrp(4,pt) + conjugation_rate.*uintrp(2,pt).*uintrp(1,pt)                   - donor_return_rate.*uintrp(4,pt);           % Dr
               g_n.*uintrp(5,pt) + conjugation_rate.*uintrp(2,pt).*(uintrp(1,pt)+2.*uintrp(3,pt)) - transconjugant_return_rate.*uintrp(5,pt);  % Tr
               -g_n.*sum(uintrp(1:5,pt))];                                                                                                     % n
end

end
