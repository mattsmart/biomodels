function a = pde_conjugation_coeff_a(p,t,u,time)

% STATES
% u1 - D    donors
% u2 - R    recipients
% u3 - T    transconjugants
% u4 - Dr   refractory donors; just conjugated
% u5 - Tr   refractory transconjugants; just received plasmid
% u6 - n    nutrients

% http://www.mathworks.com/help/pde/ug/multidimensional-coefficients.html
% parabolic system - 6 states
a = zeros(6, nt);   % 6 rows correspond to each state
for pt = 1:nt
    a(:,pt) = [0;   % D
               0;   % R 
               0;   % T
               0;   % Dr
               0;   % Tr
               0];  % n
end

end
