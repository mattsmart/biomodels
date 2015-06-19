function f = pde_conjugation_coeff_f(p,t,u,time)

% STATES
% u1 - D    donors
% u2 - R    recipients
% u3 - T    transconjugants
% u4 - Dr   refractory donors; just conjugated
% u5 - Tr   refractory transconjugants; just received plasmid
% u6 - n    nutrients

% constants
donor_return_rate = 0.25;           % "k_D"
transconjugant_return_rate = 0.25;  % "k_T"
conjugation_rate = 0.3;             % "gamma"

% interpolate function value at centroids
nt = size(t,2);
uintrp = pdeintrp(p,t,u); % size N x nt

f = zeros(6,nt);
for pt = 1:nt
    f(:,nt) = [donor_return_rate.*uintrp(4,pt);                                  % u1 - D
               0.0;                                                              % u2 - R 
               transconjugant_return_rate.*uintrp(5,pt);                         % u3 - T
               conjugation_rate.*uintrp(1,pt).*uintrp(2,pt);                     % u4 - Dr
               -conjugation_rate.*(uintrp(1,pt)+2.*uintrp(3,pt)).*uintrp(2,pt);  % u5 - Tr
               0.0];                                                             % u6 - n
end

end
