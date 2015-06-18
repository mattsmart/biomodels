function f = pde_conjugation_coeff_f(p,t,u,time)

% STATES
% u1 - D    donors
% u2 - R    recipients
% u3 - T    transconjugants
% u4 - Dr   refractory donors; just conjugated
% u5 - Tr   refractory transconjugants; just received plasmid
% u6 - n    nutrients

% constants
donor_return_rate = 0.5;           % "k_D"
transconjugant_return_rate = 0.5;  % "k_T"
conjugation_rate = 0.5;            % "gamma"

f = [donor_return_rate.*u(4);                    % u1 - D
     0.0;                                        % u2 - R 
     transconjugant_return_rate.*u(5);           % u3 - T
     conjugation_rate.*u(1).*u(2);               % u4 - Dr
     -conjugation_rate.*(u(1) + 2.*u(3)).*u(2);  % u5 - Tr
     0.0];                                       % u6 - n

end
