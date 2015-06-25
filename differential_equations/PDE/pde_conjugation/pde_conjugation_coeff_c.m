function c = pde_conjugation_coeff_c(p,t,u,time)

% STATES
% u1 - D    donors
% u2 - R    recipients
% u3 - T    transconjugants
% u4 - Dr   refractory donors; just conjugated
% u5 - Tr   refractory transconjugants; just received plasmid
% u6 - n    nutrients

% constants
diffusion_rate_nutrients = 0.05;  % "d_0" [m^2/hour]
diffusion_rate_bacteria = 0.01;  % "d_1" [m^2/(hour*bacteria)] 

% interpolate function value at centroids
nt = size(t,2);
uintrp = pdeintrp(p,t,u); % size N x nt

% function handles
%diffusion_function_bacteria = @(pt) diffusion_rate_bacteria.*sum(uintrp(1:5,pt));

c = zeros(24, nt);  % 24 rows due to arbitrary specification pattern for c
for pt = 1:nt
    %diffusion_value_bacteria = diffusion_function_bacteria(pt);
    diffusion_value_bacteria = diffusion_rate_bacteria;
    c(:,pt) = [diffusion_value_bacteria; 0.0; 0.0; diffusion_value_bacteria;  % D
               diffusion_value_bacteria; 0.0; 0.0; diffusion_value_bacteria;  % T
               diffusion_value_bacteria; 0.0; 0.0; diffusion_value_bacteria;  % R
               diffusion_value_bacteria; 0.0; 0.0; diffusion_value_bacteria;  % Dr
               diffusion_value_bacteria; 0.0; 0.0; diffusion_value_bacteria;  % Tr
               diffusion_rate_nutrients; 0.0; 0.0; diffusion_rate_nutrients]; % n
end

end
