function c = pde_conjugation_coeff_c(p,t,u,time)

% STATES
% u1 - D    donors
% u2 - R    recipients
% u3 - T    transconjugants
% u4 - Dr   refractory donors; just conjugated
% u5 - Tr   refractory transconjugants; just received plasmid
% u6 - n    nutrients

% constants
diffusion_rate_nutrients = 0.2;  % "d_0"
diffusion_rate_bacteria = 0.01;  % "d_1"

% function handles
diffusion_function_bacteria = @(u) ...
    diffusion_rate_bacteria * (u(1) + u(2) + u(3) + u(4) + u(5));

c = [diffusion_function_bacteria(u); 0.0; 0.0; diffusion_function_bacteria(u);  % D
     diffusion_function_bacteria(u); 0.0; 0.0; diffusion_function_bacteria(u);  % R
     diffusion_function_bacteria(u); 0.0; 0.0; diffusion_function_bacteria(u);  % T
     diffusion_function_bacteria(u); 0.0; 0.0; diffusion_function_bacteria(u);  % Dr
     diffusion_function_bacteria(u); 0.0; 0.0; diffusion_function_bacteria(u);  % Tr
     diffusion_rate_nutrients; 0.0; 0.0; diffusion_rate_nutrients];  % n

end
