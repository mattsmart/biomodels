function c = pde_growth_coeff_c(p,t,u,time)

diffusion_rate_bacteria = 0.01;  % TODO may want function for spread bacteria pushing
diffusion_rate_nutrients = 0.2;
c = [diffusion_rate_bacteria; 0.0; 0.0; diffusion_rate_bacteria;     % D
     diffusion_rate_bacteria; 0.0; 0.0; diffusion_rate_bacteria;     % Dr
     diffusion_rate_bacteria; 0.0; 0.0; diffusion_rate_bacteria;     % R
     diffusion_rate_bacteria; 0.0; 0.0; diffusion_rate_bacteria;     % T
     diffusion_rate_bacteria; 0.0; 0.0; diffusion_rate_bacteria;     % Tr 
     diffusion_rate_nutrients; 0.0; 0.0; diffusion_rate_nutrients];  % n

end
