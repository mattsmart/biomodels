function c = pde_growth_coeff_c(p,t,u,time)

diffusion_rate_bacteria = 0.01;
diffusion_rate_nutrients = 0.2;
c = [diffusion_rate_bacteria; 0.0; 0.0; diffusion_rate_bacteria; 
     diffusion_rate_nutrients; 0.0; 0.0; diffusion_rate_nutrients];

end
