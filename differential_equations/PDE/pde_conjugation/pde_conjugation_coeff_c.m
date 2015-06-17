function c = pde_conjugation_coeff_c(p,t,u,time)

diffusion_rate = 5.0;
c = [diffusion_rate; diffusion_rate; 1.0; 1.0]; % WRONG need symmetry

end
