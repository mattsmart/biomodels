function [c, a, f, d] = pde_conjugation_system(flag_nonlinear_diffusion, flag_monod_growth, flag_nonnegative)
% specify coefficients which define the pde system
% Args:
%     flag_nonlinear_diffusion  -- [bool] default true
%     flag_monod_growth         -- [bool] default true
%     flag_nonnegative          -- [bool] default true
% Returns:
%     [c, a, f. d]              -- pde system coefficient arrays

% STATES
% u1 - D    donors
% u2 - R    recipients
% u3 - T    transconjugants
% u4 - Dr   refractory donors; just conjugated
% u5 - Tr   refractory transconjugants; just received plasmid
% u6 - n    nutrients

% solver/system documentation:
% http://www.mathworks.com/help/pde/ug/multidimensional-coefficients.html
% parabolic system with N=6 states

% notes:
% Recipients sometimes go negative (numerical error), so use max(R,0)

% =======================================================================
% Constants and Functions
% =======================================================================

% growth parameters
growth_rate_malthusian = 1.3863;            % log(2)/(T*n0), T = generation time (30 min for ecoli) "alpha" (Mimura, 2000)
monod_mu_max = 1.3863;                      % [1/hour] "MM alpha" (Mimura, 2000)
monod_Ks = 0.26;                            % [g/m^3]  "MM beta" (Mimura, 2000) 
yield_coefficient = 1.0;                    % [#] mass of biomass or cell produced per unit nutrient mass

% conjugation [arameters
donor_return_rate = 0.25;                   % "k_D"
transconjugant_return_rate = 0.25;          % "k_T"
conjugation_rate = 9.0;                     % "gamma"

% diffusion parameters
diffusion_rate_nutrients = 0.05;            % "d_0" [m^2/hour]
diffusion_rate_bacteria_linear = 0.01;      % "d_1" [m^2/(hour*bacteria)] 
diffusion_rate_bacteria_nonlinear = 0.001;  % "d_1" [m^2/(hour*bacteria^2)] 

% function handles
% NOTE: need to choose one of the growth functions
growth_malthusian = sprintf('%0.4f.*u(6,:))',growth_rate_malthusian);
growth_monod = sprintf('%0.4f.*u(6,:)./(%0.4f+u(6,:))',monod_mu_max,monod_Ks);


% =======================================================================
% "c" Coefficient (Diffusion tensor)
% interpreted as  N^2 x  1  x  mesh points
% =======================================================================

% string formatting
diff_bacteria_linear = sprintf('%0.4f',diffusion_rate_bacteria_linear);
diff_bacteria_nonlinear = sprintf('%0.4f.*(u(1,:)+u(2,:)+u(3,:)+u(4,:)+u(5,:))',diffusion_rate_bacteria_nonlinear);
if flag_nonnegative
    diff_bacteria_nonlinear = string_array_max(diff_bacteria_nonlinear);
end
diff_nutrient = sprintf('%0.4f',diffusion_rate_nutrients);

% prepare diagonal blocks of c tensor
if flag_nonlinear_diffusion
    diff_bacteria = diff_bacteria_nonlinear;
else
    diff_bacteria = diff_bacteria_linear;
end
c_bacteria = char(diff_bacteria,'0','0',diff_bacteria);
c_nutrient = char(diff_nutrient,'0','0',diff_nutrient);

c = char(c_bacteria,c_bacteria,c_bacteria,c_bacteria,c_bacteria,c_nutrient);


% =======================================================================
% "a" Coefficient (Linear factor)
% interpreted as  N  x  N  x  mesh points
% =======================================================================

% choose growth factor and specify nutrient dependence term
if flag_monod_growth
    growth_factor = growth_monod;
    a_6 = sprintf('%0.4f./(%0.4f+u(6,:)).*(u(1,:)+u(2,:)+u(3,:)+u(4,:)+u(5,:))./%0.4f',monod_mu_max,monod_Ks,yield_coefficient);
else
    growth_factor = growth_malthusian;
    %a_6 = sprintf('%0.4f.*(u(1,:)+u(2,:)+u(3,:)+u(4,:)+u(5,:))./%0.4f',growth_rate_malthusian,yield_coefficient);
    a_6 = '0.0';  % no depletion of nutrients
end

% string formatting for diagonals of "a" matrix
if flag_nonnegative
    a_1 = sprintf('%0.4f.*max(0,u(2,:))-%s',conjugation_rate,growth_factor);
    a_2 = sprintf('%0.4f.*max(0,(u(1,:)+u(3,:)))-%s',conjugation_rate,growth_factor);
    a_3 = sprintf('%0.4f.*max(0,u(2,:))-%s',conjugation_rate,growth_factor);
    a_4 = sprintf('%0.4f-%s',donor_return_rate,growth_factor);
    a_5 = sprintf('%0.4f-%s',transconjugant_return_rate,growth_factor);
    a_6 = string_array_max(a_6);
else
    a_1 = sprintf('%0.4f.*u(2,:)-%s',conjugation_rate,growth_factor);
    a_2 = sprintf('%0.4f.*(u(1,:)+u(3,:))-%s',conjugation_rate,growth_factor);
    a_3 = sprintf('%0.4f.*u(2,:)-%s',conjugation_rate,growth_factor);
    a_4 = sprintf('%0.4f-%s',donor_return_rate,growth_factor);
    a_5 = sprintf('%0.4f-%s',transconjugant_return_rate,growth_factor);
end

a = char(a_1, a_2, a_3, a_4, a_5, a_6);

% =======================================================================
% "f" Coefficient (Forcing vector)
% interpreted as  N  x  1  x  mesh points
% =======================================================================

% string formatting for rows of "f" vector
f_1 = sprintf('%0.4f.*u(4,:)',donor_return_rate);
f_2 = '0.0';
f_3 = sprintf('%0.4f.*u(5,:)',transconjugant_return_rate);
f_4 = sprintf('%0.4f.*u(2,:).*u(1,:)',conjugation_rate);
f_5 = sprintf('%0.4f.*u(2,:).*(u(1,:)+2.*u(3,:))',conjugation_rate);
f_6 = '0.0';

f = char(f_1, f_2, f_3, f_4, f_5, f_6);
if flag_nonnegative
    f = string_array_max(f);
end


% =======================================================================
% "d" Coefficient (Time derivative dependence)
% interpreted as  N  x  N  x  mesh points
% =======================================================================

d = 1;


end
