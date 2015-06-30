function [c,a,f,d] = pde_conjugation_system()
% specify coeffiencts which define the pde system

% documentation:
% http://www.mathworks.com/help/pde/ug/multidimensional-coefficients.html
% parabolic system with N=6 states

% STATES
% u1 - D    donors
% u2 - R    recipients
% u3 - T    transconjugants
% u4 - Dr   refractory donors; just conjugated
% u5 - Tr   refractory transconjugants; just received plasmid
% u6 - n    nutrients


% =======================================================================
% Constants and Functions
% =======================================================================

% growth parameters
% alpha = log(2)/(T*n0) where T is generation time of the bacteria (0.5h e.coli)
growth_rate_malthusian = 1.3863;            % "alpha" (Mimura, 2000)
growth_rate_mm_alpha = 1.0;                 % "MM alpha" (Mimura, 2000)
growth_rate_mm_beta = 1.0;                  % "MM beta" (Mimura, 2000)

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
growth_malthusian = @(n) growth_rate_malthusian.*n;
growth_mm = @(n) growth_rate_mm_alpha.*n./(1 + growth_rate_mm_beta.*n);


% =======================================================================
% "c" Coefficient (Diffusion tensor)
% interpreted as  N^2 x  1  x  mesh points
% =======================================================================

% string formatting
diff_bacteria_linear = sprintf('%0.4f',diffusion_rate_bacteria_linear);
diff_bacteria_nonlinear = sprintf('%0.4f.*(u(1,:)+u(2,:)+u(3,:)+u(4,:)+u(5,:))',diffusion_rate_bacteria_nonlinear);
diff_nutrient = sprintf('%0.4f',diffusion_rate_nutrients);

% prepare diagonal blocks of c tensor
diff_bacteria = diff_bacteria_nonlinear;  % choose linear or non-linear diffusion
c_bacteria = char(diff_bacteria,'0','0',diff_bacteria);
c_nutrient = char(diff_nutrient,'0','0',diff_nutrient);

c = char(c_bacteria,c_bacteria,c_bacteria,c_bacteria,c_bacteria,c_nutrient);

% =======================================================================
% "a" Coefficient (Linear factor)
% interpreted as  N  x  N  x  mesh points
% =======================================================================

% string formatting for diagonals of "a" matrix
a_1 = sprintf('%0.4f.*u(2,:)-%0.4f.*u(6,:)',conjugation_rate,growth_rate_malthusian);
a_2 = sprintf('%0.4f.*(u(1,:)+u(3,:))-%0.4f.*u(6,:)',conjugation_rate,growth_rate_malthusian);
a_3 = sprintf('%0.4f.*u(2,:)-%0.4f.*u(6,:)',conjugation_rate,growth_rate_malthusian);
a_4 = sprintf('%0.4f-%0.4f.*u(6,:)',donor_return_rate,growth_rate_malthusian);
a_5 = sprintf('%0.4f-%0.4f.*u(6,:)',transconjugant_return_rate,growth_rate_malthusian);
a_6 = sprintf('%0.4f.*(u(1,:)+u(2,:)+u(3,:)+u(4,:)+u(5,:))',growth_rate_malthusian);

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


% =======================================================================
% "d" Coefficient (Time derivative dependence)
% interpreted as  N  x  N  x  mesh points
% =======================================================================

d = 1;


% =======================================================================
% Custom Settings (pick at most one of each)
% =======================================================================
% 1) nonlinear (default) / linear / no diffusion
%c = char('0.01','0.01','0.01','0.01','0.01','0.05');  % linear diffusion
%c = 0;                                                % no movement
% 2) nutrient depletion (default) / no nutrient depletion
%a(6,:) = '0'                                          % no nutrient loss

end
