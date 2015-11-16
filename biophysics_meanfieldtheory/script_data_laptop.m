% ==========================
% Define Static Concentrations for each graph
% ==========================

n1_default = 0.1 % in units of M
n2_default = 10^(-3)*0.1 % in units of molar

np_vals_3A1 = 10^(-6)*[0,0.1,1,10];
n2_vals_3A2 = 10^(-3)*[0.1,0.5,1.0,5.0];
np_vals_3B1 = 10^(-6)*[0,0.1,1,10];
n2_vals_3B2 = 10^(-3)*[0.1,0.5,1.0];
n2_vals_4A1 = [0,0.00001,0.0001,0.001]; % in units of molar
np_vals_4A2 = 10^(-6)*[0,0.1,1.0,10.0]; % in units of molar
np_vals_4B1 = 10^(-6)*[0,0.1,1.0,10.0,100.0]; % in units of molar
n2_vals_4B2 = 10^(-3)*[0.1,0.25,0.5,0.75,1.0]; % in units of molar

n2_vals_test = 10^(-3)*[0.1,1.0,2.0,5.0,10.0];
n2_vals_extra = 10^(-3)*[5.0,10.0];

np_vals_paper1 = 10^(-6)*[0.1,0.5,1];
np_vals_paper1_extra = 10^(-6)*[5,10];

n2_vals_various_AMPs = 10^(-3)*[0.1,0.5,1.0,5.0];

% Get AMP plots for various AMPs
for i = 1:4
    n1 = n1_default
    n2 = n2_vals_various_AMPs(i)
    %data = MATLAB_AMP(n1,n2)
    data = MATLAB_AMP_old(n1,n2)
end
