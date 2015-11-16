% ==========================
% Define Static Concentrations for each graph
% ==========================

n1_default = 0.1; % in units of M
n2_default = 10^(-3)*0.1; % in units of molar

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

n2_vals_MHL_test = 10^(-3)*[5.0,10.0]; % in units of molar

n2_vals_various_AMPs = 10^(-3)*[0.1,0.5,1.0,5.0];

% ==========================
% Main
% ==========================

%{
for i = 1:5
    n1_test = 0.150; % Molar
    dataset = compute_amp(n1_test,n2_vals_test(i));
end

for i = 1:5
    n1_test = 0.100; % Molar
    dataset = compute_amp(n1_test,n2_vals_test(i));
end
%}

% Get Plot 4A1 Data (still need 1 curve)
% for i = 2:4
%     np = 0
%     n2 = n2_vals_4A1(i)
%     data = compute_na(n2,np)
% end

% Get Plot 4A2 Data (still need 2 curves)
% for i = 2:4
%     n2 = n2_default
%     np = np_vals_4A2(i)
%     data = compute_na(n2,np)
% end

% Get Plot 4B1 Data (still need 1 curve)
% for i = 2:5
%     n1 = n1_default
%     np = np_vals_4B1(i)
%     data = compute_mg(n1,np)
% end

%{
Paper Figures Description
	-8 total
		-4 are frac charge
		-4 are tension

Fig 3 A1: 
CHARGE vs [Mg] mM
-Mg runs from 0,0.2,0.4,... 1.0 mM
-plots Mg, AMP curves for values of [Mg] = 0.1 mM, 0.5 mM, 1mM (6 total curves)
-[Na] value is 0.1 M

Fig 3 A2

CHARGE vs [AMP] microM
-AMP runs from 0,2,4,..10 microM
-plots Mg, AMP curves for values of AMP = 0, 0.1 microM, 1 microM, 10 microM (6 total curves)
-[Na] value is 0.1 M

Fig 3 B1 / B2 have area exclusion removed.. ignore for now

Fig 4 A1
TENSION vs [Na] mM
-Na runs from 0,20,40,..,140 (150 last pt) mM
-plots tension curves for values of [Mg] = 0, 0.01 mM (10 microM), 0.1 mM, 1mM (5 total curves)
-[AMP] value is 0 mM

Fig 4 A2
TENSION vs [Na] mM
-Na runs from 0,20,40,..,140 (150 last pt) mM
-1 curve has AMP = 0, Mg = 0
-plots tension curves for values of [AMP] = 0.01 mM (10 microM), 0.001 mM (1 microM), 0.1 microM, 0 microM (4 total curves)
-[Mg] value is 0 M

Fig 4 B1
TENSION vs [Mg] mM
-Mg runs from 0,0.2,0.4,..,1.0 mM
-plots tension curves for values of 
 [AMP] = 0 microM, 0.1 microM, 1 microM, 10 microM, 100 microM (5 curves)
-[Na] value is 0.1 M

Fig 4 B2
TENSION vs [AMP] microM
-AMP runs from 0,2,4,..,10 microM
-plots tension curves for values of 
 [Mg] = 0.1 mM, 0.25 mM, 0.5 mM, 0.75 mM, 1mM (5 curves)
-[Na] value is 0.1 M

%}

% Mg Figures
% Get Plot 4B1 Data (still need "0" amp curve)
for i = 1:3
    n1 = n1_default;
    np = np_vals_paper1(i);
    %data = compute_mg(n1,np)
end
for i = 1:2
    n1 = n1_default;
    np = np_vals_paper1_extra(i);
    %data = compute_mg(n1,np)
end

% Get Plot 4B2 Data
for i = 1:5
    n1 = n1_default;
    n2 = n2_vals_4B2(i);
    %data = compute_amp(n1,n2)
end

% Get Plot 4B2 Data (bigger spread)
for i = 1:2
    n1 = n1_default;
    n2 = n2_vals_MHL_test(i);
    %data = compute_amp(n1,n2);
end


% Get AMP plots for various AMPs
for i = 1:4
    n1 = n1_default;
    n2 = n2_vals_various_AMPs(i);
    data = compute_amp(n1,n2);
end

