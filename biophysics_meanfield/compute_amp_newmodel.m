function outputs = compute_amp_newmodel(n1,n2)

% Conversion notes: 
%   will need to call n1eff(i) with i = 1

% ==========================
% Commands
format long 
%tol = 1e-6
%options = optimoptions('fsolve','Display','iter','TolX',tol); % Option to display output
% ==========================

% ==========================
% Concentrations (in units of Molar)
% ==========================

% For [AMP] Varying
%n1 = 0.1; % [Na] 0.1 M 
%n2 = 0.0010; % [Mg] 0.1 mM % originally 0.002; 
%np_array = (1*10^-7)*[1:20]; % 0 microM to 2 msicroM
%np_array = (1*10^-7)*[1:2:9,10:5:100]; % 0.1 microM to 10 microM
% For [AMP] Varying (2)
%n1 = 0.150
%n2 = 10^(-3)*5.0
%np_array = 10^(-6)*[0.15]
%np = {0.0000001,0.0000002,0.0000003,0.0000004,0.0000005,0.0000006,0.0000007,0.0000008,0.0000009,0.000001,0.000002,0.000003,0.000004,0.000005,0.000006,0.000007,0.000008,0.000009,0.00001,0.00002,0.00003,0.00004,0.00005,0.00006,0.00007,0.00008,0.00009,0.0001,0.0002,0.0003,0.0004,0.0005,0.0006,0.0007,0.0008,0.0009,0.001,0.002,0.003,0.004,0.005,0.006,0.007,0.008,0.009,0.01,0.02,0.03,0.04,0.05,0.06,0.07,0.08,0.09,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1,2,3,4,5,6,7,8,9,10};
%np = (5*10^-8)*{0.001, 0.01,0.05, 0.1,0.5,1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20};

np_back = (1*10^-7)*[1:9,10:5:100];
np_front = [10^(-9)*[1,5], (1*10^-8)*[1:9]];
np_array = [np_front, np_back]; % 0.01 microM to 10 microM

% ==========================
% Global constants
% ==========================

d1 = 0.3;              % Binding site separation for NA+ in nm
d2 = 0.25;             % Binding site separation for Mg2+ in nm
dp = 0.4;              % Binding site separation for peptides in nm
r1 = 0.34;             % Na ion radius in nm
r2 = 0.43;             % Mg ion radius in nm
v1 = 4/3*pi*(r1)^3;    % Na ion volume in nm^3
v2 = 4/3*pi*(r2)^3;    % Mg ion volume in nm^3

partition = 1;
lattc = 0.64;          % Assumed Lattice Constant, this is a0 in nm (was 0.8, now 0.64)
del = 0.001;           % del multiplied to x to get \[Delta]x...del canot be too much smaller or bigger than x. If its too small, the approx of the numerator / that for denumerator can vary greatly...
%(*Also, not mentioned here is that we have r1=0.34nm, r2=0.43nm where ri is the hydration radius for ions*)

lipidB = 137.5;
lipidG = 12;              % is \Gamma
lipidT = 0.4;             % is \Tau
nu = 0.378*6;             % is \Nu
lh = 0.2;

epsilon = 1/40;   % eps=80 for water and 2 for membrane, so epsilon = eta = eps lipid / eps water
lb = 0.69625;     % Bjerrum length
d = 4.0;          % r_min for Kb

% ==========================
% New constants
% ==========================
A0 = 1.2e9;                    % Initial surface area in nm^2
cJ  = 4.114*10^(-18);          % conversion factor for mJ to kB T 
cA = 10^(18);                  % convert m^2 to nm^2
k0_monolayer = 120; %120 %100;
k0_bilayer = 2*k0_monolayer;
k0 = k0_bilayer;               % CHOOSE monolayer or bilayer
kA = k0_bilayer/(cA*cJ);       % Area compression modulus in mN/m = mJ/m^2, with Joules converted to kbT

% ==========================
% AMP constants
% ==========================
% magainin-2
Q_mgh2 = 4;
dA_mgh2 = 3.00;
H_mgh2 = -10;
vp_mgh2 = 2.5;
eps_sp_mgh2 = 25/(4*pi); % 4x1 rectangle
% gramicidin S
Q_gs = 2;
dA_gs = 1.75;
H_gs = -10; % ??????
vp_gs = 1.4;
eps_sp_gs = 9/(2*pi); % 2x1 rectangle  
% polymyxin B
Q_pb = 6;
dA_pb = 2.00;
H_pb = -10; % ??????
vp_pb = 1.70;
eps_sp_pb = 9/(2*pi); % 2x1 rectangle
% protegrin-1
Q_pg1 = 7;
dA_pg1 = 2.5;
H_pg1 = -10; % ??????
vp_pg1 = 2.60;
eps_sp_pg1 = 16/(3*pi); % 3x1 rectangle

% only 1 or none may be "on"
gs_flag = 0;
pb_flag = 0;
pg1_flag = 0;

if gs_flag
    Q = Q_gs;                % Charge # for peptides
    dA = dA_gs;              % Change in total lattice area in nm^2 per peptide bound (known to be approx 200-300 Angstrom^2 per peptide binding)
    H = H_gs;                % Hydrophobic energy gain in units of kB T per peptide bound
    vp = vp_gs;              % Peptide volume in nm^3 in solution (free)
    eps_sp = eps_sp_gs;      % Peptide shape parameter (from scaled particle theory)
    tag = 'gs'
elseif pb_flag
    Q = Q_pb;                % Charge # for peptides
    dA = dA_pb;              % Change in total lattice area in nm^2 per peptide bound (known to be approx 200-300 Angstrom^2 per peptide binding)
    H = H_pb;                % Hydrophobic energy gain in units of kB T per peptide bound
    vp = vp_pb;              % Peptide volume in nm^3 in solution (free)
    eps_sp = eps_sp_pb;      % Peptide shape parameter (from scaled particle theory)
    tag = 'pb'
elseif pg1_flag
    Q = Q_pg1;                % Charge # for peptides
    dA = dA_pg1;              % Change in total lattice area in nm^2 per peptide bound (known to be approx 200-300 Angstrom^2 per peptide binding)
    H = H_pg1;                % Hydrophobic energy gain in units of kB T per peptide bound
    vp = vp_pg1;              % Peptide volume in nm^3 in solution (free)
    eps_sp = eps_sp_pg1;      % Peptide shape parameter (from scaled particle theory)
    tag = 'pg1'
else
    Q = Q_mgh2;                % Charge # for peptides
    dA = dA_mgh2;              % Change in total lattice area in nm^2 per peptide bound (known to be approx 200-300 Angstrom^2 per peptide binding)
    H = H_mgh2;                % Hydrophobic energy gain in units of kB T per peptide bound
    vp = vp_mgh2;              % Peptide volume in nm^3 in solution (free)
    eps_sp = eps_sp_mgh2;      % Peptide shape parameter (from scaled particle theory)
    tag = 'mgh2'
end

% ==========================
% Declaring Functions
% ==========================

% (0) Lattice Constant Np Dependence

function a_Np = a_Np(a, da, Np)
    a_Np = sqrt(a^2 + da*Np);
end

% (1) Association Constant
function Kb = Kb(q,d)
    tmp = q.*lb;
    integrand = @(r) r.^2.*exp(tmp./r);  %dots, integral fn using matrices
    rmin = d;
    rmax = q.*lb./2;
    Kb = 4.*pi.*integral(integrand, rmin, rmax);
end

% (2) Free ion concentrations in solutions (1M = 0.6022 nm^-3):

% function handle speedups
kbval1_1 = Kb(1, 0.34);
kbval2 = Kb(2, 0.43);

% (A) Free Na+ in soln
function n1eff = n1eff(n1_index)
    % note: n1_index is a dummy variable unless script is compute_na.m
    ref = n1;
    tmp = 0.6022*ref;
    func = @(x) x/(tmp - x)^2 - kbval1_1;
    x0 = ref*0.01;
    n1eff = ref - fzero(func, x0)/0.6022;
end

% (C) Free Mg2+ in soln
function n2eff = n2eff(n2_index)
    % note: n2_index is a dummy variable unless script is compute_mg.m
    tmp = 0.6022*n2;
    func = @(x) x/(tmp - x)^2 - kbval2;
    x0 = n2*0.01; % Na script had n2/2
    n2eff = n2 - fzero(func, x0)/0.6022;
end

% ~~~~~~~~~~~~~~~~~~~~~~~
% declare these as globals using dummy variable (both fixed like n1 n2)
n1eff_amp = n1eff(1);
n2eff_amp = n2eff(1);
% ~~~~~~~~~~~~~~~~~~~~~~~

% (3) Debye Length
function kappa = kappa(j)
    kappa = sqrt(n1eff_amp + 3*n2eff_amp) / 0.3081; 
end

% (5) Dielectric Discontinuity
function Delta = Delta(i)
    kap = kappa(i);
    Delta = (epsilon + kap*d) / (2*epsilon + kap*d)*2;
end

% (7) Script M Correction Functions
% Correction terms for 1*1 site
function M1 = M1(p, kappa, a)
    tmp = a/2;
    integrand = @(x,y) exp(-kappa.*sqrt(x.^2 + y.^2)) ./ sqrt(x.^2 + y.^2);
    xmin = 0; xmax = tmp;
    ymin = 0; ymax = tmp;
    M1 = 4*integral2(integrand,xmin,xmax,ymin,ymax);
end

% Correction terms for Q*1 site
function Mp = Mp(p, kappa, a)
    integrand = @(x,y) exp(-kappa.*sqrt(x.^2 + y.^2)) ./ sqrt(x.^2 + y.^2);
    xmin = 0; xmax = p*a/2;
    ymin = 0; ymax = a/2;
    Mp = 4*integral2(integrand,xmin,xmax,ymin,ymax);
end

% (6) Sum C, the lateral correlation function
% INPUTS: m is ~ 10, # of grids per side to integrate)
%         kappa is debye length
%         lc is lattice constant, may not be a constant
function SumC = SumC(m,kappa,lc)
    summ = 0;
    for i = 1:(m*partition)
        for j = 0:i
            tempor = 1 / ((lc/partition)*sqrt(i^2 + j^2))*exp(-kappa*sqrt(i^2+j^2)*(lc/partition))*(-1)^(i+j-1);
			if j == 0
                summ = summ + 4*tempor;
            elseif j == i
                summ = summ + 4*tempor;
            else
                summ = summ + 8*tempor;
            end
        end
    end
    SumC = summ; %(*+ 1/4*(3/(lc/partition)*Exp[-\[Kappa]*(lc/partition)] - 2/(lc/partition*sqrt[2])*Exp[-\[Kappa]*sqrt[2]*(lc/partition)] - 1/(lc/partition*2)*Exp[-\[Kappa]*2*(lc/partition)])*)];
end

% (11) The free energy expressions
function FreeEnergyp = FreeEnergyp(i, aa, ions) % NEW: Correction terms
    % repeated variables
    kap = kappa(i);
    delt = Delta(i);
    m1 = M1(Q, kap, aa);
    mp = Mp(Q, kap, aa);
    expand_factor = 1 + Q*ions(3);    % N0_tilde = N0 + Q*Np = N0*expand_factor for MGH2
    % compute components
    flps_elec = delt*lb*( ...
        (pi/kap - m1/2) * (ions(1) + 2*ions(2) + Q*ions(3) - 1)^2 / (expand_factor * aa)^2 ...
        - (mp - m1)/2 * (Q*ions(3)*(ions(1) + 2*ions(2) + Q*ions(3))) / (expand_factor * aa)^2 ...
        - (ions(1)/d1 + 2*ions(2)/d2 + Q*ions(3)/dp) / expand_factor ...
        - 2*SumC(10, kap, aa) * ions(2)*(1 - ions(1) - ions(2)) / expand_factor^2);
    flps_entr = (ions(1)*log(ions(1)/(n1*v1)) + ions(2)*log(ions(2)/(n2*v2)) + ions(3)*log(ions(3)/np_array(i)*vp) ...
        + (1 - ions(1) - ions(2)) * log(1 - ions(1) - ions(2)) ...
        - ions(3)*(eps_sp + 1 - log(Q)) ...
        + eps_sp*expand_factor^2/Q )/expand_factor;    
    flps_chem = (ions(1)*0.5*lb*((delt-1)/d1 + kap/(1 + kap*r1)) ...
        + ions(2)*0.5*4*lb*((delt-1)/d2 + kap/(1 + kap*r2)) ...
        + ions(3)*0.5*Q*lb*((delt-1)*(1/dp + (mp-m1)/aa^2) + kap/(1 + kap*r1))) / expand_factor;
    flps_mech = (1/2)*kA*lattc^2*(ions(3)*Q)^2 / expand_factor;
    flps_hydr = ions(3)*H;
    % sum components (full expression for free energy)
    FreeEnergyp = flps_elec + flps_entr + flps_chem + flps_mech + flps_hydr;  
    % free energy with no mechanical AND no hydrophobic contribution below:
    %FreeEnergyp = flps_elec + flps_entr + flps_chem + 0*flps_mech + 0*flps_hydr;  
    % simple electric approximation of the free energy below:
    %flps_simple = delt*lb * pi/kap * (ions(1) + 2*ions(2) + Q*ions(3) - 1)^2 / (expand_factor * aa)^2;
    %FreeEnergyp = flps_simple + flps_entr + flps_chem;  
end  

% (12) Solve the free energy for bound ion fractions
function [ions, flps] = minimize_flps(i, aa)
    ions_guess = [0.1; 0.1; 0.1];
    lowerb = [1e-5; 1e-5; 1e-5];
    upperb = [1.0; 1.0; 1.0];
    A = [1.0, 1.0, Q; 0.0, 0.0, 0.0; 0.0, 0.0, 0.0];
    b = [1.0; 0.0; 0.0];
    func = @(x)FreeEnergyp(i, aa, x);
    [ions, flps] = fmincon(func, ions_guess, A, b, [], [], lowerb, upperb, []);
end

% (13) Tension expressions
% COMMENTS: how to modify when a itself is a function, a(Np)
function Tensionp = Tensionp(i, a0)
    %Tensionp = ( FreeEnergyp(i, sqrt((1-del)*a0^2))  -  FreeEnergyp(i, sqrt((1+del)*a0^2))) / (2*del*a0^2);  OLD
    [~, flps_down] = minimize_flps(i, sqrt((1-del)*a0^2));
    [~, flps_up] = minimize_flps(i, sqrt((1+del)*a0^2));
    Tensionp = (flps_up - flps_down) / (2*del*a0^2);
end

% (14) Mechanical Tension (See latex formulation mechanical energy cost)
function TensionMech = TensionMech(i, a0)
    ions = condp(i, a0);
    TensionMech = kA*Q*ions(3);  % adjusted aug 24 2016
end

% ==========================
% Plotting
% ==========================
% Some reused plotting constants/lists
len = length(np_array);

% Plots various quantities and saves them into individual figures
% (1) plots fractional site occupancy
% (2) plots fractional charge occupancy
% (3) plots free energy
% (4) plots electric + mechanical tension
% (5) plots purely mechanical tension
function plot_all_results = plot_all_results()
    conversion_factor = 4.114;  % for 1 k_B*T/nm^2 = 4.114 mN/m
    xlist = 10^6*np_array;
    ylist_1 = zeros(1,len);
    ylist_2 = zeros(1,len);
    ylist_2x2 = zeros(1,len);
    ylist_p = zeros(1,len);
    ylist_freep = zeros(1,len);
    ylist_tensionp = zeros(1,len);
    ylist_tensionmech = zeros(1,len);
    for i = 1:len
        [ions, flps] = minimize_flps(i, lattc);
        ylist_1(i) = ions(1); % from sigmapr1(i,lattc)
        ylist_2(i) = ions(2); % from sigmapr2(i,lattc)
        ylist_2x2(i) = 2*ions(2);
        ylist_p(i) = Q*ions(3); % from sigmaprp(i,lattc)
        ylist_freep(i) = flps;
        ylist_tensionp(i) = Tensionp(i, lattc);
        ylist_tensionmech(i) = kA*Q*ions(3);
    end
    % Plot fractional site occupancy
    h1 = figure;
    plot(xlist,ylist_1,':bs',xlist,ylist_2,':ks',xlist,ylist_p,':rs')
    xmax = 1.01*np_array(len)*10^6;
    ymax = 1.01;
    axis([0,xmax,-0.01,ymax]) %xmin xmax ymin ymax
    title(['Fractional Site Occupancy vs [AMP]; [Na^{+}] =  ', num2str(n1), ' M, [Mg^{2+}] = ', num2str(n2), ' M'])
    xlabel('[AMP] (\muM)')
    ylabel('Fractional Site Occupancy')
    legend('N_{1} / N_{0} (Na^{+})','N_{2} / N_{0} (Mg^{2+})','Q*N_{p} / N_{0} (AMP)','Location','northeast')
    saveas(h1, 'amp_frac_site.jpg')
    % Plot fractional charge occupancy
    h2 = figure;
    plot(xlist,ylist_1,':bs',xlist,ylist_2x2,':ks',xlist,ylist_p,':rs')
    xmax = 1.01*np_array(len)*10^6;
    ymax = 1.01;
    axis([0,xmax,-0.01,ymax]) %xmin xmax ymin ymax
    title(['Fractional Charge Occupancy vs [AMP]; [Na^{+}] =  ', num2str(n1), ' M, [Mg^{2+}] = ', num2str(n2), ' M'])
    xlabel('[AMP] (\muM)')
    ylabel('Fractional Charge Occupancy')
    legend('N_{1} / N_{0} (Na^{+})','2*N_{2} / N_{0} (Mg^{2+})','Q*N_{p} / N_{0} (AMP)','Location','northeast')
    saveas(h2, 'amp_frac_charge.jpg')
    % Plot free energy
    h3 = figure;
    plot(xlist,ylist_freep,':bs')
    %axis([0,160,-0.5,0.5]) set(gca,'XTickLabel',[0:20:140]) % May need to
    %remove this (could interfere with later plots)
    title(['Free Energy vs [AMP]; [Na^{+}] =  ', num2str(1000*n1),' mM, [Mg^{2+}] = ', num2str(1000*n2), ' mM'])
    xlabel('[AMP] (\muM)')
    ylabel('Free Energy (k_{B} T)')
    saveas(h3, 'amp_freep.jpg')
    % Plot differential tension
    h4 = figure;
    plot(xlist,ylist_tensionp,':bs')
    %axis([0,10^6*np_array(len),-0.6,1.0])
    title(['\Delta \Pi vs [AMP]; [Na^{+}] =  ', num2str(1000*n1),' mM, [Mg^{2+}] = ', num2str(1000*n2), ' mM'])
    xlabel('[AMP] (\muM)')
    ylabel('\Delta \Pi Differential (k_{B} T / nm^2)')
    saveas(h4, 'amp_tensionp.jpg')
    % Plot mechanical tension
    h5 = figure;
    plot(xlist,ylist_tensionmech,':bs')
    axis([0,10^6*np_array(len),0,10.0])
    title(['\Delta \Pi (Mechanical) vs [AMP]; [Na^{+}] =  ', num2str(1000*n1),' mM, [Mg^{2+}] = ', num2str(1000*n2), ' mM'])
    xlabel('[AMP] (\muM)')
    ylabel('\Delta \Pi Mechanical (k_{B} T / nm^2)')
    saveas(h5, 'amp_tensionmech.jpg')
end

% Plot Fractional Site Occupancy (AMP modified) Plots the fractional site
% occupancy based on [AMP] for N1, N2, QNp
function plot_frac_site_AMP = plot_frac_site_AMP()
    xlist = 10^6*np_array;
    ylist_1 = zeros(1,len);
    ylist_2 = zeros(1,len);
    ylist_p = zeros(1,len);
    for i = 1:len
        [ions, flps] = minimize_flps(i, lattc);
        ylist_1(i) = ions(1); % from sigmapr1(i,lattc)
        ylist_2(i) = ions(2); % from sigmapr2(i,lattc)
        ylist_p(i) = Q*ions(3); % from sigmaprp(i,lattc)
    end
    h = figure;
    plot(xlist,ylist_1,':bs',xlist,ylist_2,':ks',xlist,ylist_p,':rs')
    xmax = 1.01*np_array(len)*10^6;
    ymax = 1.01;
    axis([0,xmax,-0.01,ymax]) %xmin xmax ymin ymax
    title(['Fractional Site Occupancy vs [AMP]; [Na^{+}] =  ', num2str(n1), ' M, [Mg^{2+}] = ', num2str(n2), ' M'])
    xlabel('[AMP] (\muM)')
    ylabel('Fractional Site Occupancy')
    legend('N_{1} / N_{0} (Na^{+})','N_{2} / N_{0} (Mg^{2+})','Q*N_{p} / N_{0} (AMP)','Location','northeast')
    saveas(h, 'amp_frac_site_MH.jpg')
end

% Plot Fractional Charge Occupancy (AMP modified) Plots the fractional
% CHARGE occupancy based on [AMP] for N1, 2*N2, QNp
function plot_frac_charge_AMP = plot_frac_charge_AMP()
    xlist = 10^6*np_array;
    ylist_1 = zeros(1,len);
    ylist_2 = zeros(1,len);
    ylist_p = zeros(1,len);
    for i = 1:len
        [ions, flps] = minimize_flps(i, lattc);
        ylist_1(i) = ions(1); % from sigmapr1(i,lattc)
        ylist_2(i) = 2*ions(2); % from 2*sigmapr2(i,lattc)
        ylist_p(i) = Q*ions(3); % from sigmaprp(i,lattc)
    end
    h = figure;
    plot(xlist,ylist_1,':bs',xlist,ylist_2,':ks',xlist,ylist_p,':rs')
    xmax = 1.01*np_array(len)*10^6;
    ymax = 1.01;
    axis([0,xmax,-0.01,ymax]) %xmin xmax ymin ymax
    title(['Fractional Charge Occupancy vs [AMP]; [Na^{+}] =  ', num2str(n1), ' M, [Mg^{2+}] = ', num2str(n2), ' M'])
    xlabel('[AMP] (\muM)')
    ylabel('Fractional Charge Occupancy')
    legend('N_{1} / N_{0} (Na^{+})','2*N_{2} / N_{0} (Mg^{2+})','Q*N_{p} / N_{0} (AMP)','Location','northeast')
    saveas(h, 'amp_frac_charge_MH.jpg')
end

% Plot Free energy to understand tension
function plot_freep_AMP = plot_freep_AMP()    
    aa = lattc;
    xlist = 10^6*np_array;
    len = length(xlist);
    ylist_free_p = zeros(1,len);
    for i = 1:len
        ylist_free_p(i) = FreeEnergyp(i,lattc)
    end
    h = figure
    plot(xlist,ylist_free_p,':bs')
    %axis([0,160,-0.5,0.5]) set(gca,'XTickLabel',[0:20:140]) % May need to
    %remove this (could interfere with later plots)
    title(['Free Energy vs [AMP]; [Na^{+}] =  ', num2str(1000*n1),' mM, [Mg^{2+}] = ', num2str(1000*n2), ' mM'])
    xlabel('[AMP] (\muM)')
    ylabel('Free Energy (k_{B} T)')
    saveas(h, 'amp_freep_MH.jpg')
end

% Plot Tension (AMP Modified) Delta_pi (tension) with monovalent ions,
% divalent ions, and AMPs  Fig17, LPS_Ma.pdf
%   [Table[{n1[[i]], Tensionp[i,lattc]},{i,1,pts}],
%   PlotRange->{{0,0.21},{-2,2}},
%   AxesLabel->{"[\!\(\*SuperscriptBox[\(Na\),
%   \(+\)]\)](M)","\[CapitalDelta]\[pi] (\!\(\*SubscriptBox[\(k\),
%   \(B\)]\)T/\!\(\*SuperscriptBox[\(nm\), \(2\)]\))"}, PlotLabel->"
%   \[CapitalDelta]\[pi] vs [\!\(\*SuperscriptBox[\(Na\), \(+\)]\)],
%   [\!\(\*SuperscriptBox[\(Mg\), \(\(2\)\(+\)\)]\)]=0.5mM,
%   [AMP]=0.5\[Micro]M, revised"]*)
function plot_tensionp_AMP = plot_tensionp_AMP()    
    aa = lattc;
    xlist = 10^6*np_array;
    ylist_tensionp = zeros(1,len);
    for i = 1:len
        ylist_tensionp(i) = Tensionp(i,lattc);
    end
    h = figure
    plot(xlist,ylist_tensionp,':bs')
    %axis([0,10^6*np_array(len),-0.6,1.0])
    title(['\Delta \Pi vs [AMP]; [Na^{+}] =  ', num2str(1000*n1),' mM, [Mg^{2+}] = ', num2str(1000*n2), ' mM'])
    xlabel('[AMP] (\muM)')
    ylabel('\Delta \Pi (k_{B} T / nm^2)')
    saveas(h, 'amp_tensionp_MH.jpg')
end

% Plot Mechanical Tension
function plot_tensionmech_AMP = plot_tensionmech_AMP()    
    conversion_factor = 4.114
    xlist = 10^6*np_array;
    ylist_tensionmech = zeros(1,len);
    for i = 1:len
        ylist_tensionmech(i) = TensionMech(i,lattc) * conversion_factor;
    end
    h = figure
    plot(xlist,ylist_tensionmech,':bs')
    axis([0,10^6*np_array(len),0,40.0])
    title(['\Delta \Pi (Mechanical) vs [AMP]; [Na^{+}] =  ', num2str(1000*n1),' mM, [Mg^{2+}] = ', num2str(1000*n2), ' mM'])
    xlabel('[AMP] (\muM)')
    %ylabel('\Delta \Pi Mechanical (k_{B} T / nm^2) ')
    ylabel('\Delta \Pi Mechanical (mN / m) ')
    saveas(h, 'amp_tensionmech_MH.jpg')
end

% ========================== Save Data ==========================

function get_custom_data = get_custom_data()
    xlist = np_array;
    ylist_1 = zeros(1,len);
    ylist_2 = zeros(1,len);
    ylist_p = zeros(1,len);
    ylist_free_p = zeros(1,len);
    ylist_tensionp = zeros(1,len);
    ylist_tensionmech = zeros(1,len);
    for i = 1:len
        condp_tmp = condp(i,lattc);
        ylist_1(i) = condp_tmp(1); % from sigmapr1(i,lattc)
        ylist_2(i) = 0.5 + condp_tmp(2); % from sigmapr2(i,lattc)
        ylist_p(i) = Q*(condp_tmp(3)); % from sigmaprp(i,lattc)
        ylist_free_p(i) = FreeEnergyp(i,lattc);
        ylist_tensionp(i) = Tensionp(i,lattc);
        ylist_tensionmech(i) = TensionMech(i,lattc);
    end
    ylist_22 = 2*ylist_2; % reps frac charge occ for Mg
    %M = real([np_array', ylist_1', ylist_2', ylist_22', ylist_p',
    %ylist_free_p', ylist_tensionp']);
    M = real([np_array', ylist_1', ylist_2', ylist_22', ylist_p', ylist_free_p', ylist_tensionp', ylist_tensionmech']);
    filename = sprintf('data\\AMP_%s_data_na_%d_mM_mg_%1.1f_mM.txt', tag, 1000*n1, 1000*n2);
    fid = fopen(filename, 'w');
    col_id = sprintf('_na_%d_mg_%1.1f_%s', 1000*n1, 1000*n2, tag);
    col_header = sprintf('AMP,n1%s,n2%s,2_n2%s,Q_np%s,Free%s,Ten%s,TenMech%s\n', col_id, col_id, col_id, col_id, col_id, col_id, col_id);
    fprintf(fid,col_header);
    fclose(fid)
    dlmwrite(filename,M,'-append','precision',4)
    get_custom_data = M;
end

function get_tensionmech_data = get_tensionmech_data()
    conversion_factor = 4.114
    ylist_tensionmech = zeros(1,len);
    for i = 1:len
        ylist_tensionmech(i) = TensionMech(i,lattc)*conversion_factor;
    end
    M = real([np_array', ylist_tensionmech'])
    filename = sprintf('data\\AMP_MechTen_na_%d_mM_mg_%1.1f_mM_k_%d.txt', 1000*n1, 1000*n2, k0);
    fid = fopen(filename, 'w');
    col_id = sprintf('_na_%d_mg_%1.1f_k_%d', 1000*n1, 1000*n2, k0);
    col_header = sprintf('AMP,MechTen%s\n', col_id);
    fprintf(fid, col_header);
    fclose(fid)
    dlmwrite(filename,M,'-append','precision',4)
    get_tensionmech_data = M;
end

function get_line = get_line()
    PL = 0.1
    PL_line = PL + zeros(1,pts_np)
    filename = sprintf('data\\AMP_PL_line_%d.txt', pts_np);
    fid = fopen(filename, 'w');
    col_header = sprintf('AMP,PL\n');
    fprintf(fid, col_header);
    fclose(fid)
    dlmwrite(filename,[np_array', PL_line'],'-append','precision',4)
end

function get_tension_data = get_tension_data()
    conversion_factor = 4.114
    ylist_tensionmech = zeros(1,len);
    ylist_tensionelec = zeros(1,len);
    for i = 1:len
        ylist_tensionmech(i) = TensionMech(i,lattc)*conversion_factor;
        ylist_tensionelec(i) = Tensionp(i,lattc)*conversion_factor
    end
    M = real([np_array', ylist_tensionmech', ylist_tensionelec'])
    get_tension_data = M;
end

% ========================== Main ==========================

function main = main()
    plot_all_results()
    %plot_frac_site_AMP() 
    %plot_frac_charge_AMP() 
    %plot_freep_AMP()
    %plot_tensionp_AMP()
    %plot_tensionmech_AMP()
    %get_line() get_tension_data() n1 n2*1000 np_array*1000*1000
    %outputs = get_custom_data();
end

main()

end

%{
TODO 1. fix mupccc, mu1c
%}