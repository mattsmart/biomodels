function outputs = compute_mg_newmodel(n1,np)

% ==========================
% Commands
format long 
%tol = 1e-6
%options = optimoptions('fsolve','Display','iter','TolX',tol); % Option to display output
% ==========================

% ==========================
% Concentrations (in units of Molar)
% ==========================

% For [Mg] Varying
%n1 = 0.1;
%np = 0.00001;

%ORIGINAL
n2_array = 40*[0.00000001,0.0000001,0.0000002,0.0000003,0.0000004, 0.0000005, 0.000001, 0.0000015, 0.000002, 0.0000025, 0.000003, 0.0000035, 0.000004, 0.0000045, 0.000005, 0.0000055, 0.000006, 0.0000065, 0.000007, 0.0000075,  0.000008, 0.0000085, 0.000009, 0.0000095, 0.00001, 0.000011, 0.000012, 0.000013, 0.000014, 0.000015, 0.000016, 0.000017, 0.000018, 0.000019, 0.00002, 0.000021, 0.000022, 0.000023, 0.000024, 0.000025]; 

% FOR INVESTIGATION
n2_array_5mM_back = [110:15:500]*10^(-5);
n2_array = [n2_array, n2_array_5mM_back];

%n2_array = [5.0*10^(-3)]
%n1 = 0.150
%np = 0.615*10^(-6)

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
am_na = 0.022990;      % Na atomic mass kg/mol
am_mg = 0.024305;      % Mg atomic mass kg/mol

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
kA = k0/(cA*cJ);               % Area compression modulus in mN/m = mJ/m^2, with Joules converted to kbT

% ==========================
% AMP constants
% ==========================
% magainin-2
Q_mgh2 = 4;
dA_mgh2 = 3.00;
H_mgh2 = -10;
vp_mgh2 = 2.5;
eps_sp_mgh2 = 25/(4*pi); % 4x1 rectangle
am_mgh2 = 2.466900;      % MGH2 atomic mass kg/mol
% gramicidin S
Q_gs = 2;
dA_gs = 1.75;
H_gs = -10; % ??????
vp_gs = 1.4;
eps_sp_gs = 9/(2*pi);    % 2x1 rectangle
am_gs = 9999999999;      % atomic mass kg/mol (fix************************)
% polymyxin B
Q_pb = 6;
dA_pb = 2.00;
H_pb = -10; % ??????
vp_pb = 1.70;
eps_sp_pb = 9/(2*pi);    % 2x1 rectangle
am_pb = 9999999999;      % atomic mass kg/mol (fix************************)
% protegrin-1
Q_pg1 = 7;
dA_pg1 = 2.5;
H_pg1 = -10; % ??????
vp_pg1 = 2.60;
eps_sp_pg1 = 16/(3*pi);  % 3x1 rectangle
am_pg1 = 9999999999;     % atomic mass kg/mol (fix************************)

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
    am_amp = am_gs;          % Peptide atomic mass kg/mol
elseif pb_flag
    Q = Q_pb;                % Charge # for peptides
    dA = dA_pb;              % Change in total lattice area in nm^2 per peptide bound (known to be approx 200-300 Angstrom^2 per peptide binding)
    H = H_pb;                % Hydrophobic energy gain in units of kB T per peptide bound
    vp = vp_pb;              % Peptide volume in nm^3 in solution (free)
    eps_sp = eps_sp_pb;      % Peptide shape parameter (from scaled particle theory)
    am_amp = am_pb;          % Peptide atomic mass kg/mol
elseif pg1_flag
    Q = Q_pg1;                % Charge # for peptides
    dA = dA_pg1;              % Change in total lattice area in nm^2 per peptide bound (known to be approx 200-300 Angstrom^2 per peptide binding)
    H = H_pg1;                % Hydrophobic energy gain in units of kB T per peptide bound
    vp = vp_pg1;              % Peptide volume in nm^3 in solution (free)
    eps_sp = eps_sp_pg1;      % Peptide shape parameter (from scaled particle theory)
    am_amp = am_pg1;          % Peptide atomic mass kg/mol
else
    Q = Q_mgh2;                % Charge # for peptides
    dA = dA_mgh2;              % Change in total lattice area in nm^2 per peptide bound (known to be approx 200-300 Angstrom^2 per peptide binding)
    H = H_mgh2;                % Hydrophobic energy gain in units of kB T per peptide bound
    vp = vp_mgh2;              % Peptide volume in nm^3 in solution (free)
    eps_sp = eps_sp_mgh2;      % Peptide shape parameter (from scaled particle theory)
    am_amp = am_mgh2;          % Peptide atomic mass kg/mol
end

% ==========================
% IO
% ==========================
tag = 'Mg';
filetag = sprintf('data\\%s_na_%d_mM_amp_%1.1f_uM', tag, 1000*n1, 1e6*np);

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
    tmp = 0.6022*n2_array(n2_index);
    func = @(x) x/(tmp - x)^2 - kbval2;
    x0 = n2_array(n2_index)*0.01; % Na script had n2/2
    n2eff = n2_array(n2_index) - fzero(func, x0)/0.6022;
end

% ~~~~~~~~~~~~~~~~~~~~~~~
% declare these as globals using dummy variable (only n1 fixed)
n1eff_mg = n1eff(1);
% ~~~~~~~~~~~~~~~~~~~~~~~

% (3) Debye Length
function kappa = kappa(j)
    kappa = sqrt(n1eff_mg + 3*n2eff(j)) / 0.3081; 
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
    % molarity to SI conversion (mol/L to #/nm^3 via 6.022*10^23 #/mol * 1000 L/m^3 * (10^-9)^3 m^3/nm^3)
    % note 6.022*10^23 * 1000 * 10^-27 = 0.6022
    molarity_to_si = 0.6022;
    n1_si = n1 * molarity_to_si;
    n2_si = n2_array(i) * molarity_to_si;
    np_si = np * molarity_to_si;
    % repeated variables
    kap = kappa(i);
    delt = Delta(i);
    m1 = M1(Q, kap, aa);
    mp = Mp(Q, kap, aa);
    expand_factor = 1 + Q*ions(3);    % N0_tilde = N0 + Q*Np = N0*expand_factor for MGH2
    % compute components
    flps_elec = delt*lb*( ...
        (pi/kap - m1/2) * (ions(1) + 2*ions(2) + Q*ions(3) - 1)^2 / (expand_factor * aa^2) ...
        - (mp - m1)/2 * (Q*ions(3)*(ions(1) + 2*ions(2) + Q*ions(3))) / (expand_factor * aa^2) ...
        - (ions(1)/d1 + 2*ions(2)/d2 + Q*ions(3)/dp) ...
        - 2*SumC(10, kap, aa) * ions(2)*(1 - ions(1) - ions(2)) / expand_factor);
    flps_entr = (ions(1)*log(ions(1)/(n1_si*v1)) + ions(2)*log(ions(2)/(n2_si*v2)) + ions(3)*log(ions(3)/np_si*vp) ...
        + (1 - ions(1) - ions(2)) * log(1 - ions(1) - ions(2)) ...
        - ions(3)*(eps_sp + 1 - log(Q)) ...
        + eps_sp*expand_factor^2/Q);
    flps_chem = (ions(1)*0.5*lb*((delt-1)/d1 + kap/(1 + kap*r1)) ...
        + ions(2)*0.5*4*lb*((delt-1)/d2 + kap/(1 + kap*r2)) ...
        + ions(3)*0.5*Q*lb*((delt-1)*(1/dp + (mp-m1)/aa^2) + kap/(1 + kap*r1)));
    flps_mech = (1/2)*kA*lattc^2*(ions(3)*Q)^2;
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
len = length(n2_array);

% Plots various quantities and saves them into individual figures
% (1) plots fractional site occupancy
% (2) plots fractional charge occupancy
% (3) plots free energy
% (4) plots differntial tension
% (5) plots purely mechanical tension
function plot_all_results = plot_all_results()
    conversion_factor = 4.114;  % for 1 k_B*T/nm^2 = 4.114 mN/m
    xlist = 10^3*n2_array;
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
    
    % Store data as csv
    M = real([n2_array', ylist_1', ylist_2', ylist_2x2', ylist_p', ylist_freep', ylist_tensionp', ylist_tensionmech']);
    filename = sprintf('data\\Mg_%s_data_na_%d_mM_amp_%1.1f_uM.txt', tag, 1000*n1, 1e6*np);
    fid = fopen(filename, 'w');
    col_id = sprintf('_na_%d_amp_%1.1f_%s', 1000*n1, 1e6*np, tag);
    col_header = sprintf('Mg,n1%s,n2%s,2_n2%s,Q_np%s,Free%s,TenDiff%s,TenMech%s\n', col_id, col_id, col_id, col_id, col_id, col_id, col_id);
    fprintf(fid, col_header);
    fclose(fid)
    dlmwrite(filename,M,'-append','precision',4)
    get_custom_data = M;
    
    % Plot fractional site occupancy
    h1 = figure;
    plot(xlist,ylist_1,':bs',xlist,ylist_2,':ks',xlist,ylist_p,':rs')
    xmax = 1.01*n2_array(len)*10^3;
    ymax = 1.01;
    axis([0,xmax,-0.01,ymax]) %xmin xmax ymin ymax
    title(['Fractional Site Occupancy vs [Mg^{2+}]; [Na^{+}] =  ', num2str(1000*n1), ' mM, [AMP] = ', num2str(np*10^6), '\muM'])
    xlabel('[Mg^{2+}] (mM)')
    ylabel('Fractional Site Occupancy')
    legend('N_{1} / N_{0} (Na^{+})','N_{2} / N_{0} (Mg^{2+})','Q*N_{p} / N_{0} (AMP)','Location','northeast')
    plotname = sprintf('%s_frac_site.jpg', filetag);
    saveas(h1, plotname)
    % Plot fractional charge occupancy
    h2 = figure;
    plot(xlist,ylist_1,':bs',xlist,ylist_2x2,':ks',xlist,ylist_p,':rs')
    xmax = 1.01*n2_array(len)*10^3;
    ymax = 1.01;
    axis([0,xmax,-0.01,ymax]) %xmin xmax ymin ymax
    title(['Fractional Charge Occupancy vs [Mg^{2+}]; [Na^{+}] =  ', num2str(1000*n1), ' mM, [AMP] = ', num2str(np*10^6), '\muM'])
    xlabel('[Mg^{2+}] (mM)')
    ylabel('Fractional Charge Occupancy')
    legend('N_{1} / N_{0} (Na^{+})','2*N_{2} / N_{0} (Mg^{2+})','Q*N_{p} / N_{0} (AMP)','Location','northeast')
    plotname = sprintf('%s_frac_charge.jpg', filetag);
    saveas(h2, plotname)
    % Plot free energy
    h3 = figure;
    plot(xlist,ylist_freep,':bs')
    %axis([0,160,-0.5,0.5]) set(gca,'XTickLabel',[0:20:140]) % May need to
    %remove this (could interfere with later plots)
    title(['Free Energy vs [AMP]; [Na^{+}] =  ', num2str(1000*n1),' mM, [AMP] = ', num2str(np*10^6), '\muM'])
    xlabel('[Mg^{2+}] (mM)')
    ylabel('Free Energy (k_{B} T)')
    plotname = sprintf('%s_freep.jpg', filetag);
    saveas(h3, plotname)
    % Plot differential tension
    h4 = figure;
    plot(xlist,ylist_tensionp,':bs')
    %axis([0,10^3*n2_array(len),-0.6,1.0])
    title(['\Delta \Pi vs [Mg^{2+}]; [Na^{+}] =  ', num2str(1000*n1),' mM, [AMP] = ', num2str(np*10^6), '\muM'])
    xlabel('[Mg^{2+}] (mM)')
    ylabel('\Delta \Pi Differential (k_{B} T / nm^2)')
    plotname = sprintf('%s_tensionp.jpg', filetag);
    saveas(h4, plotname)
    % Plot mechanical tension
    h5 = figure;
    plot(xlist,ylist_tensionmech,':bs')
    axis([0,10^3*n2_array(len),0,10.0])
    title(['\Delta \Pi (Mechanical) vs [AMP]; [Na^{+}] =  ', num2str(1000*n1),' mM, [AMP] = ', num2str(np*10^6), '\muM'])
    xlabel('[Mg^{2+}] (mM)')
    ylabel('\Delta \Pi Mechanical (k_{B} T / nm^2)')
    plotname = sprintf('%s_tensionmech.jpg', filetag);
    saveas(h5, plotname)
end

% ========================== Save Data ==========================

% Plots various quantities and saves them into individual figures
% (1) stores fractional site occupancy
% (2) stores fractional charge occupancy
% (3) stores free energy
% (4) stores differential tension
% (5) stores mechanical tension
function get_custom_data = get_custom_data()
    conversion_factor = 4.114;  % for 1 k_B*T/nm^2 = 4.114 mN/m
    xlist = 10^3*n2_array;
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

end

function get_tensionmech_data = get_tensionmech_data()
    conversion_factor = 4.114
    ylist_tensionmech = zeros(1,len);
    for i = 1:len
        ylist_tensionmech(i) = TensionMech(i,lattc)*conversion_factor;
    end
    M = real([np_array', ylist_tensionmech'])
    filename = sprintf('%s.txt', filetag);
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
    %plot_all_results()
    %plot_frac_site_AMP() 
    %plot_frac_charge_AMP() 
    %plot_freep_AMP()
    %plot_tensionp_AMP()
    %plot_tensionmech_AMP()
    %get_line() get_tension_data() n1 n2*1000 np_array*1000*1000
    outputs = get_custom_data();
end

main()

end

%{
TODO 1. fix mupccc, mu1c
%}