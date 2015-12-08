function outputs = compute_amp(n1,n2)

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

np_back = (1*10^-7)*[1:9,10:5:100];
np_front = [10^(-9)*[1,5], (1*10^-8)*[1:9]];
np_array = [np_front, np_back]; % 0.01 microM to 10 microM

%n1 = 0.150
%n2 = 10^(-3)*5.0
%np_array = 10^(-6)*[0.15]

%(*np = {0.0000001,0.0000002,0.0000003,0.0000004,0.0000005,0.0000006,0.0000007,0.0000008,0.0000009,0.000001,0.000002,0.000003,0.000004,0.000005,0.000006,0.000007,0.000008,0.000009,0.00001,0.00002,0.00003,0.00004,0.00005,0.00006,0.00007,0.00008,0.00009,0.0001,0.0002,0.0003,0.0004,0.0005,0.0006,0.0007,0.0008,0.0009,0.001,0.002,0.003,0.004,0.005,0.006,0.007,0.008,0.009,0.01,0.02,0.03,0.04,0.05,0.06,0.07,0.08,0.09,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1,2,3,4,5,6,7,8,9,10};*)
%(*np = (5*10^-8)*{0.001, 0.01,0.05, 0.1,0.5,1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20};*)
pts_np = length(np_array);

% ==========================
% Global constants
% ==========================

d1 = 0.3;              % Binding site separation for NA+ in nm
d2 = 0.25;             % Binding site separation for Mg2+ in nm
dp = 0.4;              % Binding site separation for peptides in nm

partition = 1;
lattc = 0.8;           % Assumed Lattice Constant, this is a0 in nm
del = 0.001;           % del multiplied to x to get \[Delta]x...del canot be too much smaller or bigger than x. If its too small, the approx of the numerator / that for denumerator can vary greatly...
%(*Also, not mentioned here is that we have r1=0.34nm, r2=0.43nm where ri is the hydration radius for ions*)

lipidB = 137.5;
lipidG = 12;              % is \Gamma
lipidT = 0.4;             % is \Tau
nu = 0.378*6;             % is \Nu
lh = 0.2;

% could be used to fit initial guess problem
init1=[0.000994, 0.001809, 0.002605, 0.003418, 0.004259, 0.005130, 0.006030, 0.006959, 0.007915, 0.008895];
init2=[0.485380, 0.478778, 0.479529, 0.479266, 0.478510, 0.477475, 0.476264, 0.474937, 0.473528, 0.472059];
temp = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0];

epsilon = 1/40;   % eps=80 for water and 2 for membrane, so epsilon = eta = eps lipid / eps water
lb = 0.69625;     % Bjerrum length
d = 4.0;          % r_min for Kb

% ==========================
% New constants
% ==========================
A0 = 1.2e9;            % Initial surface area in nm^2
cJ  = 4.114*10^(-18);  % conversion factor for mJ to kB T 
cA = 10^(18);          % convert m^2 to nm^2
k0 = 120; %120 %100;
kA = k0/(cA*cJ);       % Area compression modulus in mN/m = mJ/m^2, with Joules converted to kbT

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
% declare these as globals using dummy variable
n1eff_amp = n1eff(1);
n2eff_amp = n2eff(1);
% ~~~~~~~~~~~~~~~~~~~~~~~

% (3) Debye Length
% COMMENTS: *0.3081^2 = D*episilon_ 0*kT/2z^2*e^2 ? *)   (*definition from Roham p37*)
function kappa = kappa(j)
    kappa = sqrt(n1eff_amp + 3*n2eff_amp) / 0.3081; 
end

% (4) Sigma ???
% COMMENTS: ??? what is this
function sigma = sigma(sig00, sig11, sig22)
    sigma = sig00 - sig11 - 2*sig22; 
end

% (5) Dielectric Discontinuity
function Delta = Delta(i)
    kap = kappa(i);
    Delta = (epsilon + kap*d) / (2*epsilon + kap*d)*2;
end

% (7) Script M Correction Functions
% COMMENTS:

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

% ???
function M2 = M2(p, kappa, a)
    tmp = a/2;
    integrand = @(x,y) exp(-kappa.*sqrt(x.^2 + y.^2)) ./ sqrt(x.^2 + y.^2);
    xmin = -tmp; xmax = tmp;
    ymin = -tmp; ymax = tmp;
    M2 = integral2(integrand,xmin,xmax,ymin,ymax);
end

% (6) Sum C, the lateral correlation function
% INPUTS: m is ~ 10, # of grids per side to integrate)
%         kappa is debye length
%         lc is lattice constant, may not be a constant
% COMMENTS:
function SumC = SumC(m,kappa,lc)
    summ = 0;
    for i = 1:(m*partition)
        for j = 0:i
            tempor = 1 / ((lc/partition)*sqrt(i^2 + j^2))*exp(-kappa*sqrt(i^2+j^2)*(lc/partition))*(-1)^(i+j);
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

% (7) Bulk chemical potentials
% COMMENTS: Chemical potential in bulk should be log[n1 * v0], but log[v0] cancelled with the one in mu_condensed
%           Should we include the effect of tthe 2nd part of bulk potentials to K_b (which we used to find n1eff,n2eff from n1,n2) ???
% NOTE: The N1, N2, Np were initally used as input... still there
% Some constants
Ct = 6*(10^6)*(10^-21);   % Cell Concentration = 6E4/mL, from Azadeh's thesis          % For cell concentration effect which we argued is insignificant for general cell conc
N0 = 1.875*(10^7);        % N0 = A/a^2, A= cell area = 1.2E9 from Azadeh's thesis      % For cell concentration effect which we argued is insignificant for general cell conc
% NOTE: I think the N0 value is wrong and should be N0 since we have A0 = 1.2e9 nm^2 and a0 = 0.8 nm

function mu1b = mu1b(i, N1)  % condp Verified
    mu1b = log(4/3 * pi * (0.34)^3*(0.6022*n1eff_amp)) - 0.5*lb*(kappa(i)/(1+0.36*kappa(i)) + (Delta(i)-1)/d1);
end

function mu2b = mu2b(i, N2)  % condp Verified
    mu2b = log(4/3 * pi * (0.43)^3*(0.6022*n2eff_amp)) - 2*lb*(kappa(i)/(1+0.43*kappa(i)) + (Delta(i)-1)/d2);
end

function mupb = mupb(i, Np)  % condp Verified
    mupb = log(vp * (0.6022*np_array(i))) - 2*lb*(kappa(i)/(1+0.36*kappa(i)) + (Delta(i)-1)/dp);
end

% (8) Surface chemical potentials
% COMMENTS: 

function mu1c = mu1c(i, a, sigma1, d11)
    kap = kappa(i);
    mu1c = Delta(i) * lb * (-2*pi/kap*(1/a^2-sigma1) + (1/a^2-sigma1) * M1(Q, kap, a) - 1/d1) ...  %************POSSIBLE ERROR************ for mg/amp code, he had M1(kap, a)
        + log(sigma1/(1/a^2-sigma1));
end

function mu1cc = mu1cc(i, a, N1, N2, d11)
    kap = kappa(i);
    mu1cc = Delta(i) * lb * (-2*(N2 + 0.5)*SumC(10, kap, a) + 2*pi / kap * (N1 + 2*N2) /a^2 ...
        - (N1 + 2*N2)/a^2 * M1(Q, kap, a) - 1/d1) + log(N1/(0.5-N1-N2));
end

function mu2cc = mu2cc(i, a, N1, N2, d22)
    kap = kappa(i);
    mu2cc = Delta(i) * lb * (-2*(N1 + 2*N2)*SumC(10, kap, a) + 4*pi/kap * (N1 + 2*N2)/a^2 ...
        - 2*(N1 + 2*N2)/a^2 * M1(Q, kap, a) - 2/d2) + log((N2+0.5)/(0.5-N1-N2));
end

function mu1ccc = mu1ccc(i, a, N1, N2, Np)  % condp Verified
    kap = kappa(i);
    m1 = M1(Q, kap, a);
    mu1ccc = Delta(i) * lb * (-2*(N2+0.5)*SumC(10, kap, a) + 2*pi/kap * (N1 + 2*N2 + Q*Np)/a^2 ...
        - ((N1 + 2*N2) * m1 + Q*Np*(m1 + Mp(Q, kap, a))/2)/a^2 - 1/d1) + log( N1/(0.5-N1-N2-Q*Np));
end

function mu2ccc = mu2ccc(i, a, N1, N2, Np)  % condp Verified
    kap = kappa(i);
    m1 = M1(Q, kap, a);
    mu2ccc = Delta(i) * lb * (-2*(N1+2*N2+Q*Np)*SumC(10,kap,a) + 4*pi/kap*(N1 + 2*N2 + Q*Np)/a^2 ...
        - 2 * ((N1 + 2*N2) * m1 + Q*Np*(m1 + Mp(Q,kap,a))/2)/a^2 - 2/d2) + log((N2+0.5)/(0.5-N1-N2-Q*Np));
end

function mupccc = mupccc(i, a, N1, N2, Np)  % condp Verified
    kap = kappa(i);
    delt = Delta(i);
    m1 = M1(Q,kap,a);
    mp = Mp(Q,kap,a);
    % Na version
    %mupccc = delt * lb * (-2*Q*(N2+0.5)*SumC(10,kap,a) + Q*2*pi / kap * (N1 + 2*N2 + Q*Np)/a^2 ...
    %    - Q * (-1*m1+(N1+2*N2+1)*(m1 + mp)/2+Q*Np*mp ...
    %    - 0.5*((mp - m1)-1/delt*(a^2/lattc^2)*(Mp(Q, kap, lattc) - M1(Q, kap, lattc))))/a^2 ...
    %    - Q/dp)+log(Np*(1-Q*Np)^(Q-1)/(0.5-N1-N2-Q*Np)^Q)-(eps_sp+1-log(Q))+1/(1-Q*Np)+(eps_sp)/(1-Q*Np)^2 ...
    %    + H + (kA*dA^2/2)*Np/(lattc)^2; %this term is the hydrophobic + mech energy
    % Mg Version
    %mupccc = delt * lb * (-2*Q*(N2+0.5)*SumC(10,kap,a) + Q*2*pi / kap * (N1 + 2*N2 + Q*Np)/a^2 ...
    %    - Q * (-1*m1+(N1+2*N2+1)*(m1 + mp)/2+Q*Np*mp ... 
    %    - 0.5*(1-1/delt)*(mp-m1))/a^2 ...     % THIS LINE DIFFERS from na
    %    - Q/dp)+log(Np*(1-Q*Np)^(Q-1)/(0.5-N1-N2-Q*Np)^Q)-(eps_sp+1-log(Q))+1/(1-Q*Np)+(eps_sp)/(1-Q*Np)^2 ...
    %    + H + (kA*dA2/2)*Np/(lattc)^2; %this term is the hydrophobic + mech energy
    % AMP version - original
    %mupccc = delt * lb * (-2*Q*(N2+0.5)*SumC(10,kap,a) + Q*2*pi / kap * (N1 + 2*N2 + Q*Np)/a^2 ...
    %    - Q * (-1*m1+(N1+2*N2+1)*(m1 + mp)/2+Q*Np*mp ...
    %    - 0.5*((mp - m1)-1/delt*(a^2/lattc^2)*(Mp(Q, kap, lattc) - M1(Q, kap, lattc))))/a^2 ...
    %    - Q/dp)+log(Np*(1-Q*Np)^(Q-1)/(0.5-N1-N2-Q*Np)^Q) ...%- (eps_sp+1-log(Q))+1/(1-Q*Np)+(eps_sp)/(1-Q*Np)^2 ... % THIS LINE DIFFERS from na
    %    + H + (kA*dA^2/2)*Np/(lattc)^2; %this term is the hydrophobic + mech energy
    % AMP version - with area exclusion (same as Na)
    mupccc = delt * lb * (-2*Q*(N2+0.5)*SumC(10,kap,a) + Q*2*pi / kap * (N1 + 2*N2 + Q*Np)/a^2 ...
        - Q * (-1*m1+(N1+2*N2+1)*(m1 + mp)/2+Q*Np*mp ...
        - 0.5*((mp - m1)-1/delt*(a^2/lattc^2)*(Mp(Q, kap, lattc) - M1(Q, kap, lattc))))/a^2 ...
        - Q/dp)+log(Np*(1-Q*Np)^(Q-1)/(0.5-N1-N2-Q*Np)^Q) - (eps_sp+1-log(Q))+1/(1-Q*Np)+(eps_sp)/(1-Q*Np)^2 ...
        + H + (kA*dA^2/2)*Np/(lattc)^2; %this term is the hydrophobic + mech energy
end

function mu1ccm = mu1ccm(i, a, N1, Np)
    kap = kappa(i);
    m1 = M1(Q, kap,a);
    mu1ccm = Delta(i) * lb * (2*pi / kap * (N1 + Q*Np -1)/a^2 - 1 *((N1 - 1) * m1 ...
        + Q*Np*(m1 + Mp(Q, kap, a))/2)/a^2  - 1/d1) + log(N1/(1-N1-Q*Np));
end

function mupccm = mupccm(i, a, N1, Np)
    kap = kappa(i);
    m1 = M1(Q, kap,a);
    mp = Mp(Q,kap,a);
    mupccm = Delta(i) * lb * (2*Q*pi / kap * (N1 + Q*Np -1)/a^2 - Q *( -1*m1 ...
        + N1*(m1 + mp)/2+Q*Np*mp- 0.5*((mp ...
        - m1) - 1/Delta(i)*(a^2/lattc^2)*(Mp(Q,kap,lattc)-M1(Q, kap, lattc))))/a^2 ...
        - Q/dp) + log(Np*(1-Q*Np)^(Q-1)/(1-N1-Q*Np)^Q)-(eps_sp+1-log(Q))+1/(1-Q*Np)+(eps_sp)/(1-Q*Np)^2 ...
        + H + (kA*dA^2/2)*Np/(lattc)^2; %this term is the hydrophobic + mech energy
end

%{
% (9) First "cond" thing
% COMMENTS: this may need to be tweaked
function cond = cond(i,aa,d1)
    func = @(x) mu1b(i,x) - mu1c(i,aa,x,d1);
    x0 = 1; % was 0.1 before, but now I can use fzero and get valid results for plot2()
    ob = fzero(func, x0);
    a = ob;
    res = a;
    cond = res;
end
%}

% (9) First "cond" thing
% COMMENTS: this fsolve version causes "equation solved" messages to
% appear, and gives a bad initial/second value (in plot2) i think based on n1 array
function cond = cond(i,aa,d1)
    function F = nle(x)
        F = mu1b(i,x) - mu1c(i, aa, x, d1);
    end
    x0 = 0.1;
    ob = fsolve(@nle, x0);
    a = ob;
    res = a;
    cond = res;
end


% (10) Block of plot stuff
% COMMENTS: this WILL need to be tweaked

% Fig10, LPS_Ma.pdf    

function sigma1r1 = sigma1r1(i, aa)
    sigma1r1 = cond(i, aa,d1)*aa^2;
end

function sigma1eff = sigma1eff(i, aa)
    sigma1eff = (1/aa^2 - cond(i, aa,d1));  % Same condition as Fig10c in LPS_Ma.pdf
end

function sigma1reff = sigma1reff(i, aa)
    sigma1reff = 1 - cond(i, aa,d1)*aa^2;
end

function cond2 = cond2(i,aa,d1,d2) 
    function F = nle(x)
        F = [mu1b(i,x(1)) - mu1cc(i, aa, x(1), x(2), d1);
            mu2b(i,x(2)) - mu2cc(i, aa, x(1), x(2), d2)];
    end
    % Na script guesses
    %x0 = [0.2*sqrt( n1(i) );
    %    0.2*sqrt( n1(i) )];
    % Mg and AMP script guesses
    x0 = [0.1;
        -0.1];
    ob = fsolve(@nle,x0);
    a = ob(1);
    b = ob(2);
    res = ob;
    cond2 = res;
end

function sigma2r1 = sigma2r1(i, aa)
    tmp = cond2(i,aa,d1,d2);
    sigma2r1 = tmp(1);
end

function sigma2r2 = sigma2r2(i, aa)
    tmp = cond2(i,aa,d1,d2);
    sigma2r2 = 0.5 + tmp(2);
end

function sigma2r2t = sigma2r2t(i, aa)
    tmp = cond2(i,aa,d1,d2);
    sigma2r2t = tmp(2);
end

function negsigma2reff = negsigma2reff(i, aa)
    tmp = cond2(i,aa,d1,d2);
    negsigma2reff = tmp(1) + 2*tmp(2);  % Same condition as Fig10c in LPS_Ma.pdf
end

% piece-wise functions used to solve initial guess problem for mdp case
%testfcn1[i_] :=  If[i<30, 0.2-0.2*55*np_array(i).5, 0.01]; 
%testfcn2[i_] :=  If[i<30, -0.15-0.2*90*np_array(i)^0.5, -0.49]; 
%testfcnp[i_] :=  If[i<30, 0.2*75*np_array(i)^0.5, 60*np_array(i)^0.5]; 

function condp = condp(i,aa)
    function F = nle(x)
        F = [mu1b(i, x(1)) - mu1ccc(i, a_Np(aa,dA,x(3)), x(1), x(2), x(3));
            mu2b(i, x(2)) - mu2ccc(i, a_Np(aa,dA,x(3)), x(1), x(2), x(3));
            mupb(i, x(3)) - mupccc(i, a_Np(aa,dA,x(3)), x(1), x(2), x(3))];
        %F = [mu1b(i, x(1)) - mu1ccc(i, aa, x(1), x(2), x(3));
        %    mu2b(i, x(2)) - mu2ccc(i, aa, x(1), x(2), x(3));
        %    mupb(i, x(3)) - mupccc(i, aa, x(1), x(2), x(3))];
    end
    % Na script guesses
    %x0 = [n1(i)^0.5; 
    %     -0.01-n1(i)^0.5;
    %     0.01-0*0.3*n1(i)^0.5]; % LAM has 0*0.3*n1(i)^0.5].. why the 0* ?
    % Mg script guesses
    %x0 = [0.6-3*4.5*n2_array(i)^0.5;
    %     -0.5+3*3*n2_array(i)^0.5;
    %     0.05 - n2_array(i)^0.5];
    % AMP script guesses
    x0 = [0.2-0.2*55*np_array(i)^0.5;
         -0.15-0.2*90*np_array(i)^0.5;
         0.2*75*np_array(i)^0.5];
    ob = fsolve(@nle,x0);
    a = ob(1);
    b = ob(2);
    c = ob(3);
    res = ob;
    condp = res;
end

function sigmapr1 = sigmapr1(i, aa)
    tmp_p = condp(i, aa);
    sigmapr1 = tmp_p(1);
end

function sigmapr2t = sigmapr2t(i, aa)
    tmp_p = condp(i, aa);
    sigmapr2t = tmp_p(2);
end

function sigmapr2 = sigmapr2(i, aa)
    tmp_p = condp(i, aa);
    sigmapr2 = 0.5 + tmp_p(2);
end

function sigmaprp = sigmaprp(i, aa)
    tmp_p = condp(i, aa);
    sigmaprp = tmp_p(3);
end

function negsigmapreff = negsigmapreff(i, aa)
    tmp_p = condp(i, aa);
    negsigmapreff = tmp_p(1) + 2*tmp_p(2) + Q*tmp_p(3);  % (*Same condiion as Fig10c in LPS_Ma.pdf)*)
end

% this function was left commented out:  
% (its just negsigmapreff(i,aa) output divided by aa^2)
function sigmaeffp = sigmaeffp(i,aa)  % Translation Verified
    tmp_p = conp(i,aa);
    sigmaeffp = (tmp_p(1) + 2*tmp_p(2) + Q*tmp_p(3) ) / aa^2;
end

function condmp = condmp(i,aa)
    function F = nle(x)
        F = [mu1b(i,x(1)) - mu1ccm(i, aa, x(1), x(2)); % variables appropriately removed in the mg script for the mu1b, mupb calls
            mupb(i,x(2)) - mupccm(i, aa, x(1), x(2))];
    end
    % Na script guesses
    %x0 = [10*n1(i)^2;
    %     0.25 - 6*n1(i)^2];
    % Mg and AMP script guesses
    x0 = [0.001;
         0.24];
    ob = fsolve(@nle,x0);
    a = ob(1);
    b = ob(2);
    res = ob;
    condmp = res;
end

function sigmampr1 = sigmampr1(i, aa)
    tmp_mp = condmp(i, aa);
    sigmampr1 = tmp_mp(1);
end

function sigmamprp = sigmamprp(i, aa)
    tmp_mp = condmp(i, aa);
    sigmamprp = tmp_mp(2);
end

function negsigmampreff = negsigmampreff(i, aa)
    tmp_mp = condmp(i, aa);
    tmp_p = condp(i, aa);
    negsigmampreff = tmp_mp(1) + Q*tmp_p(2);   % MATT: not sure if its condmp..(2) or condp..(2) but he had the latter, i think it should be the first
end

function cond2test = cond2test(i,aa,empty)
    function F = nle(x)
        tmp = cond2(i, aa, d1, x);
        F = 1/2 - tmp(i) - tmp(2) - empty;
    end
    x0 = 0.3;
    ob = fsolve(@nle, x0);
    a = ob;
    res = ob;
    cond2test = res;
end

% (11) The free energy expressions
% COMMENTS: why the modules again? is it for plotting?

function FreeEnergy = FreeEnergy(i, aa)
    % repeated variables
    kap = kappa(i);
    delt = Delta(i);
    ion = cond(i, aa, d1);                 % Here "ion" <=> sigma
    % energy components
    es = delt * pi * lb / kap * (1/aa^2 - ion)^2;
    entr = ion * log(ion*aa^2) + (1/aa^2-ion) * log(1 - ion*aa^2) - ion * mu1b(i, ion*aa^2); % idk if last input is needed... ion*aa^2
    corr = -delt * lb * ((1/aa^2 - ion)^2 * M1(Q, kap, aa)/2 + ion/d1);
    res = aa^2*(es + entr + corr);  % sum components
    FreeEnergy = res;
end

function FreeEnergy2 = FreeEnergy2(i, aa)
    % repeated variables
    kap = kappa(i);
    delt = Delta(i);
    ion = cond2(i, aa, d1, d2);                 % Here "ion" <=> Ni, N2 tilted
    % energy components
    es = delt * pi * lb / kap * (ion(1) + 2*ion(2))^2/aa^2;
    entr = ion(1) * log(ion(1)) + (ion(2)+0.5) * log((ion(2)+0.5)) + (0.5-ion(1)-ion(2)) * log(0.5-ion(1)-ion(2)) - ion(1) * mu1b(i,ion(1)) - (ion(2)+0.5) * mu2b(i,ion(2));
    corr = delt * lb * (-(ion(1)+2*ion(2))^2/aa^2 * M1(Q, kap, aa)/2 + 2*(ion(2)+0.5)*(0.5-ion(1)-ion(2)) * SumC(10, kap, aa) +(- ion(1)*1/d1 - (ion(2)+0.5)*2/d2));
    res = es + entr + corr;  % sum components
    FreeEnergy2 = res;
end


function FreeEnergyp = FreeEnergyp(i, aa) % NEW: Correction terms
    % repeated variables
    kap = kappa(i);
    delt = Delta(i);
    m1 = M1(Q, kap, aa);
    mp = Mp(Q, kap, aa);
    m1_lattc = M1(Q, kap, lattc);
    mp_lattc = Mp(Q, kap, lattc);
    ion = condp(i, aa);
    
    % F_LPS components
    es = delt * pi * lb / kap * (ion(1)+2*ion(2)+Q*ion(3))^2/aa^2;
    entr = (ion(1) * log(ion(1)) + (ion(2)+0.5) * log((ion(2)+0.5)) + ion(3) * log(ion(3)) + (0.5-ion(1)-ion(2)-Q*ion(3)) * log(0.5-ion(1)-ion(2)-Q*ion(3))) + (-ion(1) * mu1b(i,ion(1)) - (ion(2)+0.5) * mu2b(i,ion(2)) - ion(3) * mupb(i,ion(3)));
    
    % (THIS IS WITH AREA EXLUSION) 
    entr2 = ((1-Q)/Q) * (1 - Q*ion(3)) * log(1 - Q*ion(3))-ion(3)*(eps_sp+1-log(Q)) -log(1-Q*ion(3))/Q+(eps_sp/Q)/(1-Q*ion(3));
    
    % (THIS IS WITHOUT AREA EXCLUSION - currently disabled)
    %entr2 = ((1-Q)/Q) * (1 - Q*ion(3)) * log(1 - Q*ion(3)); %-ion(3)*(eps_sp+1-log(Q)) -log(1-Q*ion(3))/Q+(eps_sp/Q)/(1-Q*ion(3));
    
    % this was commented out in AMP original, probably depricated: (*corr = \[CapitalDelta][i] * lb * ((-0.5*M1[Q, \[Kappa][i], aa]*((ion[[1]]+2*ion[[2]])^2) +M1[Q, \[Kappa][i], lattc]*Q*ion[[3]] - 0.5(M1[Q,\[Kappa][i], aa]+Mp[Q, \[Kappa][i], aa])*(Q*ion[[3]])*(ion[[1]]+2*ion[[2]]+1)-0.5M1[Q, \[Kappa][i], aa]*(Q*ion[[3]])^2)/aa^2      +2*(ion[[2]]+0.5)(0.5-ion[[1]]-ion[[2]]-Q*ion[[3]]) * SumC[10, \[Kappa][i], aa] + (-ion[[1]]/\[Delta]1 - 2(ion[[2]]+0.5)/\[Delta]2- Q*ion[[3]]/\[Delta]p)) - 0*ion[[1]]*Log[Kbtest[i, 1, \[Delta]1]/(4/3*Pi*\[Delta]1^3)] - 0*(ion[[2]]+0.5)*Log[Kbtest[i, 2, \[Delta]2]/(4/3*Pi*\[Delta]2^3)];*)  
    corr = delt * lb * ((-0.5*m1*((ion(1)+2*ion(2))^2) + m1*Q*ion(3) - 0.5*(m1 + mp)*(Q*ion(3))*(ion(1)+2*ion(2)+1)-0.5*mp*(Q*ion(3))^2  + 0.5*Q*ion(3)*((mp - m1)-1/delt*aa^2/lattc^2*(mp_lattc - m1_lattc)))/aa^2 + 2*(ion(2)+0.5)*(0.5-ion(1)-ion(2)-Q*ion(3)) * SumC(10, kap, aa) + (-ion(1)/d1 - 2*(ion(2)+0.5)/d2- Q*ion(3)/dp)); 
    mech = (1/4)*kA*(dA*ion(3)/lattc)^2;
    hydro = ion(3)*H;
    res = es + entr + corr + entr2 + hydro + mech;  % sum components
    FreeEnergyp = res;
end  

function FreeEnergymp = FreeEnergymp(i, aa)
    % repeated variables
    kap = kappa(i);
    delt = Delta(i);
    m1 = M1(Q, kap, aa);
    mp = Mp(Q, kap, aa);
    m1_lattc = M1(Q, kap, lattc);
    mp_lattc = Mp(Q, kap, lattc);
    ion = condmp(i, aa);
    % energy components
    es = delt * pi * lb / kap * (ion(1)+Q*ion(2)-1)^2/aa^2;
	entr = ion(1) * log(ion(1)) + ion(2) * log(ion(2)) + (1-ion(1)-Q*ion(2)) * log(1-ion(1)-Q*ion(2)) - ion(1) * mu1b(i,ion(1)) - ion(2) * mupb(i,ion(2));
	entr2 = ((1-Q)/Q) * (1 - Q*ion(2)) * log(1 - Q*ion(2)) -ion(2)*(eps_sp+1-log(Q))-log(1-Q*ion(2))/Q+(eps_sp/Q)/(1-Q*ion(2));
	corr = delt * lb * ((-0.5*m1*(ion(1)-1)^2 + m1*Q*ion(2) - 0.5*(m1 + mp)*(Q*ion(2))*(ion(1))-0.5*mp*(Q*ion(2))^2+ 0.5*Q*ion(2)*((mp-m1)-1/delt*aa^2/lattc^2*(mp_lattc - m1_lattc)))/aa^2  + (-ion(1)/d1 - Q*ion(2)/dp)) ;
    mech = (1/4)*kA*(dA*ion(2)/lattc)^2;  % should be ion(3)?
    hydro = ion(2)*H;  % should be ion(3)?
    res = es + entr + corr + entr2 + hydro + mech;  % sum components
    FreeEnergymp = res;
end

function FreeEnergy0 = FreeEnergy0(i, aa)
    es = Delta(i) * lb * 1 / aa^4 * (pi/ kappa(i));
    res = es;
    FreeEnergy0 = res;
end


% (13) Tension expressions
% COMMENTS: how to modify when a itself is a function, a(Np)
function Tension = Tension(i, a0)
    Tension = ( FreeEnergy(i,  sqrt((1-del)*a0^2))  -  FreeEnergy(i,  sqrt((1+del)*a0^2))) / (2*del*a0^2);
end

function Tension2 = Tension2(i, a0)
    Tension2 = ( FreeEnergy2(i, sqrt((1-del)*a0^2))  -  FreeEnergy2(i, sqrt((1+del)*a0^2))) / (2*del*a0^2);
end

function Tensionp = Tensionp(i, a0)
    Tensionp = ( FreeEnergyp(i, sqrt((1-del)*a0^2))  -  FreeEnergyp(i, sqrt((1+del)*a0^2))) / (2*del*a0^2);
end

function Tensionm = Tensionm(i, a0)
    Tensionm = ( FreeEnergymp(i,sqrt((1-del)*a0^2))  -  FreeEnergymp(i,sqrt((1+del)*a0^2))) / (2*del*a0^2);
end

% (14) Mechanical Tension (See latex formulation mechanical energy cost)
function TensionMech = TensionMech(i, a0)
    ion = condp(i, a0);
    sigmap = ion(3);
    TensionMech = (kA*dA/a0^2)*sigmap;
end

% ==========================
% Plotting
% ==========================
% Some reused plotting constants/lists
len = length(np_array);

% Plot Fractional ite Occupancy (AMP modified)
% Plots the fractional site occupancy based on [AMP] for N1, N2, QNp
function plot_frac_site_AMP = plot_frac_site_AMP()
    xlist = 10^6*np_array;
    ylist_1 = zeros(1,len);
    ylist_2 = zeros(1,len);
    ylist_p = zeros(1,len);
    for i = 1:len
        condp_tmp = condp(i,lattc);
        ylist_1(i) = condp_tmp(1); % from sigmapr1(i,lattc)
        ylist_2(i) = 0.5 + condp_tmp(2); % from sigmapr2(i,lattc)
        ylist_p(i) = Q*(condp_tmp(3)); % from sigmaprp(i,lattc)
    end
    h = figure
    plot(xlist,ylist_1,':bs',xlist,ylist_2,':ks',xlist,ylist_p,':rs')
    xmax = 1.01*np_array(len)*10^6;
    axis([0,xmax,-0.01,1.01]) %xmin xmax ymin ymax
    title(['Fractional Site Occupancy vs [AMP]; [Na^{+}] =  ', num2str(n1), ' M, [Mg^{2+}] = ', num2str(n2), ' M'])
    xlabel('[AMP] (\muM)')
    ylabel('Fractional Site Occupancy')
    legend('N_{1} / N_{0} (Na^{+})','N_{2} / N_{0} (Mg^{2+})','Q*N_{p} / N_{0} (AMP)','Location','northeast')
    saveas(h, 'amp_frac_site_MH.jpg')
end

% Plot Fractional Charge Occupancy (AMP modified)
% Plots the fractional CHARGE occupancy based on [AMP] for N1, N2, QNp
%
function plot_frac_charge_AMP = plot_frac_charge_AMP()
    xlist = 10^6*np_array;
    ylist_1 = zeros(1,len);
    ylist_2 = zeros(1,len);
    ylist_p = zeros(1,len);
    for i = 1:len
        condp_tmp = condp(i,lattc);
        ylist_1(i) = condp_tmp(1); % from sigmapr1(i,lattc)
        ylist_2(i) = 2*(0.5 + condp_tmp(2)); % from 2*sigmapr2(i,lattc)
        ylist_p(i) = Q*(condp_tmp(3)); % from sigmaprp(i,lattc)
    end
    h = figure
    plot(xlist,ylist_1,':bs',xlist,ylist_2,':ks',xlist,ylist_p,':rs')
    xmax = 1.01*np_array(len)*10^6;
    axis([0,xmax,-0.01,1.01]) %xmin xmax ymin ymax
    title(['Fractional Charge Occupancy vs [AMP]; [Na^{+}] =  ', num2str(n1), ' M, [Mg^{2+}] = ', num2str(n2), ' M'])
    xlabel('[AMP] (\muM)')
    ylabel('Fractional Charge Occupancy')
    legend('N_{1} / N_{0} (Na^{+})','2*N_{2} / N_{0} (Mg^{2+})','Q*N_{p} / N_{0} (AMP)','Location','northeast')
    saveas(h, 'amp_frac_charge_MH.jpg')
end

% Plot Free Energy
% Free energy to understand tension
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
    %axis([0,160,-0.5,0.5])
    %set(gca,'XTickLabel',[0:20:140]) % May need to remove this (could interfere with later plots)
    title(['Free Energy vs [AMP]; [Na^{+}] =  ', num2str(1000*n1),' mM, [Mg^{2+}] = ', num2str(1000*n2), ' mM'])
    xlabel('[AMP] (\muM)')
    ylabel('Free Energy (k_{B} T)')
    saveas(h, 'amp_freep_MH.jpg')
end

% Plot Tension (AMP Modified)
% Delta_pi (tension) with monovalent ions, divalent ions, and AMPs  Fig17, LPS_Ma.pdf 
%   [Table[{n1[[i]],
%   Tensionp[i,lattc]},{i,1,pts}], 
%   PlotRange->{{0,0.21},{-2,2}}, 
%   AxesLabel->{"[\!\(\*SuperscriptBox[\(Na\), \(+\)]\)](M)","\[CapitalDelta]\[pi] (\!\(\*SubscriptBox[\(k\), \(B\)]\)T/\!\(\*SuperscriptBox[\(nm\), \(2\)]\))"}, 
%   PlotLabel->" \[CapitalDelta]\[pi] vs [\!\(\*SuperscriptBox[\(Na\), \(+\)]\)], [\!\(\*SuperscriptBox[\(Mg\), \(\(2\)\(+\)\)]\)]=0.5mM, [AMP]=0.5\[Micro]M, revised"]*)
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

% ==========================
% Save Data
% ==========================

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
    %M = real([np_array', ylist_1', ylist_2', ylist_22', ylist_p', ylist_free_p', ylist_tensionp']);
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

% ==========================
% Main
% ==========================

function main = main()
    %plot_frac_site_AMP()
    %plot_frac_charge_AMP()
    %plot_freep_AMP()    
    plot_tensionp_AMP()
    plot_tensionmech_AMP()
    %get_line()
    %get_tension_data()
    %n1
    %n2*1000
    %np_array*1000*1000
    outputs = get_custom_data();
end

main()

end

%{
TODO
    1. fix mupccc, mu1c
%}