function OUTPUTS = MATLAB_Na(n2, np)

% n2 should be 0.0001, 0.0005, 0.001 (0.1 mM, 0.5 mM, 1mM)

% ==========================
% Commands
format long 
%tol = 1e-6
%options = optimoptions('fsolve','Display','iter','TolX',tol); % Option to display output
% ==========================

% ==========================
% Concentrations (in units of Molar)
% ==========================

% For [Na] Varying
n1 = [0.00001, 0.001, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1, 0.11, 0.12, 0.13, 0.14, 0.15, 0.16, 0.17, 0.18, 0.19, 0.2];

%n1 = [0.150]
%n2 = 5*10^(-3); % mg may be 0.1 mM, 0.5 mM, 1mM
%np = 0.615*10^(-6); % 10 micromolar

% ==========================
% Global constants
% ==========================

pts = 22;     % # of points in the graph
pts2 = 13;
ptsp = 12;
%(*a0 = sqrt[5];*) (*in the unit of nm^(-2)*)
%dlim = a0/sqrt[2];
%dlim2 = a0/2;

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
init1 = [0.001, 0.001, 0.001, 0.001, 0.01, 0.01, 0.01, 0.02, 0.02, 0.03, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3];
init2 = -0.5 + [0.001, 0.001, 0.001, 0.001, 0.01, 0.01, 0.01, 0.02, 0.02, 0.03, 0.05, 0.1, 0.2, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3];
initp = [0.245, 0.245, 0.245, 0.245, 0.24, 0.24, 0.24, 0.23, 0.23, 0.22, 0.2, 0.1, 0.05, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001];
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
k0 = 120 %100;
kA = k0/(cA*cJ);       % Area compression modulus in mN/m = mJ/m^2, with Joules converted to kbT

% ==========================
% AMP constants
% ==========================
Q = 4;                 % Charge # for peptides
dA = 3.0;              % Change in total lattice area in nm^2 per peptide bound (known to be approx 200-300 Angstrom^2 per peptide binding)
H = -12;               % Hydrophobic energy gain in units of kB T per peptide bound
vp = 2.5;              % Peptide volume in nm^3 in solution (free)
eps_sp = 25/(4*pi);    % Peptide shape parameter (from scaled particle theory)



% ==========================
% Declaring Functions
% ==========================

% (0) Lattice Constant Np Dependence

function a_Np = a_Np(a, da, Np)
    a_Np = sqrt(a^2 + da*Np);
end

% (1) Association Constant
function Kb = Kb(q,d)  % Translation Verified
    tmp = q.*lb;
    integrand = @(r) r.^2.*exp(tmp./r);            % dots cuz integral fn using matrices
    rmin = d;
    rmax = q.*lb./2;
    Kb = 4.*pi.*integral(integrand, rmin, rmax);
end

% (2) Free ion concentrations in solutions (1M = 0.6022 nm^-3):
% COMMENTS I understand that roots are being found in each of the three
% function calls, but I don't know what is being done with them? 
% What's ob? a? res??? why is x being called undeclared? is x just 0?

% function handle speedups
kbval1_1 = Kb(1, 0.34);
kbval1_2 = log(Kb(1, d1));
kbval2 = Kb(2, 0.43);

% (A) Free Na+ in soln
function n1eff = n1eff(jj)  % Translation Verified
    ref = n1(jj);
    tmp = 0.6022*ref;
    func = @(x) x/(tmp - x)^2 - kbval1_1;
    x0 = ref*0.01;
    n1eff = ref - fzero(func, x0)/0.6022;
end

% (B)
function n1eff2 = n1eff2(jjj)  % Translation BROKEN (this gives NaNs)
    tmp = 0.6022 * n1eff(jjj);
    func = @(x) log(x/(1-x)/tmp) - kbval1_2;
    x0 = 0.01;
    n1eff2 = fzero(func, x0);
end

% (C) Free Mg2+ in soln
function n2eff = n2eff()  % Translation Verified
    tmp = 0.6022*n2;
    func = @(x) x/(tmp - x)^2 - kbval2;
    x0 = n2/2;
    n2eff = n2 - fzero(func, x0)/0.6022;
end

%{
% (A2) Free Na+ in soln
function n1eff = n1eff(jj)   % using this instead of A1 breaks things
    ref = n1(jj);
    tmp = 0.6022*ref;
    function F = nle(x)
        F = x/(tmp - x)^2 - kbval1_1;
    end
    x0 = ref; %ref*0.01;
    n1eff = ref - fsolve(@nle, x0)/0.6022;
    %ob = fzero(func, x0);
    %a = ob;
    %n1eff = ref - a/0.6022;
end

% (B2)
function n1eff2 = n1eff2(jjj)     %using these instead of B,C doesnt change things up to plot2()
    tmp = 0.6022 * n1eff(jjj);
    function F = nle(x)
        F = log(x*tmp/(1-x)) - kbval1_2;
    end
    x0 = 0.01;
    n1eff2 = fsolve(@nle, x0);
    %ob = fzero(func, x0);
    %a = ob;
    %n1eff2 = a;
end

% (C2) Free Mg2+ in soln
function n2eff = n2eff() 
    tmp = 0.6022*n2;
    function F = nle(x)
        F = x/(tmp - x)^2 - kbval2;
    end
    x0 = n2/2;
    n2eff = n2 - fsolve(@nle, x0)/0.6022;
    %ob = fzero(func, x0);
    %a = ob;
    %n2eff = n2-a/0.6022;
end
%}

% ~~~~~~~~~~~~~~~~~~~~~~~
% Can declare as global (may need to change)
n2eff_val = n2eff();
% ~~~~~~~~~~~~~~~~~~~~~~~

% (3) Debye Length
% COMMENTS: *0.3081^2 = D*episilon_ 0*kT/2z^2*e^2 ? *)   (*definition from Roham p37*)
function kappa = kappa(j)  % Translation Verified
    kappa = sqrt(n1eff(j) + 3*n2eff_val) / 0.3081; 
end

% (4) Sigma ???
% COMMENTS: ??? what is this
function sigma = sigma(sig00, sig11, sig22)
    sigma = sig00 - sig11 - 2*sig22; 
end

% (5) Dielectric Discontinuity
function Delta = Delta(i)  % Translation Verified
    kap = kappa(i);
    Delta = (epsilon + kap*d) / (2*epsilon + kap*d)*2;
end

% (6) Kb Test
% COMMENTS: Why?
%{
function Kbtest = Kbtest(i,q,d)
    integrand = @(r) r.^2.*exp(q.*Delta(i).*lb./r);
    rmin = d;
    rmax = q*lb/2;
    Kbtest = 2*pi*integral(integrand, rmin, rmax);
end
%}

% (7) Script M Correction Functions
% COMMENTS:

% Correction terms for 1*1 site
function M1 = M1(p, kappa, a)  % Translation Verified
    tmp = a/2;
    integrand = @(x,y) exp(-kappa.*sqrt(x.^2 + y.^2)) ./ sqrt(x.^2 + y.^2);
    xmin = 0; xmax = tmp;
    ymin = 0; ymax = tmp;
    M1 = 4*integral2(integrand,xmin,xmax,ymin,ymax);
end

% Correction terms for Q*1 site
function Mp = Mp(p, kappa, a)  % Translation Verified
    integrand = @(x,y) exp(-kappa.*sqrt(x.^2 + y.^2)) ./ sqrt(x.^2 + y.^2);
    xmin = 0; xmax = p*a/2;
    ymin = 0; ymax = a/2;
    Mp = 4*integral2(integrand,xmin,xmax,ymin,ymax);
end

% ???
function M2 = M2(p, kappa, a)  % Translation Verified
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
function SumC = SumC(m,kappa,lc)  % Translation Verified
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
    mu1b = log(4/3 * pi * (0.34)^3*(0.6022*n1eff(i))) - 0.5*lb*(kappa(i)/(1+0.36*kappa(i)) + (Delta(i)-1)/d1);
end

function mu2b = mu2b(i, N2)  % condp Verified
    mu2b = log(4/3 * pi * (0.43)^3*(0.6022*n2eff_val)) - 2*lb*(kappa(i)/(1+0.43*kappa(i)) + (Delta(i)-1)/d2);
end

function mupb = mupb(i, Np)  % condp Verified
    mupb = log(vp * (0.6022*np)) - 2*lb*(kappa(i)/(1+0.36*kappa(i)) + (Delta(i)-1)/dp);
end

% (8) Surface chemical potentials
% COMMENTS: 

function mu1c = mu1c(i, a, sigma1, d11)
    kap = kappa(i);
    mu1c = Delta(i) * lb * (-2*pi/kap*(1/a^2-sigma1) + (1/a^2-sigma1) * M1(Q, kap, a) - 1/d1) ...
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
    mupccc = delt * lb * (-2*Q*(N2+0.5)*SumC(10,kap,a) + Q*2*pi / kap * (N1 + 2*N2 + Q*Np)/a^2 ...
        - Q * (-1*m1+(N1+2*N2+1)*(m1 + mp)/2+Q*Np*mp ...
        - 0.5*((mp - m1)-1/delt*(a^2/lattc^2)*(Mp(Q, kap, lattc) - M1(Q, kap, lattc))))/a^2 ...
        - Q/dp)+log(Np*(1-Q*Np)^(Q-1)/(0.5-N1-N2-Q*Np)^Q)-(25/4/pi+1-log(Q))+1/(1-Q*Np)+(25/4/pi)/(1-Q*Np)^2 ...
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
        - Q/dp) + log(Np*(1-Q*Np)^(Q-1)/(1-N1-Q*Np)^Q)-(25/4/pi+1-log(Q))+1/(1-Q*Np)+(25/4/pi)/(1-Q*Np)^2 ...
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

function sigma1r1 = sigma1r1(i, aa)  % Translation Verified
    sigma1r1 = cond(i, aa,d1)*aa^2;
end

function sigma1eff = sigma1eff(i, aa)  % Translation Verified
    sigma1eff = (1/aa^2 - cond(i, aa,d1));               % Same condiion as Fig10c in LPS_Ma.pdf
end

function sigma1reff = sigma1reff(i, aa)  % Translation Verified
    sigma1reff = 1 - cond(i, aa,d1)*aa^2;
end

function cond2 = cond2(i,aa,d1,d2) 
    function F = nle(x)
        F = [mu1b(i,x(1)) - mu1cc(i, aa, x(1), x(2), d1);
            mu2b(i,x(2)) - mu2cc(i, aa, x(1), x(2), d2)];
    end
    x0 = [0.2*sqrt( n1(i) ); 
        0.2*sqrt( n1(i) )];
    ob = fsolve(@nle,x0);
    a = ob(1);
    b = ob(2);
    res = ob;
    cond2 = res;
end

function sigma2r1 = sigma2r1(i, aa)  % Translation Verified
    tmp = cond2(i,aa,d1,d2);
    sigma2r1 = tmp(1);
end

function sigma2r2 = sigma2r2(i, aa)  % Translation Verified
    tmp = cond2(i,aa,d1,d2);
    sigma2r2 = 0.5 + tmp(2);
end

function sigma2r2t = sigma2r2t(i, aa)  % Translation Verified
    tmp = cond2(i,aa,d1,d2);
    sigma2r2t = tmp(2);
end

function negsigma2reff = negsigma2reff(i, aa)  % Translation Verified
    tmp = cond2(i,aa,d1,d2);
    negsigma2reff = tmp(1) + 2*tmp(2);              % Same condition as Fig10c in LPS_Ma.pdf
end

function condp = condp(i,aa)  % Translation Verified
    function F = nle(x)
        F = [mu1b(i, x(1)) - mu1ccc(i, a_Np(aa,dA,x(3)), x(1), x(2), x(3));
            mu2b(i, x(2)) - mu2ccc(i, a_Np(aa,dA,x(3)), x(1), x(2), x(3));
            mupb(i, x(3)) - mupccc(i, a_Np(aa,dA,x(3)), x(1), x(2), x(3))];
        %F = [mu1b(i, x(1)) - mu1ccc(i, aa, x(1), x(2), x(3));
        %    mu2b(i, x(2)) - mu2ccc(i, aa, x(1), x(2), x(3));
        %    mupb(i, x(3)) - mupccc(i, aa, x(1), x(2), x(3))];
    end
    x0 = [n1(i)^0.5; 
         -0.01-n1(i)^0.5;
         0.01-0*0.3*n1(i)^0.5]; % LAM has 0*0.3*n1(i)^0.5].. why the 0* ?
    ob = fsolve(@nle,x0);
    a = ob(1);
    b = ob(2);
    c = ob(3);
    res = ob;
    condp = res;
end

function sigmapr1 = sigmapr1(i, aa)  % Translation Verified
    tmp_p = condp(i, aa);
    sigmapr1 = tmp_p(1);
end

function sigmapr2t = sigmapr2t(i, aa)  % Translation Verified
    tmp_p = condp(i, aa);
    sigmapr2t = tmp_p(2);
end

function sigmapr2 = sigmapr2(i, aa)  % Translation Verified
    tmp_p = condp(i, aa);
    sigmapr2 = 0.5 + tmp_p(2);
end

function sigmaprp = sigmaprp(i, aa)  % Translation Verified
    tmp_p = condp(i, aa);
    sigmaprp = tmp_p(3);
end

function negsigmapreff = negsigmapreff(i, aa)  % Translation Verified
    tmp_p = condp(i, aa);
    negsigmapreff = tmp_p(1) + 2*tmp_p(2) + Q*tmp_p(3);                 % (*Same condiion as Fig10c in LPS_Ma.pdf)*)
end

% this function was left commented out:  
% (its just negsigmapreff(i,aa) output divided by aa^2)
function sigmaeffp = sigmaeffp(i,aa)  % Translation Verified
    tmp_p = conp(i,aa);
    sigmaeffp = (tmp_p(1) + 2*tmp_p(2) + Q*tmp_p(3) ) / aa^2;
end

function condmp = condmp(i,aa)
    function F = nle(x)
        F = [mu1b(i,x(1)) - mu1ccm(i, aa, x(1), x(2));
            mupb(i,x(2)) - mupccm(i, aa, x(1), x(2))];
    end
    x0 = [10*n1(i)^2;
         0.25 - 6*n1(i)^2];
    ob = fsolve(@nle,x0);
    a = ob(1);
    b = ob(2);
    res = ob;
    condmp = res;
end

function sigmampr1 = sigmampr1(i, aa)  % Translation Verified
    tmp_mp = condmp(i, aa);
    sigmampr1 = tmp_mp(1);
end

function sigmamprp = sigmamprp(i, aa)  % Translation Verified
    tmp_mp = condmp(i, aa);
    sigmamprp = tmp_mp(2);
end

function negsigmampreff = negsigmampreff(i, aa)  % Translation Verified
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

% ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
% Some new global vars, as Lam had it
ion = 0;
es = 0;
entr = 0;
entr2 = 0;
corr = 0;
res = 0; % OK this one isn't new
% ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

function FreeEnergy = FreeEnergy(i, aa)
    kap = kappa(i);
    delt = Delta(i);
    ion = cond(i, aa, d1);                 % Here "ion" <=> sigma
    es = delt * pi * lb / kap * (1/aa^2 - ion)^2;
    entr = ion * log(ion*aa^2) + (1/aa^2-ion) * log(1 - ion*aa^2) - ion * mu1b(i, ion*aa^2);
    corr = -delt * lb * ((1/aa^2 - ion)^2 * M1(Q, kap, aa)/2 + ion/d1);
    res = aa^2*(es + entr + corr);
    FreeEnergy = res;
end

function FreeEnergy2 = FreeEnergy2(i, aa)
    kap = kappa(i);
    delt = Delta(i);
    ion = cond2(i, aa, d1, d2);                 % Here "ion" <=> Ni, N2 tilted
    es = delt * pi * lb / kap * (ion(1) + 2*ion(2))^2/aa^2;
    entr = ion(1) * log(ion(1)) + (ion(2)+0.5) * log((ion(2)+0.5)) + (0.5-ion(1)-ion(2)) * log(0.5-ion(1)-ion(2)) - ion(1) * mu1b(i,ion(1)) - (ion(2)+0.5) * mu2b(i,ion(2));
    corr = delt * lb * (-(ion(1)+2*ion(2))^2/aa^2 * M1(Q, kap, aa)/2 + 2*(ion(2)+0.5)*(0.5-ion(1)-ion(2)) * SumC(10, kap, aa) +(- ion(1)*1/d1 - (ion(2)+0.5)*2/d2));
    res = es + entr + corr;
    FreeEnergy2 = res;
end

function FreeEnergyp = FreeEnergyp(i, aa) % NEW: Correction terms
    kap = kappa(i);
    delt = Delta(i);
    m1 = M1(Q, kap, aa);
    mp = Mp(Q, kap, aa);
    m1_lattc = M1(Q, kap, lattc);
    mp_lattc = Mp(Q, kap, lattc);
    ion = condp(i, aa);
    es = delt * pi * lb / kap * (ion(1)+2*ion(2)+Q*ion(3))^2/aa^2;
    entr = (ion(1) * log(ion(1)) + (ion(2)+0.5) * log((ion(2)+0.5)) + ion(3) * log(ion(3)) + (0.5-ion(1)-ion(2)-Q*ion(3)) * log(0.5-ion(1)-ion(2)-Q*ion(3))) + (-ion(1) * mu1b(i,ion(1)) - (ion(2)+0.5) * mu2b(i,ion(2)) - ion(3) * mupb(i,ion(3)));
    entr2 = ((1-Q)/Q) * (1 - Q*ion(3)) * log(1 - Q*ion(3))-ion(3)*(25/4/pi+1-log(Q)) -log(1-Q*ion(3))/Q+(25/4/pi/Q)/(1-Q*ion(3));
    corr = delt * lb * ((-0.5*m1*((ion(1)+2*ion(2))^2) + m1*Q*ion(3) - 0.5*(m1 + mp)*(Q*ion(3))*(ion(1)+2*ion(2)+1)-0.5*mp*(Q*ion(3))^2  + 0.5*Q*ion(3)*((mp - m1)-1/delt*aa^2/lattc^2*(mp_lattc - m1_lattc)))/aa^2 + 2*(ion(2)+0.5)*(0.5-ion(1)-ion(2)-Q*ion(3)) * SumC(10, kap, aa) + (-ion(1)/d1 - 2*(ion(2)+0.5)/d2- Q*ion(3)/dp)); 
    mech = (1/4)*kA*(dA*ion(3)/lattc)^2;
    hydro = ion(3)*H;
    res = es + entr + corr + entr2 + hydro + mech;
    FreeEnergyp = res;
end  

function FreeEnergymp = FreeEnergymp(i, aa)
    kap = kappa(i);
    delt = Delta(i);
    m1 = M1(Q, kap, aa);
    mp = Mp(Q, kap, aa);
    m1_lattc = M1(Q, kap, lattc);
    mp_lattc = Mp(Q, kap, lattc);
    ion = condmp(i, aa);
    es = delt * pi * lb / kap * (ion(1)+Q*ion(2)-1)^2/aa^2;
	entr = ion(1) * log(ion(1)) + ion(2) * log(ion(2)) + (1-ion(1)-Q*ion(2)) * log(1-ion(1)-Q*ion(2)) - ion(1) * mu1b(i,ion(1)) - ion(2) * mupb(i,ion(2));
	entr2 = ((1-Q)/Q) * (1 - Q*ion(2)) * log(1 - Q*ion(2)) -ion(2)*(25/4/pi+1-log(Q))-log(1-Q*ion(2))/Q+(25/4/pi/Q)/(1-Q*ion(2));
	corr = delt * lb * ((-0.5*m1*(ion(1)-1)^2 + m1*Q*ion(2) - 0.5*(m1 + mp)*(Q*ion(2))*(ion(1))-0.5*mp*(Q*ion(2))^2+ 0.5*Q*ion(2)*((mp-m1)-1/delt*aa^2/lattc^2*(mp_lattc - m1_lattc)))/aa^2  + (-ion(1)/d1 - Q*ion(2)/dp)) ;
    mech = (1/4)*kA*(dA*ion(2)/lattc)^2;
    hydro = ion(2)*H;
    res = es + entr + corr + entr2 + hydro + mech;
    FreeEnergymp = res;
end

function FreeEnergy0 = FreeEnergy0(i_, aa)
    es = Delta(i) * lb * 1 / aa^4 * (pi/ kappa(i));
    res = es;
    FreeEnergy0 = res;
end


% (12) The lipid energy expressions
% COMMENTS: what are these functions?

% ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
% Some new global vars, as Lam had it
ah = 0;
ai = 0;
b0 = 0;
b = 0; % this isn't new?
elastic = 0;
res = 0; % also not new
% ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

function LipidEnergy = LipidEnergy(area, aa, i, curv, delta)
    ah = area * (1 + curv * (delta + lh));
    ai = area * (1 + curv * delta );
    b0 = nu / area;
    b = b0 * (1 + curv * (b0/2 - delta) + curv^2 / 2 * (b0^2 - 3 * b0 * delta + 2 * delta^2));
    elastic = lipidB / ah + lipidG * ai + lipidT * b^2;
    res = elastic + FreeEnergy(i, aa)*area;
    LipidEnergy = res;
end

function LipidEnergy2 = LipidEnergy2(area, aa, i, curv, delta)
    ah = area * (1 + curv * (delta + lh));
    ai = area * (1 + curv * delta );
    b0 = nu / area;
    b = b0 * (1 + curv * (b0/2 - delta) + curv^2 / 2 * (b0^2 - 3 * b0 * delta + 2 * delta^2));
    elastic = lipidB / ah + lipidG * ai + lipidT * b^2;
    res = elastic + FreeEnergy2(i, aa)*area
    LipidEnergy2 = res;
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

% NOTE: The functions above should all be vectorized, it would make the
% computation much faster, and allow for cleaner plotting

%{
functions: input (i,aa)
 tmp_p = condp(i, aa);
 sigmapr1 = tmp_p(1);
 sigmapr2 = 0.5 + tmp_p(2);
 sigmaprp = tmp_p(3);
 negsigmapreff = tmp_p(1) + 2*tmp_p(2) + Q*tmp_p(3);
 sigmaeffp = (tmp_p(1) + 2*tmp_p(2) + Q*tmp_p(3) ) / aa^2;

 tmp_0 = cond(i, aa, d1);
 sigma1r1 = tmp_0*aa^2;
 sigma1eff = (1/aa^2-cond(i, aa,d1));               % Same condiion as Fig10c in LPS_Ma.pdf
 sigma1reff = 1 - cond(i, aa,d1)*aa^2;

 tmp_2 = cond2(i,aa,d1,d2);
 sigma2r1 = tmp_2(1);
 sigma2r2 = 0.5 + tmp_2(2);
 sigma2r2t = tmp_2(2);
 negsigma2reff = tmp_2(1) + 2*tmp_2(2);              % Same condition as Fig10c in LPS_Ma.pdf

 tmp_mp = condmp(i, aa);    
 sigmampr1 = tmp_mp(1);
 sigmamprp = tmp_mp(2);
 negsigmampreff = tmp_mp(1) + Q*tmp_p(2);   % MATT: not sure if its condmp..(2) or condp..(2) but he had the latter, i think it should be the first

%instead of calling all the above function individually, just call condp
%once, obtain array, and manipulate values
%}

% ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
% Some reused plotting constants/lists
len = length(n1);
% ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

% Plot Fractional Site Ocuupancy
% Plots the fractional site occupancy based on [Na] for N1, N2, QNp
function plot_frac_site_Na = plot_frac_site_Na()
    xlist = n1;
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
    xmax = n1(len) + n1(len)/len;
    axis([-0.001,xmax,-0.01,1.01]) %xmin xmax ymin ymax
    title(['Fractional Site Occupancy vs [Na^{+}]; [Mg^{2+}] =  ', num2str(n2), ' M, [AMP] = ', num2str(np), ' M'])
    xlabel('[Na^{+}] (M)')
    ylabel('Fractional Site Occupancy')
    legend('N_{1} / N_{0} (Na^{+})','N_{2} / N_{0} (Mg^{2+})','Q*N_{p} / N_{0} (AMP)','Location','northeast')
    saveas(h, 'na_frac_site_MHL.jpg')
end

% Plot Fractional Charge Ocuupancy
% Plots the fractional CHARGE occupancy based on [Na] for N1, N2, QNp
function plot_frac_charge_Na = plot_frac_charge_Na()
    xlist = n1;
    ylist_1 = zeros(1,len);
    ylist_2 = zeros(1,len);
    ylist_p = zeros(1,len);
    for i = 1:len
        condp_tmp = condp(i,lattc);
        ylist_1(i) = condp_tmp(1); % from sigmapr1(i,lattc)
        ylist_2(i) = 2*(0.5 + condp_tmp(2)); % from sigmapr2(i,lattc)
        ylist_p(i) = Q*(condp_tmp(3)); % from sigmaprp(i,lattc)
    end
    h = figure
    plot(xlist,ylist_1,':bs',xlist,ylist_2,':ks',xlist,ylist_p,':rs')
    xmax = n1(len) + n1(len)/len;
    axis([-0.001,xmax,-0.01,1.01]) %xmin xmax ymin ymax
    title(['Fractional Charge Occupancy vs [Na^{+}]; [Mg^{2+}] =  ', num2str(n2), ' M, [AMP] = ', num2str(np), ' M'])
    xlabel('[Na^{+}] (M)')
    ylabel('Fractional Charge Occupancy')
    legend('N_{1} / N_{0} (Na^{+})','2*N_{2} / N_{0} (Mg^{2+})','Q*N_{p} / N_{0} (AMP)','Location','northeast')
    saveas(h, 'na_frac_charge_MHL.jpg')
end

% Plot Free Energy
% Free energy to understand
function plot_freep_Na = plot_freep_Na()    
    aa = lattc;
    xlist = n1;
    len_mod = length(xlist);
    ylist_free_p = zeros(1,len);
    for i = 1:len
        ylist_free_p(i) = FreeEnergyp(i,lattc)
    end
    h = figure
    plot(xlist,ylist_free_p,':bs')
    %axis([0,160,-0.5,0.5])
    %set(gca,'XTickLabel',[0:20:140]) % May need to remove this (could interfere with later plots)
    title(['Free Energy vs [Na^{+}]; [Mg^{2+}] =  ', num2str(1000*n2),' mM, [AMP] = ', num2str(1000*np), ' mM'])
    xlabel('[Na^{+}] (M)')
    ylabel('Free Energy (k_B T')
    saveas(h, 'na_freep_MHL.jpg')
end

% Plot #Tension
% Delta_pi (tension) with monovalent ions, divalent ions, and AMPs  Fig17, LPS_Ma.pdf 
function plot_tensionp_Na = plot_tensionp_Na()    
    aa = lattc;
    xlist = n1;
    ylist_tensionp = zeros(1,len);
    for i = 1:len
        ylist_tensionp(i) = Tensionp(i,lattc);
    end
    h = figure
    plot(xlist,ylist_tensionp,':bs')
    axis([0,0.21,-1,1])
    title(['\Delta \Pi vs [Na^{+}]; [Mg^{2+}] =  ', num2str(n2),' M, [AMP] = ', num2str(np), ' M'])
    xlabel('[Na^{+}] (M)')
    ylabel('\Delta \Pi (k_{B} T / nm^2)')
    saveas(h, 'na_tensionp_MHL.jpg')
end

%{
% Plot #2
% Lateral Pressure vs [Na]
function plot2 = plot2()
    xlist = n1;
    ylist = zeros(1,len);
    for i = 1:len
        ylist(i) = Tension(i,lattc);
    end
    figure
    plot(xlist,ylist,':rs')
    xmax = n1(len) + n1(len)/len;
    %axis([-0.001,xmax,-2,2]) %xmin xmax ymin ymax
    title(['Lateral Pressure vs [Na^{+}]; [Mg^{2+}] =  ', num2str(n2), ' M, [AMP] = ', num2str(np), ' M'])
    xlabel('[Na^{+}] (M)')
    ylabel('Lateral Pressure (mN)')
end

% Plot #3
% "This plots N1,N2,Np, -sigmareff/sigma0 for a given concentration"
% "Comparable to Fig15, LPS_Ma.pdf (Not sure what is [Mg2+] in that plot"
%     {Table[{n1[[i]],sigmaprp[i,lattc]},{i,1,pts}], 
%     Table[{n1[[i]],sigmapr1[i,lattc]},{i,1,pts}] , 
%     Table[{n1[[i]],sigmapr2[i,lattc]},{i,1,pts}] , 
%     Table[{n1[[i]],negsigmapreff[i,lattc]},{i,1,pts}]} , 
%     PlotRange\[Rule]{{0,0.21},{0,2.2}}, 
%     PlotLegends->{"\!\(\*SubscriptBox[\(sigma\), \(p\)]\)/\!\(\*SubscriptBox[\(sigma\), \(0\)]\)", "\!\(\*SubscriptBox[\(sigma\), \(1\)]\)/\!\(\*SubscriptBox[\(sigma\), \(0\)]\)", "\!\(\*SubscriptBox[\(sigma\), \(2\)]\)/\!\(\*SubscriptBox[\(sigma\), \(0\)]\)", "-\!\(\*SubscriptBox[\(sigma\), \(eff\)]\)/\!\(\*SubscriptBox[\(sigma\), \(0\)]\)"},
%     AxesLabel->{"[\!\(\*SuperscriptBox[\(Na\), \(+\)]\)](M)","sigma/\!\(\*SubscriptBox[\(sigma\), \(0\)]\)"}, 
%     PlotLabel-> "\!\(\*SubscriptBox[\(sigma\), \(i\)]\)/\!\(\*SubscriptBox[\(sigma\), \(0\)]\) and -\!\(\*SubscriptBox[\(sigma\), \(eff\)]\)/\!\(\*SubscriptBox[\(sigma\), \(0\)]\) vs [\!\(\*SuperscriptBox[\(Na\), \(+\)]\)], [\!\(\*SuperscriptBox[\(Mg\), \(\(2\)\(+\)\)]\)]=1mM, [AMP]=1\[Micro]M, new MF Correction"]*)
function plot3 = plot3()
    xlist = n1;
    ylist_1 = zeros(1,len);
    ylist_2 = zeros(1,len);
    ylist_p = zeros(1,len);
    ylist_negsigmapreff = zeros(1,len);
    for i = 1:len
        condp_tmp = condp(i,lattc);
        ylist_1(i) = condp_tmp(1); % from sigmapr1(i,lattc)
        ylist_2(i) = 0.5 + condp_tmp(2); % from sigmapr2(i,lattc)
        ylist_p(i) = condp_tmp(3); % from sigmaprp(i,lattc), no Q mult as per lam plot calls above
        ylist_negsigmapreff(i) = condp_tmp(1) + 2*condp_tmp(2) + Q*condp_tmp(3); % from negsigmapreff(i, lattc)
    end
    figure
    plot(xlist,ylist_1,':bs',xlist,ylist_2,':ks',xlist,ylist_p,':rs',xlist,ylist_negsigmapreff,':gs')
    %xmax = n1(len) + n1(len)/len;
    %axis([-0.001,xmax,0,2.2]) %xmin xmax ymin ymax
    title(['N_{x} / N_{0} and -\sigma_{eff} / \sigma_{0} vs [Na^{+}]']); % [Mg^{2+}] =  ', num2str(n2), ' M, [AMP] = ', num2str(np), ' M']);
    xlabel('[Na^{+}] (M)');
    ylabel('N_{x} / \N_{0}');
    legend('N_{1} / N_{0} (Na^{+})','N_{2} / N_{0} (Mg^{2+})','N_{p} / N_{0} (AMP)','N_{eff} / N_{0} (???)','Location','northeast');
end

% Plot #4
% plot for sigmaeffp
%   ListPlot[Table[{n1[[i]],sigmaeffp[i,0.8]}, {i,1,pts}],
%   AxesLabel->{"[\!\(\*SuperscriptBox[\(Na\), \(+\)]\)](M)","sigmaeff(\!\(\*SuperscriptBox[\(nm\), \(-2\)]\))"}, 
%   PlotLabel->"sigmaeff vs [\!\(\*SuperscriptBox[\(Na\), \(+\)]\)], [\!\(\*SuperscriptBox[\(MG\), \(\(2\)\(+\)\)]\)]=1mM,[P]=10\[Micro]M"];
function plot4 = plot4()
    aa = 0.8;
    xlist = n1;
    ylist_sigmaeffp = zeros(1,len);
    for i = 1:len
        condp_tmp = condp(i,aa);
        ylist_sigmaeffp(i) = (condp_tmp(1) + 2*condp_tmp(2) + Q*condp_tmp(3))/aa^2; % from sigmaeffp(i, lattc)
    end
    figure
    plot(xlist,ylist_sigmaeffp,':bs')
    title(['sigmaeffp vs [Na^{+}]']);
    xlabel('[Na^{+}] (M)');
    ylabel('sigmaeffp');
end

% Plot #5
%   {Table[{n1[[i]],sigmampr1[i,lattc]},{i,1,pts}], 
%   Table[{n1[[i]],Q*sigmamprp[i,lattc]},{i,1,pts}]} , 
%   PlotRange->{{0,0.21},{0,1.1}}, 
%   PlotLegends->{"\!\(\*SubscriptBox[\(sigma\), \(1\)]\)/\!\(\*SubscriptBox[\(sigma\), \(0\)]\)", "\!\(\*SubscriptBox[\(sigma\), \(p\)]\)/\!\(\*SubscriptBox[\(sigma\), \(0\)]\)"},
%   AxesLabel->{"[\!\(\*SuperscriptBox[\(Na\), \(+\)]\)](M)","sigma/\!\(\*SubscriptBox[\(sigma\), \(0\)]\)"}, 
%   PlotLabel-> "Site Occupancy vs [\!\(\*SuperscriptBox[\(Na\), \(+\)]\)], [AMP]=1\[Micro]M, Revised"]*)
function plot5 = plot5()
    xlist = n1;
    ylist_m1 = zeros(1,len);
    ylist_mp = zeros(1,len);
    for i = 1:len
        condmp_tmp = condmp(i,lattc);
        ylist_m1(i) = condmp_tmp(1); % from sigmampr1(i,lattc)
        ylist_mp(i) = Q*(condmp_tmp(2)); % from sigmamprp(i,lattc)
    end
    figure
    plot(xlist,ylist_m1,':bs',xlist,ylist_mp,':ks')
    axis([0,0.21,0,1.1]) %xmin xmax ymin ymax
    title(['Site Occupancy vs [Na^{+}]; [Mg^{2+}] =  ', num2str(n2), ' M, [AMP] = ', num2str(np), ' M'])
    xlabel('[Na^{+}] (M)')
    ylabel('N_{x} / N_{0}')
    legend('N_{1} / N_{0} (Na^{+})','N_{p} / N_{0} (AMP)','Location','northeast')
end

% Plot #6
% Fig11, LPS_Ma.pdf         (*This plots N1 with or without Mg ions*)      
%   {Table[{n1[[i]],sigma1r1[i,lattc]},{i,1,pts}], 
%   Table[{n1[[i]],sigma2r1[i,lattc]},{i,1,pts}]} , 
%   PlotRange->{{0,0.2},{0,1}}, 
%   PlotLegends->{"[\!\(\*SuperscriptBox[\(Mg\), \(\(2\)\(+\)\)]\)]=0","[\!\(\*SuperscriptBox[\(Mg\), \(\(2\)\(+\)\)]\)]=1\[Micro]M" },
%   AxesLabel->{"[\!\(\*SuperscriptBox[\(Na\), \(+\)]\)](M)","sigma/\!\(\*SubscriptBox[\(sigma\), \(0\)]\)"}, 
%   PlotLabel-> HoldForm["\!\(\*SubscriptBox[\(sigma\), \(1\)]\)/\!\(\*SubscriptBox[\(sigma\), \(0\)]\) vs [\!\(\*SuperscriptBox[\(Na\), \(+\)]\)], [\!\(\*SuperscriptBox[\(Mg\), \(\(2\)\(+\)\)]\)]=0 and 1\[Micro]M ,a="lattc "nm"]]*)
function plot6 = plot6()
    aa = lattc;
    xlist = n1;
    ylist_1r1 = zeros(1,len);
    ylist_2r1 = zeros(1,len);
    for i = 1:len
        cond_tmp = cond(i, aa, d1);
        cond2_tmp = cond2(i,aa,d1,d2);
        ylist_1r1(i) = cond_tmp*aa^2; % from sigma1r1(i,lattc)
        ylist_2r1(i) = cond2_tmp(1); % from sigma2r1(i,lattc)
    end
    figure
    plot(xlist,ylist_1r1,':bs',xlist,ylist_2r1,':ks')
    axis([0,0.2,0,1]) %xmin xmax ymin ymax
    title(['N_{1} / N_{0} vs [Na^{+}]; [Mg^{2+}] =  ', num2str(n2), ' M, [AMP] = ', num2str(np), ' M'])
    xlabel('[Na^{+}] (M)')
    ylabel('N_{x} / N_{0}')
    legend('N_{1r1} / N_{0} (Na^{+})','N_{2r1} / N_{0} (Na^{+})','Location','northeast')
end

% Plot #7
% Fig11, LPS_Ma.pdf         (*This plots N1,N2, -sigmareff/sigma0 for a given concentration*)     
%   {Table[{n1[[i]],negsigma2reff[i,lattc]},{i,1,pts}], 
%   Table[{n1[[i]],sigma2r1[i,lattc]},{i,1,pts}] , 
%   Table[{n1[[i]],sigma2r2[i,lattc]},{i,1,pts}]} , 
%   PlotRange->{{0,0.21},{-0.5,1}}, 
%   PlotLegends->{"-\!\(\*SubscriptBox[\(sigma\), \(eff\)]\)/\!\(\*SubscriptBox[\(sigma\), \(0\)]\)", "\!\(\*SubscriptBox[\(sigma\), \(1\)]\)/\!\(\*SubscriptBox[\(sigma\), \(0\)]\)", "\!\(\*SubscriptBox[\(sigma\), \(2\)]\)/\!\(\*SubscriptBox[\(sigma\), \(0\)]\)"},
%   AxesLabel->{"[\!\(\*SuperscriptBox[\(Na\), \(+\)]\)](M)","sigma/\!\(\*SubscriptBox[\(sigma\), \(0\)]\)"}, 
%   PlotLabel-> "\!\(\*SubscriptBox[\(sigma\), \(i\)]\)/\!\(\*SubscriptBox[\(sigma\), \(0\)]\) and -\!\(\*SubscriptBox[\(sigma\), \(eff\)]\)/\!\(\*SubscriptBox[\(sigma\), \(0\)]\) vs [\!\(\*SuperscriptBox[\(Na\), \(+\)]\)], [\!\(\*SuperscriptBox[\(Mg\), \(\(2\)\(+\)\)]\)]=1mM, Revised Script"]*)
function plot7 = plot7()
    aa = lattc;
    xlist = n1;
    ylist_negsigma2reff = zeros(1,len);
    ylist_2r1 = zeros(1,len);
    ylist_2r2 = zeros(1,len);
    for i = 1:len
        cond2_tmp = cond2(i,aa,d1,d2);
        ylist_negsigma2reff(i) = cond2_tmp(1) + 2*cond2_tmp(2); % from negsigma2reff(i, lattc)
        ylist_2r1(i) = cond2_tmp(1); % from sigma2r1(i,lattc)
        ylist_2r2(i) = 0.5 + cond2_tmp(2); % from sigma2r2(i,lattc)
    end
    figure
    plot(xlist,ylist_negsigma2reff,':bs',xlist,ylist_2r1,':ks',xlist,ylist_2r2,':rs')
    axis([0,0.21,-0.5,1]) %xmin xmax ymin ymax
    title(['\sigma_{x} / \sigma_{0} and -\sigma_{eff} / \sigma_{0} vs [Na^{+}]'])
    xlabel('[Na^{+}] (M)')
    ylabel('\sigma_{x} / \sigma_{0}')
    legend('- \sigma_{eff} / \sigma_{0}','\sigma_{2r1} / \sigma_{0}','\sigma_{2r2} / \sigma_{0}','Location','northeast')
end

% Plot #8
% plots sigma1eff
%    Table[{n1[[i]],sigma1eff[i,0.8]}, {i,1,pts}], 
%    AxesLabel->{"[Na^+](M)","sigmaeff(nm^-2)"}, 
%    PlotLabel->"sigmaeff for only monvalent counterions, "]
function plot8 = plot8()
    aa = 0.8;
    xlist = n1;
    ylist_1eff = zeros(1,len);
    for i = 1:len
        cond_tmp = cond(i, aa, d1);
        ylist_1eff(i) = (1/aa^2 - cond_tmp);
    end
    figure
    plot(xlist,ylist_1eff,':bs')
    axis([0,0.21,-0.5,1]) %xmin xmax ymin ymax
    title(['sigmaeff for only monvalent counterions'])
    xlabel('[Na^{+}] (M)')
    ylabel('\sigma_{1eff}  (nm ^{-2})')
end

% Plot #9
% plots sigma1r1
%   {Table[{n1[[i]],sigma1r1[i,lattc]}, {i,1,pts}]}, 
%   PlotRange->{{0,0.2},{-0.5,1}}, 
%   PlotLegends->{"\!\(\*SubscriptBox[\(sigma\), \(1\)]\)/\!\(\*SubscriptBox[\(sigma\), \(0\)]\)"}, 
%   AxesLabel->{"[\!\(\*SuperscriptBox[\(Na\), \(+\)]\)](M)","sigma/\!\(\*SubscriptBox[\(sigma\), \(0\)]\)"}, 
%   PlotLabel-> HoldForm["N1 vs [\!\(\*SuperscriptBox[\(Na\), \(+\)]\)], [\!\(\*SuperscriptBox[\(Mg\), \(\(2\)\(+\)\)]\)]=0"]]*)
function plot9 = plot9()
    aa = lattc;
    xlist = n1;
    ylist_1r1 = zeros(1,len);
    for i = 1:len
        cond_tmp = cond(i, aa, d1);
        ylist_1r1(i) = cond_tmp*aa^2; % from sigma1r1(i,lattc)
    end
    figure
    plot(xlist,ylist_1r1,':bs')
    axis([0,0.2,-0.5,1])
    title(['N1 vs [Na^{+}], [Mg^{2+}] = 0'])
    xlabel('[Na^{+}] (M)')
    ylabel('N_{1r1} / N_{0}')
end

% Plot #10
% plots 
%   {Table[{n1[[i]],-sigma1reff[i,lattc]}, {i,1,pts}], 
%   Table[{n1[[i]],sigma1r1[i,lattc]}, {i,1,pts}]}, 
%   PlotRange->{{0,0.2},{-0.5,1}}, 
%   PlotLegends->{"-\!\(\*SubscriptBox[\(sigma\), \(eff\)]\)/\!\(\*SubscriptBox[\(sigma\), \(0\)]\)", "\!\(\*SubscriptBox[\(sigma\), \(1\)]\)/\!\(\*SubscriptBox[\(sigma\), \(0\)]\)"}, 
%   AxesLabel->{"[\!\(\*SuperscriptBox[\(Na\), \(+\)]\)](M)","sigma/\!\(\*SubscriptBox[\(sigma\), \(0\)]\)"}, 
%   PlotLabel-> HoldForm["\!\(\*SubscriptBox[\(sigma\), \(i\)]\)/\!\(\*SubscriptBox[\(sigma\), \(0\)]\) and -\!\(\*SubscriptBox[\(sigma\), \(eff\)]\)/\!\(\*SubscriptBox[\(sigma\), \(0\)]\) vs [\!\(\*SuperscriptBox[\(Na\), \(+\)]\)], [\!\(\*SuperscriptBox[\(Mg\), \(\(2\)\(+\)\)]\)]=0 ,a="a "nm"]];*)
function plot10 = plot10()
    aa = lattc;
    xlist = n1;
    ylist_1reff = zeros(1,len);
    ylist_1r1 = zeros(1,len);
    for i = 1:len
        cond_tmp = cond(i, aa, d1);
        ylist_1reff(i) = 1 - cond_tmp*aa^2;
        ylist_1r1(i) = cond_tmp*aa^2;
    end
    ylist_1reff = (-1)*ylist_1reff % as per lam code
    figure
    plot(xlist,ylist_1reff,':bs',xlist,ylist_1r1,':ks')
    axis([0,0.2,-0.5,1])
    title(['\sigma_{x} / \sigma_{0} and -\sigma_{eff} / \sigma_{0} vs [Na^{+}], [Mg^{2+}] = 0'])
    xlabel('[Na^{+}] (M)')
    ylabel('\N_{x} / \N_{0}')
end

% Plot #11
% Delta_pi (tension) with only monovalent and divalent ions         (*Fig12, LPS_Ma.pdf*) 
%   [Table[{n1[[i]],Tension2[i,lattc]},{i,1,pts}], 
%   PlotRange->{{0,0.21},{-3,3}}, 
%   AxesLabel->{"[\!\(\*SuperscriptBox[\(Na\), \(+\)]\)](M)","\[CapitalDelta]\[pi] (Subscript[k, B]T/nm^2)"}, 
%   PlotLabel->" \[CapitalDelta]\[pi] vs [\!\(\*SuperscriptBox[\(Na\), \(+\)]\)], [\!\(\*SuperscriptBox[\(Mg\), \(\(2\)\(+\)\)]\)]=10mM, Revised"]*)
function plot11 = plot11()
    aa = lattc;
    xlist = n1;
    ylist_tension2 = zeros(1,len);
    for i = 1:len
        ylist_tension2(i) = Tension2(i,lattc);
    end
    figure
    plot(xlist,ylist_tension2,':bs')
    axis([0,0.21,-3,3])
    title(['\Delta \Pi vs [Na^{+}]; [Mg^{2+}] =  ', num2str(n2),' M'])
    xlabel('[Na^{+}] (M)')
    ylabel('\Delta \Pi (k_{B} T / nm^2)')
end

% Plot #12
% Delta_pi (tension) with only monovalent and divalent ions         (*Fig12, LPS_Ma.pdf*) 
%   [{Table[{n1[[i]],Tension2[i,lattc]},{i,1,pts}],
%   Table[{n1[[i]],Tension[i,lattc]},{i,1,pts}]}, 
%   PlotRange\[Rule]{{0,0.21},{-3,3}}, 
%   AxesLabel->{"[\!\(\*SuperscriptBox[\(Na\), \(+\)]\)](M)","\[CapitalDelta]\[pi] (\!\(\*SubscriptBox[\(k\), \(B\)]\)T/\!\(\*SuperscriptBox[\(nm\), \(2\)]\))"}, 
%   PlotLegends->{"[\!\(\*SuperscriptBox[\(Mg\), \(\(2\)\(+\)\)]\)]=1\[Micro]M","[\!\(\*SuperscriptBox[\(Mg\), \(\(2\)\(+\)\)]\)]=0"},  
%   PlotLabel->" \[CapitalDelta]\[pi] vs [\!\(\*SuperscriptBox[\(Na\), \(+\)]\)], [\!\(\*SuperscriptBox[\(Mg\), \(\(2\)\(+\)\)]\)]=1\[Micro]M ,a=0.8nm"]*)
function plot12 = plot12()
    aa = lattc;
    xlist = n1;
    ylist_tension2 = zeros(1,len);
    ylist_tension = zeros(1,len);
    for i = 1:len
        ylist_tension2(i) = Tension2(i,lattc);
        ylist_tension(i) = Tension(i,lattc);
    end
    figure
    plot(xlist,ylist_tension2,':bs',xlist,ylist_tension,':ks')
    axis([0,0.21,-3,3])
    title(['\Delta \Pi vs [Na^{+}]; [Mg^{2+}] =  ', num2str(n2),' M'])
    xlabel('[Na^{+}] (M)')
    ylabel('\Delta \Pi (k_{B} T / nm^2)')
    legend(['[Mg^{2+}] =  ', num2str(n2),' (M)'], ['[Mg^{2+}] = 0'],'Location','northeast')
end

% Plot #13c (PAPER REPLICATOR)
% Delta_pi (tension) with monovalent ions, divalent ions, and AMPs  Fig17, LPS_Ma.pdf 
function plot13c = plot13c()    
    aa = lattc;
    xlist = 1000*n1(1:17);
    len_mod = length(xlist);
    ylist_tensionp = zeros(1,len_mod);
    for i = 1:len_mod
        ylist_tensionp(i) = Tensionp(i,lattc)
    end
    figure
    plot(xlist,ylist_tensionp,':bs')
    axis([0,160,-1.0,1.0])
    %set(gca,'XTickLabel',[0:20:140]) % May need to remove this (could interfere with later plots)
    title(['\Delta \Pi vs [Na^{+}]; [Mg^{2+}] =  ', num2str(1000*n2),' mM, [AMP] = ', num2str(1000*np), ' mM'])
    xlabel('[Na^{+}] (mM)')
    ylabel('\Delta \Pi (k_{B} T / nm^2)')
end

% Plot #14
% Delta_pi (tension) with monovalent ions, divalent ions, and with or without AMPs*)  (*Fig17, LPS_Ma.pdf*) 
%   [{Table[{n1[[i]],Tensionp[i,lattc]},{i,1,pts}],
%   Table[{n1[[i]],Tension2[i,lattc]},{i,1,pts}]}, 
%   PlotRange->{{0,0.2},{-2,2}}, 
%   AxesLabel->{"[\!\(\*SuperscriptBox[\(Na\), \(+\)]\)](M)","\[CapitalDelta]\[pi] (\!\(\*SubscriptBox[\(k\), \(B\)]\)T/\!\(\*SuperscriptBox[\(nm\), \(2\)]\))"}, 
%   PlotLegends->{"[\!\(\*SuperscriptBox[\(Mg\), \(\(2\)\(+\)\)]\)]=1mM","[\!\(\*SuperscriptBox[\(Mg\), \(\(2\)\(+\)\)]\)]=0"},  
%   PlotLabel->" \[CapitalDelta]\[pi] vs [\!\(\*SuperscriptBox[\(Na\), \(+\)]\)], [\!\(\*SuperscriptBox[\(Mg\), \(\(2\)\(+\)\)]\)]=1mM [AMP]=0 or 1\[Micro]M, new MF Correction"]*)
function plot14 = plot14()
    aa = lattc;
    xlist = n1;
    ylist_tensionp = zeros(1,len);
    ylist_tension2 = zeros(1,len);
    for i = 1:len
        ylist_tensionp(i) = Tensionp(i,lattc);
        ylist_tension2(i) = Tension2(i,lattc);
    end
    figure
    plot(xlist,ylist_tensionp,':bs',xlist,ylist_tension2,':ks')
    axis([0,0.21,-3,3])
    title(['\Delta \Pi vs [Na^{+}]; [Mg^{2+}] =  ', num2str(n2),' M, [AMP] = 0 or ', num2str(np), ' M'])
    xlabel('[Na^{+}] (M)')
    ylabel('\Delta \Pi (k_{B} T / nm^2)')
    legend(['[Mg^{2+}] =  ', num2str(n2),' (M)'], ['[Mg^{2+}] = 0'],'Location','northeast')
end

% Plot #15
% Delta_pi (tension) with monovalent ions and AMPs only*)
%   [Table[{n1[[i]],Tensionm[i,lattc]},{i,1,pts}], 
%   PlotRange->{{0,0.21},{-2,2}}, 
%   AxesLabel->{"[\!\(\*SuperscriptBox[\(Na\), \(+\)]\)](M)","\[CapitalDelta]\[pi] (\!\(\*SubscriptBox[\(k\), \(B\)]\)T/\!\(\*SuperscriptBox[\(nm\), \(2\)]\))"}, 
%   PlotLabel->" \[CapitalDelta]\[pi] vs [\!\(\*SuperscriptBox[\(Na\), \(+\)]\)], [AMP]=10nM, Revised"]*)
function plot15 = plot15()    
    aa = lattc;
    xlist = n1;
    ylist_tensionm = zeros(1,len);
    for i = 1:len
        ylist_tensionm(i) = Tensionm(i,lattc);
    end
    figure
    plot(xlist,ylist_tensionm,':bs')
    axis([0,0.21,-2,2])
    title(['\Delta \Pi vs [Na^{+}]; [AMP] = ', num2str(np), ' M'])
    xlabel('[Na^{+}] (M)')
    ylabel('\Delta \Pi (k_{B} T / nm^2)')
end

%}

% ==========================
% Save Data
% ==========================

function get_custom_data = get_custom_data()
    xlist = n1;
    ylist_1 = zeros(1,len);
    ylist_2 = zeros(1,len);
    ylist_p = zeros(1,len);
    ylist_free_p = zeros(1,len);
    ylist_tensionp = zeros(1,len);
    for i = 1:len
        condp_tmp = condp(i,lattc);
        ylist_1(i) = condp_tmp(1); % from sigmapr1(i,lattc)
        ylist_2(i) = 0.5 + condp_tmp(2); % from sigmapr2(i,lattc)
        ylist_p(i) = Q*(condp_tmp(3)); % from sigmaprp(i,lattc)
        ylist_free_p(i) = FreeEnergyp(i,lattc);
        ylist_tensionp(i) = Tensionp(i,lattc);
    end
    ylist_22 = 2*ylist_2; % reps frac charge occ for Mg
    M = real([n1', ylist_1', ylist_2', ylist_22', ylist_p', ylist_free_p', ylist_tensionp'])
    filename = sprintf('data\\na_custom_mg_%1.1f_mM_amp_%1.1f_muM_MHL.txt', 1000*n2, 1e6*np);
    fid = fopen(filename, 'w');
    fprintf(fid, '[Na],N_1 / N_0,N_2 / N_0,2*N_2 / N_0,Q*N_p / N_0,Free Energy,Tension\n');
    fclose(fid)
    dlmwrite(filename,M,'-append','precision',4)
    
    get_custom_data = M;
end

function get_data_no_AMP = get_data_no_AMP()
    xlist = n1;
    ylist_tension2 = zeros(1,len);
    for i = 1:len
        ylist_tension2(i) = Tension2(i,lattc);
    end
    M = real([n1', ylist_tension2'])
    filename = sprintf('data\\na_custom_mg_%1.1f_mM_amp_ZERO_muM_MHL.txt', 1000*n2);
    fid = fopen(filename, 'w');
    fprintf(fid, '[Na],Tension\n');
    fclose(fid)
    dlmwrite(filename,M,'-append','precision',4)
    get_data_no_AMP = M;
end

function get_data_no_AMP_no_Mg = get_data_no_AMP_no_Mg()
    xlist = n1;
    ylist_tension = zeros(1,len);
    for i = 1:len
        ylist_tension(i) = Tension(i,lattc);
    end
    M = real([n1', ylist_tension'])
    filename = sprintf('data\\na_custom_mg_ZERO_mM_amp_ZERO_muM_MHL.txt');
    fid = fopen(filename, 'w');
    fprintf(fid, '[Na],Tension\n');
    fclose(fid)
    dlmwrite(filename,M,'-append','precision',4)
    get_data_no_AMP_no_Mg = M;
end

function get_site_free_tension_Na = get_site_free_tension_Na()
    xlist = n1;
    ylist_1 = zeros(1,len);
    ylist_2 = zeros(1,len);
    ylist_p = zeros(1,len);
    ylist_free_p = zeros(1,len);
    ylist_tensionp = zeros(1,len);
    for i = 1:len
        condp_tmp = condp(i,lattc);
        ylist_1(i) = condp_tmp(1); % from sigmapr1(i,lattc)
        ylist_2(i) = 0.5 + condp_tmp(2); % from sigmapr2(i,lattc)
        ylist_p(i) = Q*(condp_tmp(3)); % from sigmaprp(i,lattc)
        ylist_free_p(i) = FreeEnergyp(i,lattc);
        ylist_tensionp(i) = Tensionp(i,lattc);
    end
    ylist_22 = 2*ylist_2; % reps frac charge occ for Mg
    M = real([n1', ylist_1', ylist_2', ylist_22', ylist_p', ylist_free_p', ylist_tensionp'])
    filename = 'data\\na_data.txt';
    fid = fopen(filename, 'w');
    fprintf(fid, '[Na],N_1 / N_0,N_2 / N_0,2*N_2 / N_0,Q*N_p / N_0,Free Energy,Tension\n');
    fclose(fid)
    dlmwrite(filename,M,'-append','precision',4)
    
    get_site_free_tension_Na = M;
end

function get_tension_data = get_tension_data()
    conversion_factor = 4.114
    ylist_tensionmech = zeros(1,len);
    ylist_tensionelec = zeros(1,len);
    for i = 1:len
        ylist_tensionmech(i) = TensionMech(i,lattc)*conversion_factor;
        ylist_tensionelec(i) = Tensionp(i,lattc)*conversion_factor
    end
    M = real([n1', ylist_tensionmech', ylist_tensionelec'])
    get_tension_data = M;
end

% ==========================
% Main
% ==========================

function main = main()
    %plot_frac_site_Na()
    %plot_frac_charge_Na()
    %plot_freep_Na()
    %plot_tensionp_Na()
    %get_data_no_AMP()
    %get_data_no_AMP_no_Mg()
    %data_plot = get_custom_data()
    Q = get_tension_data()
    OUTPUTS = 1;
end

main()
%OUTPUTS = get_data_no_AMP();
%OUTPUTS = get_data_no_AMP_no_Mg();

end


%{
Changes from the original code:
    1. mech and hydro terms added to [mupccc]
    2. mech and hydro terms added to [FreeEnergyp]
    3. [a(Np)] fn added, used in condp (instead of lattc)
    4. global consts dA H kA, conversions (cA, cJ)

TODO
 1. fix condp so it has a(Np) again
%}