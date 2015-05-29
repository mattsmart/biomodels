% extended levin ODEs for conjugation
% reference: simonsen, gordon, stewart, and levin (1990)
% usage: [t,x] = conjugation_model_levin_extended([0,12],[1e5;0;1e5;750])
% input:
% - tspan: initial time, final time as an array
% - x0: array of initial state values
% output:
% - matrix of values over time, representing state evolution

function [t,x] = conjugation_model_levin_extended(tspan, x0)

assert(length(tspan) == 2)
assert(length(x0) == 4)

% constants:
R_max = 1.0;   % [per hour] max growth rate
K_max = 1e-9;  % [per hour] max conjugation/transmission rate
Q = 4;         % [ug/ml] half-saturation constant for the monod functions
delta = 5e-7;  % [ug/cell] resource cost per new cell

% states: D (x_1), T (x_2), R (x_3), resources/nutrients (x_4)
% initial state values: x0 prespecified
%   e.g. 1e3 to 1e7 donors and recipients, try 5e6
%        initial resources 750 ug/ml
% tspan: tspan prespecified

% equations of motion:
f = @(C) R_max * C / (Q + C)  % monod growth function
g = @(C) K_max * C / (Q + C)  % monod transfer function
function dxdt = dstate(t,x)
    dxdt = zeros(size(x0));
    dxdt(1) = f(x(4)) * x(1);
    dxdt(2) = f(x(4)) * x(2) + g(x(4)) * x(3) * (x(1) + x(2));
    dxdt(3) = f(x(4)) * x(3) - g(x(4)) * x(3) * (x(1) + x(2));
    dxdt(4) = -f(x(4)) * (x(1) + x(2) + x(3)) * delta;
end

% solve the DEs subject to initial conditions and timespan
[t,x] = ode45(@dstate, tspan, x0);

% plots
subplot(2,2,1);
plot(t,x(:,1));
title('Donors');
subplot(2,2,2);
plot(t,x(:,2));
title('Transconjugants');
subplot(2,2,3);
plot(t,x(:,3))
title('Recipients');
subplot(2,2,4);
plot(t,x(:,4))
title('Resources');

end
