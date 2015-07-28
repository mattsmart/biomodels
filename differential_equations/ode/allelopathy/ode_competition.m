% competition ODEs for n species
% reference: Boyce and DiPrima Ch.9 (text), 
% usage: [t,x] = conjugation_model_levin_basic([0,12],[1e5;1e4])
% input:
% - tspan: initial time, final time as an array
% - x0: array of initial state values
% output:
% - matrix of values over time, representing state evolution

function [t,x] = ode_competition(tspan, x0)

assert(length(tspan) == 2)
assert(length(x0) == 2)

% constants:
% growth rates by species
epsilon = [1.0; 2.0];
% growth rate / saturation level by species
saturation_limit = 1e4;  % match abm limit
theta = epsilon / saturation_limit;
% competitiveness by species
alpha = [1.0; 2.0];

% states: species 1 (x_1), species 2 (x_2)
% we write x = [x_1, x_2] so that x(1) == x_1, etc
% initial state values: x0 prespecified
% tspan: tspan prespecified

% equations of motion:
function dxdt = dstate(t,x)
    dxdt = zeros(size(x0));
    dxdt(1) = x(1) * (epsilon(1) - theta(1) * x(1) - alpha(1) * x(2));
    dxdt(2) = x(2) * (epsilon(2) - theta(2) * x(2) - alpha(2) * x(1));
end

% solve the DEs subject to initial conditions and timespan
[t,x] = ode45(@dstate, tspan, x0);

% plots
subplot(1,2,1);
plot(t,x(:,1));
title('Species 1');
subplot(1,2,2);
plot(t,x(:,2));
title('Species 2');

end
