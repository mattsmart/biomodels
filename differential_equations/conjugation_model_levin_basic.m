% levin ODEs for chemostat, distinguish transconjugants from donors
% reference: levin, stewart, and rice (1978)
% input:
% - tspan: initial time, final time as an array
% - x0: array of initial state values
% output:
% - matrix of values over time, representing state evolution

function [t,x] = conjugation_model_levin_basic(tspan, x0)

assert(length(tspan) == 2)
assert(length(x0) == 3)

% constants:
R = 1.0;  % growth rate
K = 0.5;  % conjugation/transmission rate

% states: donors (x_1), transconjugants (x_2), recipients (x_3)
% we write x = [x_1, x_2, x_3] so that x(1) == x_1, etc
% initial state values: x0 prespecified
% tspan: tspan prespecified

% equations of motion:
function dxdt = dstate(t,x)
    dxdt = zeros(size(x0));
    dxdt(1) = R * x(1);
    dxdt(2) = R * x(2) + K * x(3) * (x(1) + x(2));
    dxdt(3) = R * x(3) - K * x(3) * (x(1) + x(2));
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

end
