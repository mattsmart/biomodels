function [x_list, stab_list] = phenotype_model_B_bifurcdiagram_get(params)

%{
params is array of parameters in following format
params(1) = alpha_plus
params(2) = alpha_minus
params(3) = mu
params(4) = a
params(5) = b
params(6) = c
params(7) = N

x_list and stab_list are cols of the FPs and their stabiities (true=stable)
%}

alpha_plus = params(1);
alpha_minus = params(2);
mu = params(3);
a = params(4);
b = params(5);
c = params(6);
N = params(7);
delta = 1-b;
s = c-1;
at=a-alpha_plus;
bt=b-alpha_minus-mu;
A = [-alpha_plus, alpha_minus,         0;
      alpha_plus, -(alpha_minus + mu), 0;
      0,          mu,                  0];
B = diag([a,b,c]);
I = diag([1,1,1]);
one = [1,1,1];
M = A + B;
avec = [a;b;c];


[V,D] = eig(M);
v1 = V(:,1);
v2 = V(:,2);
v3 = V(:,3);

disp('normalizing')
n1=one*v1;
n2=one*v2;
n3=one*v3;
x1 = N/n1*v1;
x2 = N/n2*v2;
x3 = N/n3*v3;

disp('linearizing')
J1 = M - 1/N*(avec'*x1*I + x1*avec');
[J1v, J1d] = eig(J1);
J1d;

J2 = M - 1/N*(avec'*x2*I + x2*avec');
[J2v, J2d] = eig(J2);
J2d;

J3 = M - 1/N*(avec'*x3*I + x3*avec');
[J3v, J3d] = eig(J3);
J3d;

stab1 = all(all(J1d(:,:)<=0));
stab2 = all(all(J2d(:,:)<=0));
stab3 = all(all(J3d(:,:)<=0));


threshold1 = 2*s + delta + alpha_plus + alpha_minus + mu
threshold2 = (s + alpha_plus)*(s + delta + alpha_minus + mu) - alpha_minus*alpha_plus

x_list = [x1,x2,x3];
stab_list = [stab1, stab2, stab3];
end

