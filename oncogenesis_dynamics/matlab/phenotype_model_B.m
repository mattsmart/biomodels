alpha_plus = 0.4;
alpha_minus = 0.5;
mu = 0.01;
a = 1;
b = 1.376666; %1.3;
c = 1.2;
N = 100;
at=a-alpha_plus
bt=b-alpha_minus-mu
A = [-alpha_plus, alpha_minus,         0;
      alpha_plus, -(alpha_minus + mu), 0;
      0,          mu,                  0];
B = diag([a,b,c]);
I = diag([1,1,1]);
one = [1,1,1];
M = A + B
avec = [a;b;c];


[V,D] = eig(M)
v1 = V(:,1);
v2 = V(:,2);
v3 = V(:,3);

disp('normalizing')
n1=one*v1;
n2=one*v2;
n3=one*v3;
x1 = N/n1*v1
x2 = N/n2*v2
x3 = N/n3*v3

disp('linearizing')
J1 = M - 1/N*(avec'*x1*I + x1*avec');
[J1v, J1d] = eig(J1);
J1d

J2 = M - 1/N*(avec'*x2*I + x2*avec');
[J2v, J2d] = eig(J2);
J2d

J3 = M - 1/N*(avec'*x3*I + x3*avec');
[J3v, J3d] = eig(J3);
J3d
disp('J2 has all negative eigenvalues therefore stable???????')

bma = b-alpha_minus-mu-a+alpha_plus;
q_plus = (0.5/alpha_minus)*(bma + sqrt(bma^2 + 4*alpha_plus*alpha_minus));
q_minus = (0.5/alpha_minus)*(bma - sqrt(bma^2 + 4*alpha_plus*alpha_minus));
q1 = max(q_plus, q_minus)
q1_check = x2(2)/x2(1)

xsol_1 = N*((a-alpha_plus - c) +alpha_minus*q1) / (a-c + q1*(b-c))
ysol_1 = q1*xsol_1

stab1 = all(all(J1d(:,:)<=0));
stab2 = all(all(J2d(:,:)<=0));
stab3 = all(all(J3d(:,:)<=0));

x_list = [x1,x2,x3]
stab_list = [stab1, stab2, stab3]
