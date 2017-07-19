% CURRENT BIFUCATION PARAMETER: b (or delta)

alpha_plus = 0.4;
alpha_minus = 0.5;
mu = 0.01;
a = 1;
%b = 1.376666; %1.3;
c = 1.2;
N = 100;

%threshold1 = 2*s + delta + alpha_plus + alpha_minus + mu;
%threshold2 = (s + alpha_plus)*(s + delta + alpha_minus + mu) - alpha_minus*alpha_plus;

density = 100;
bifurcation_search = linspace(1.3, 1.5, density);
nn = length(bifurcation_search);
fixed_points = zeros(3, nn*3); %each set of fp is a row of three 3vecs
stabilities = zeros(1,nn*3);
scatter_colours = zeros(nn*3,3);
x1_colours = [0,0,0; 1,0,0];  % black stable, red unstable
x2_colours = [0,0,1; 0.196078, 0.803922, 0.196078];  % blue stable, green unstable
x3_colours = [1,0,1; 1, 0.54902,0];  % pink stable, orange unstable
for idx = 1:nn
    b = bifurcation_search(idx);
    params = [alpha_plus; alpha_minus; mu; a; b; c; N];
    [fp,stabs] = phenotype_model_B_bifurcdiagram_get(params);
    fixed_points(:,3*(idx-1) + 1:3*idx) = fp;
    stabilities(3*(idx-1) + 1:3*idx) = stabs;
    x1_colours(stabs(1),:)
    scatter_colours(3*(idx-1)+1, :) = x1_colours(2-stabs(1),:);
    scatter_colours(3*(idx-1)+2, :) = x2_colours(2-stabs(2),:);
    scatter_colours(3*(idx-1)+3, :) = x3_colours(2-stabs(3),:);

end

figure
%plot simplex
simplex_colour = [0 0.75 0.75];
plot_simplex = patch([N;0;0], [0;N;0], [0;0;N], simplex_colour, 'FaceAlpha',.25);
hold on
plot_simplex_big = patch([1000, 1000, N-2000], [1000, N-2000, 1000], [N-2000, 1000, 1000], [0,0,0], 'FaceAlpha',.15);
hold on
grid on
%plot FPs using scatter3
X = fixed_points(1,:);
Y = fixed_points(2,:);
Z = fixed_points(3,:);
scatter_size = 4;
axis_lim = N*10;
xlabel('X');
ylabel('Y');
zlabel('Z');
for i = 2:nn*3
    s = scatter3(X(1:i),Y(1:i),Z(1:i), scatter_size, scatter_colours(1:i,:));
    axis([-axis_lim axis_lim -axis_lim axis_lim -axis_lim axis_lim]);
    title(num2str(bifurcation_search(round(i/3))))
    pause(0.25)
end
%details

