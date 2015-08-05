n = 100;

c_range = [1:10];
e1_range = [1];
e2_range = [1:n,1];
a1_range = [1:n,1];
a2_range = [1:n,1];

10*1*n*n*n
stable_column = zeros(n*n*n, 1);

counter=1;
for c = c_range
    for e1 = e1_range
        for e2 = e2_range
            for a1 = a1_range
                for a2 = a2_range
                    is_stable = ode_competition_stability(c,e1,e2,a1,a2);
                    stable_column(counter) = is_stable;
                    counter = counter + 1;
                end
            end
        end
    end
end
