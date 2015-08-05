n = 2;
step = 0.1;

c_range = [1];
e1_range = [1];
e2_range = [0+step:n, step];
a1_range = [0+step:n, step];
a2_range = [0+step:n, step];

counter=1;
for c = c_range
    for e1 = e1_range
        for e2 = e2_range
            for a1 = a1_range
                for a2 = a2_range
                    is_stable = ode_competitionbasic_stability(c,e1,e2,a1,a2)
                    %stable_column(counter) = is_stable;
                    counter = counter + 1;
                end
            end
        end
    end
end