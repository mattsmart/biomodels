n = 2;
step = 0.1;

c_range = [1];
e1_range = [1];
e2_range = [0+step:step:n];
a1_range = [0+step+step/10:step:n];
a2_range = [0+step:step:n];

n = length(c_range)*length(e1_range)*length(e2_range)*length(a1_range)*length(a2_range)
stats_array = zeros(n, 6);

counter=1;
for c = c_range
    for e1 = e1_range
        for e2 = e2_range
            for a1 = a1_range
                for a2 = a2_range
                    [eval1, eval2, is_stable] = ode_competitionbasic_stability(c,e1,e2,a1,a2);
                    stats_array(counter,:) = [eval1, eval2, is_stable, e2, a1, a2];
                    counter = counter + 1;
                end
            end
        end
    end
end
stats_array
stability_column = stats_array(:,3);