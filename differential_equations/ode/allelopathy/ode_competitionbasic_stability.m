function [eval1, eval2, is_stable] = ode_competitionbasic_stability(c, e1, e2, a1, a2)
%   check stability at the non-trivial EQ point based on eigenvalues

    function J = get_jacobian_at_crit(c, e1, e2, a1, a2)
        crit_gamma = c/((c^2)*a1*a2 - e1*e2);
        crit_x = crit_gamma * (a1*c - e1) * e2;
        crit_y = crit_gamma * (a2*c - e2) * e1;
        j11 = e1 - 2*(e1/c)*crit_x - a1*crit_y;
        j12 = -a1*crit_x;
        j21 = -a2*crit_y;
        j22 = e2 - 2*(e2/c)*crit_y - a2*crit_x;
        J = [j11, j12; j21, j22];
    end

    function evals = stability_at_crit(c, e1, e2, a1, a2)
        J_crit = get_jacobian_at_crit(c, e1, e2, a1, a2);
        [V,D] = eig(J_crit);
        evals = diag(D);
    end
   
    function is_stable = assess_evals(evals)
        if (real(evals(1)) < 0) && (real(evals(2)) < 0)
            is_stable = true;
        else
            is_stable = false;
        end
    end

    evals = stability_at_crit(c, e1, e2, a1, a2);
    is_stable = assess_evals(evals);
    eval1 = evals(1);
    eval2 = evals(2);

end
