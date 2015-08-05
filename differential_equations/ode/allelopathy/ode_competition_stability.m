function is_stable = ode_competition_stability(c, e1, e2, a1, a2)
%   check stability at the non-trivial EQ point based on eigenvalues

    function J = get_jacobian_at_crit(c, e1, e2, a1, a2)
        crit_gamma = c/(c*a1*a2 + a1*e2 + a2*e1);
        crit_x = crit_gamma * a1 * e2;
        crit_y = crit_gamma * a2 * e1;
        j11 = e1 - 2*(e1/c)*crit_x - (e1/c + a1)*crit_y;
        j12 = -(e1/c + a1)*crit_x;
        j21 = -(e2/c + a2)*crit_y;
        j22 = e2 - 2*(e2/c)*crit_y - (e2/c + a2)*crit_x;
        J = [j11, j12; j21, j22];
    end

    function is_stable = stability_at_crit(c, e1, e2, a1, a2)
        J_crit = get_jacobian_at_crit(c, e1, e2, a1, a2);
        [V,D] = eig(J_crit);
        evals = diag(D);
        if (evals(1) > 0) && (evals(2) > 0)
            is_stable = true;
        else
            is_stable = false;
        end
    end
   
    is_stable = stability_at_crit(c, e1, e2, a1, a2);

end
