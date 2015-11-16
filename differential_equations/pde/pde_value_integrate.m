function state_totals = pde_value_integrate(u, model)
% integrate total value of a given state at a given time

% sums the contribution of each area slice to the total value of a
% specified state (equation index) of u at a specified timepoint (index)

% extract pde solution parameters
timesteps = size(u, 2);
np = size(model.Mesh.Nodes, 2);
N = size(u,1)/np;

% reformat mesh data for easier looping
[p, ~, t] = meshToPet(model.Mesh);

state_totals = zeros(N,timesteps);
for tt = 1:timesteps
    for tri_n = 1: size(t,2)  
        % vertex numbers
        v1n = t(1,tri_n);
        v2n = t(2,tri_n);
        v3n = t(3,tri_n);
        % vertex coordinates (x,y)
        v1 = p(:,v1n);
        v2 = p(:,v2n);
        v3 = p(:,v3n);
        % get triangle area
        A = 0.5*abs(v1(1)*(v2(2)-v3(2)) + v2(1)*(v3(2)-v1(2)) + v3(1)*(v1(2)-v2(2)));
        % update state integral values with new area element
        for state = 1:N
            state_idx = (state-1)*np;
            state_tri_average = (u(state_idx+v1n,tt)+u(state_idx+v2n,tt)+u(state_idx+v3n,tt))/3;
            state_totals(state,tt) = state_totals(state,tt) + state_tri_average*A;
        end
    end
end

end
