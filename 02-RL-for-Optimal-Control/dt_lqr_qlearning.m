%% Model-free Q-learning policy iteration for discrete-time LQR
% HW6, Q2 (data-driven version). The Q-function of policy u = -Kx is
% Q_K(x,u) = [x;u]' H [x;u]. H is identified from measured data only:
%   z_k' H z_k - z_{k+1}' H z_{k+1} = x_k'Q x_k + u_k'R u_k,  z = [x; u],
% with u_{k+1} = -K x_{k+1} (target policy) and u_k = -K x_k + e_k
% (behaviour policy with probing noise). Then K <- H_uu^{-1} H_ux.
clear; close all; rng(1);

A = [0.9065 0.0816 -0.0005; 0.0743 0.9012 -0.0007; 0 0 0.1327];
B = [-0.0027; -0.0068; 1];      % used only to simulate the plant
Q = eye(3); R = 1; n = 3; m = 1; nz = n + m;
[~, ~, Kstar] = dare(A, B, Q, R);

K = zeros(m, n);  % A is stable, so K0 = 0 is admissible
nSamples = 60; maxIt = 30; Khist = K; Hhist = {};
for j = 1:maxIt
    Phi = []; y = [];
    x = 10*randn(n,1);
    for k = 1:nSamples
        u  = -K*x + 0.5*randn(m,1);          % behaviour (exploring) input
        xn = A*x + B*u;                       % measured next state
        un = -K*xn;                           % target policy at next state
        Phi(end+1,:) = quad_basis([x; u])' - quad_basis([xn; un])'; %#ok<SAGROW>
        y(end+1,1)   = x'*Q*x + u'*R*u;                             %#ok<SAGROW>
        x = xn; if norm(x) < 1e-3, x = 10*randn(n,1); end
    end
    H = basis_to_sym(Phi \ y, nz);
    Kn = H(n+1:end, n+1:end) \ H(n+1:end, 1:n);
    Hhist{end+1} = H; Khist(end+1,:) = Kn; %#ok<SAGROW>
    if norm(Kn - K) < 1e-9, K = Kn; break; end
    K = Kn;
end
fprintf('Q-learning: %d iterations, K = [%s]\n', j, num2str(K, '%9.5f'));
fprintf('DARE:                    K* = [%s]\n', num2str(Kstar, '%9.5f'));
fprintf('|K - K*| = %.2e\n', norm(K - Kstar));

figure('Position', [100 100 600 300]);
semilogy(0:size(Khist,1)-1, vecnorm(Khist - Kstar, 2, 2), '-o', 'LineWidth', 1.3); grid on;
xlabel('Iteration j'); ylabel('$\|K_j - K^*\|$', 'Interpreter', 'latex');
title('Model-free Q-learning: distance to the DARE gain');


% Save the figure
if ~exist('figures', 'dir'), mkdir('figures'); end
exportgraphics(gcf, fullfile('figures', 'dt_lqr_qlearning.png'), 'Resolution', 150);

function b = quad_basis(z)
% upper-triangular quadratic monomials, off-diagonal ones doubled
nz = numel(z); b = [];
for i = 1:nz
    for jj = i:nz
        if i == jj, b(end+1,1) = z(i)^2; else, b(end+1,1) = 2*z(i)*z(jj); end %#ok<AGROW>
    end
end
end
function H = basis_to_sym(h, nz)
H = zeros(nz); c = 0;
for i = 1:nz
    for jj = i:nz
        c = c + 1; H(i,jj) = h(c); H(jj,i) = h(c);
    end
end
end
