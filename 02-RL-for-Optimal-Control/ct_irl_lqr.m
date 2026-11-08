%% Integral reinforcement learning (IRL) for continuous-time LQR
% HW7, Q1. Vrabie et al. (2009) policy iteration: for the current policy
% u = -K_j x, the value V_j = x' P_j x satisfies, over any interval [t, t+T],
%   x(t)' P_j x(t) - x(t+T)' P_j x(t+T) = int_t^{t+T} x'(Q + K_j' R K_j) x dtau.
% P_j comes from measured state data by least squares (A is never used),
% then K_{j+1} = R^{-1} B' P_j (needs B only).
clear; close all; rng(1);

A = [-1.01887 0.90506 -0.00215; 0.82225 -1.07741 -0.17555; 0 0 -1];
B = [0; 0; 1]; Q = eye(3); R = 1; n = 3;
[Pstar, ~, Kstar] = care(A, B, Q, R);

T = 0.05; dt = 1e-3; nInt = round(T/dt); nEq = 30;   % 30 intervals per iteration
K = zeros(1, n);                 % A is Hurwitz, so u = 0 is admissible
x = [10; -10; -3];               % the given initial state
Khist = K; Phist = {};
for j = 1:15
    Phi = zeros(nEq, 6); y = zeros(nEq, 1);
    for k = 1:nEq
        b0 = quad_basis(x); c = 0;
        for s = 1:nInt
            u  = -K*x;
            xn = rk4(@(z) A*z + B*(-K*z), x, dt);
            c  = c + dt/2*(x'*(Q + K'*R*K)*x + xn'*(Q + K'*R*K)*xn);
            x  = xn;
        end
        Phi(k,:) = (b0 - quad_basis(x))'; y(k) = c;
        if mod(k, 5) == 0, x = 10*randn(n,1); end   % new trial every 5 intervals keeps the data rich
    end
    P = basis_to_sym(Phi \ y);
    Kn = R \ (B'*P);
    Phist{j} = P; Khist(end+1,:) = Kn; %#ok<SAGROW>
    fprintf('j = %2d  K = [%8.5f %8.5f %8.5f]  |P - P*| = %.2e\n', j, Kn, norm(P - Pstar));
    if norm(Kn - K) < 1e-5, K = Kn; break; end
    K = Kn;
end
fprintf('CARE:   K* = [%8.5f %8.5f %8.5f]\n', Kstar);

% Closed loop with the learned gain for 10 s from x(0) = [10 -10 -3]
[tc, xc] = ode45(@(t, z) (A - B*K)*z, 0:0.01:10, [10; -10; -3]);
figure('Position', [100 100 600 520]);
subplot(2,1,1); plot(0:size(Khist,1)-1, Khist, '-o', 'LineWidth', 1.3); grid on; hold on;
plot([0 size(Khist,1)-1], [Kstar; Kstar], 'k:');
xlabel('Iteration j'); ylabel('K_j'); legend('K_1', 'K_2', 'K_3', 'Location', 'east');
title('IRL policy iteration (dotted: CARE)');
subplot(2,1,2); plot(tc, xc, 'LineWidth', 1.3); grid on;
xlabel('Time (s)'); ylabel('x(t)'); legend('x_1', 'x_2', 'x_3'); title('Closed loop with the learned gain');


% Save the figure
if ~exist('figures', 'dir'), mkdir('figures'); end
exportgraphics(gcf, fullfile('figures', 'ct_irl_lqr.png'), 'Resolution', 150);

function b = quad_basis(x)
b = [x(1)^2; 2*x(1)*x(2); 2*x(1)*x(3); x(2)^2; 2*x(2)*x(3); x(3)^2];
end
function P = basis_to_sym(p)
P = [p(1) p(2) p(3); p(2) p(4) p(5); p(3) p(5) p(6)];
end
function x = rk4(f, x, h)
k1 = f(x); k2 = f(x + h/2*k1); k3 = f(x + h/2*k2); k4 = f(x + h*k3);
x = x + h/6*(k1 + 2*k2 + 2*k3 + k4);
end
