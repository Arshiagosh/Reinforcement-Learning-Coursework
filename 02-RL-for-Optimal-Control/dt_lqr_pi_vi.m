%% Discrete-time LQR by policy iteration, value iteration and lambda-PI
% HW6, Q1. Plant x(k+1) = A x(k) + B u(k), cost sum x'Qx + u'Ru, u = -K x.
clear; close all;

A = [0.9065 0.0816 -0.0005; 0.0743 0.9012 -0.0007; 0 0 0.1327];
B = [-0.0027; -0.0068; 1];
Q = eye(3); R = 1; n = 3;

% (a) Model-based reference from the Riccati equation
[Pstar, ~, Kstar] = dare(A, B, Q, R);     % u = -Kstar x
fprintf('DARE:  K* = [%s]\n', num2str(Kstar, '%9.5f'));

% (b)-(d) Iterative solutions. A is Schur stable, so K0 = 0 is admissible.
K0 = zeros(1, n); P0 = zeros(n); maxIt = 3000; tol = 1e-10;
methods = {'PI', 'VI', 'lambda-PI 0.5', 'lambda-PI 0.9'};
lambdas = [1, 0, 0.5, 0.9];   % lambda = 1 is PI, lambda = 0 is VI
out = struct();
for m = 1:numel(methods)
    lam = lambdas(m); K = K0; P = P0; dP = []; dK = [];
    for j = 1:maxIt
        Acl = A - B*K;
        % P_{j+1} = (1-lam) Acl' P_j Acl + lam Acl' P_{j+1} Acl + K'RK + Q
        rhs = (1 - lam)*(Acl'*P*Acl) + Q + K'*R*K;
        if lam == 0
            Pn = rhs;                                   % value iteration
        else
            Pn = dlyap(sqrt(lam)*Acl', rhs);            % Pn - lam Acl'Pn Acl = rhs
        end
        Kn = (R + B'*Pn*B) \ (B'*Pn*A);                 % policy improvement
        dP(end+1) = norm(Pn - P); dK(end+1) = norm(Kn - K); %#ok<SAGROW>
        P = Pn; K = Kn;
        if dP(end) < tol && dK(end) < tol, break; end
    end
    out(m).name = methods{m}; out(m).iters = j; out(m).dP = dP; out(m).dK = dK;
    out(m).errK = norm(K - Kstar); out(m).errP = norm(P - Pstar);
    fprintf('%-14s %4d iterations, |K-K*| = %.1e, |P-P*| = %.1e\n', methods{m}, j, out(m).errK, out(m).errP);
end

% (e) Convergence plots
figure('Position', [100 100 700 520]);
mk = {'-o', '-', '-', '-'};
subplot(2,1,1);
for m = 1:4, semilogy(out(m).dK, mk{m}, 'MarkerSize', 3, 'LineWidth', 1.2); hold on; end
grid on; ylabel('$\|K_{j+1}-K_j\|$', 'Interpreter', 'latex'); legend(methods, 'Location', 'northeast');
title('Convergence of the feedback gain');
subplot(2,1,2);
for m = 1:4, semilogy(out(m).dP, mk{m}, 'MarkerSize', 3, 'LineWidth', 1.2); hold on; end
grid on; ylabel('$\|P_{j+1}-P_j\|$', 'Interpreter', 'latex'); xlabel('Iteration j');
title('Convergence of the value matrix');

% Save the figure
if ~exist('figures', 'dir'), mkdir('figures'); end
exportgraphics(gcf, fullfile('figures', 'dt_lqr_pi_vi.png'), 'Resolution', 150);
