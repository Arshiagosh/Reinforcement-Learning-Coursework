%% Model-free off-policy RL for H-infinity state feedback
% HW7, Q2. Plant  xdot = A x + B1 u + B2 d,  attenuation level beta = 5,
%   min_u max_d  int x'Qx + u'Ru - beta^2 d'd.
% (a) Model-based: the game algebraic Riccati equation (GARE)
%     A'P + PA + Q - P B1 R^-1 B1' P + beta^-2 P B2 B2' P = 0.
% (b) Off-policy RL (Luo, Wu and Huang, 2015 / Modares, Lewis and Jiang, 2015):
%     data come once from a behaviour input u = -K1_0 x + e, d = e_d. With
%     target policies u_j = -K1_j x, d_j = K2_j x and V_j = x'P_j x, along the
%     behaviour trajectories
%     x(t+T)'P_j x(t+T) - x(t)'P_j x(t) =
%        int [ -x'(Q + K1_j'R K1_j - beta^2 K2_j'K2_j) x
%              + 2 (u + K1_j x)' R K1_{j+1} x + 2 beta^2 (d - K2_j x)' K2_{j+1} x ] dtau
%     which is linear in (P_j, K1_{j+1}, K2_{j+1}) and needs neither A, B1 nor B2.
clear; close all; rng(2);

A  = [-1.01887 0.90506 -0.00215; 0.82225 -1.07741 -0.17555; 0 0 -1];
B1 = [0; 0; 1]; B2 = [1; 0; 0]; Q = eye(3); R = 1; beta = 5; n = 3;

%% (a) GARE through the stable invariant subspace of the Hamiltonian
Pg = gare(A, [B1 B2], Q, blkdiag(R, -beta^2));
Pc = care(A, B1, Q, R);                 % plain LQR, for comparison (HW7 Q1)
K1g = R \ (B1'*Pg); K2g = (B2'*Pg)/beta^2;
fprintf('GARE:  K1 = [%s]   K2 = [%s]\n', num2str(K1g, '%9.5f'), num2str(K2g, '%9.5f'));
fprintf('|P_GARE - P_LQR| = %.3e (the disturbance makes P larger)\n', norm(Pg - Pc));

%% (b) collect data once with exploring behaviour policies
dt = 1e-3; T = 0.02; nInt = round(T/dt); nEq = 400;
w1 = 1 + 20*rand(1,8); w2 = 1 + 20*rand(1,8); p1 = 2*pi*rand(1,8); p2 = 2*pi*rand(1,8);
K10 = zeros(1,n);                                   % admissible (A Hurwitz)
x = [10; -10; -3]; t = 0;
D = struct('db', {}, 'Ixx', {}, 'Iux', {}, 'Idx', {});
for k = 1:nEq
    S.db = quad_basis(x); S.Ixx = zeros(n); S.Iux = zeros(1,n); S.Idx = zeros(1,n);
    for s = 1:nInt
        e  = 0.5*sum(sin(w1*t + p1));               % probing noise on u
        dd = 0.5*sum(sin(w2*t + p2));               % excitation on d
        u  = -K10*x + e;
        xn = rk4(@(z) A*z + B1*u + B2*dd, x, dt);   % u, d held over the step
        S.Ixx = S.Ixx + dt/2*(x*x' + xn*xn');
        S.Iux = S.Iux + dt/2*u*(x + xn)';
        S.Idx = S.Idx + dt/2*dd*(x + xn)';
        x = xn; t = t + dt;
    end
    D(k).db = quad_basis(x) - S.db; D(k).Ixx = S.Ixx; D(k).Iux = S.Iux; D(k).Idx = S.Idx;
end

%% (b) off-policy policy iteration on the stored data
K1 = K10; K2 = zeros(1,n); P = zeros(n); hist = [];
for j = 1:30
    Phi = zeros(nEq, 6 + 2*n); y = zeros(nEq,1);
    for k = 1:nEq
        % unknowns: [p (6); K1_{j+1}' (3); K2_{j+1}' (3)]
        a1 = 2*R*(D(k).Iux + K1*D(k).Ixx);           % int 2 (u + K1_j x)' R x'
        a2 = 2*beta^2*(D(k).Idx - K2*D(k).Ixx);      % int 2 beta^2 (d - K2_j x) x'
        Phi(k,:) = [D(k).db', -a1, -a2];
        y(k) = -trace((Q + K1'*R*K1 - beta^2*(K2'*K2))*D(k).Ixx);
    end
    th = Phi \ y;
    Pn = basis_to_sym(th(1:6)); K1n = th(7:9)'; K2n = th(10:12)';
    hist(end+1) = norm(Pn - P); %#ok<SAGROW>
    fprintf('j = %2d  |P_j - P_GARE| = %.2e\n', j, norm(Pn - Pg));
    P = Pn; K1 = K1n; K2 = K2n;
    if hist(end) < 1e-4, break; end            % stopping rule from the assignment
end
fprintf('RL:    K1 = [%s]   K2 = [%s]\n', num2str(K1, '%9.5f'), num2str(K2, '%9.5f'));

%% (c) closed loop from x(0) with the learned controller (no noise, d = 0)
[tc, xc] = ode45(@(t, z) (A - B1*K1)*z, 0:0.01:10, [10; -10; -3]);

%% (d) attenuation with w(t) = 0.2 e^{-0.2t} cos t from x(0) = 0
tw = 0:0.01:60;
[~, xw] = ode45(@(t, z) (A - B1*K1)*z + B2*0.2*exp(-0.2*t)*cos(t), tw, zeros(n,1));
uw = -(K1*xw')'; ww = 0.2*exp(-0.2*tw').*cos(tw');
num = cumtrapz(tw', sum((xw*Q).*xw, 2) + R*uw.^2); den = cumtrapz(tw', ww.^2);
rd = sqrt(num./max(den, eps)); rd(1) = 0;
fprintf('attenuation ratio at t = 60 s: %.3f (bound beta = %g)\n', rd(end), beta);

figure('Position', [100 100 640 560]);
subplot(3,1,1); semilogy(hist, '-o', 'LineWidth', 1.3); grid on;
ylabel('$\|P_{j}-P_{j-1}\|$', 'Interpreter', 'latex'); xlabel('Iteration j');
title('Off-policy H_\infty learning');
subplot(3,1,2); plot(tc, xc, 'LineWidth', 1.3); grid on;
legend('x_1', 'x_2', 'x_3'); xlabel('Time (s)'); ylabel('x(t)'); title('Closed loop after learning');
subplot(3,1,3); plot(tw, rd, 'LineWidth', 1.3); hold on; plot(tw([1 end]), [beta beta], 'k--'); grid on;
xlabel('Time (s)'); ylabel('r_d(t)'); legend('r_d(t)', '\beta = 5', 'Location', 'east');
title('Attenuation ratio for w(t) = 0.2 e^{-0.2t} cos t');


% Save the figure
if ~exist('figures', 'dir'), mkdir('figures'); end
exportgraphics(gcf, fullfile('figures', 'ct_offpolicy_hinf.png'), 'Resolution', 150);

function P = gare(A, B, Q, R)
% Stabilising solution of A'P + PA + Q - P B R^-1 B' P = 0 (R may be indefinite)
n = size(A,1); H = [A, -B/R*B'; -Q, -A'];
[V, E] = eig(H); V = V(:, real(diag(E)) < 0);
P = real(V(n+1:end,:)/V(1:n,:)); P = (P + P')/2;
end
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
