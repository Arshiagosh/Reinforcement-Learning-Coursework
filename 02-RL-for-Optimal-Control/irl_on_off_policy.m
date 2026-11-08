%% On-policy and off-policy IRL for an unstable continuous-time plant
% Comprehensive exercise, Q2 (a)-(c). Plant xdot = A x + B u, y = [1 0] x,
% A = [0.5 1.5; 2 -2] (unstable), B = [1; 4], Q = I, R = 1, x(0) = [5; -5].
%
% (a) On-policy IRL (Vrabie et al., 2009): run u = -K_j x + e (probing
%     noise e in [0, 0.1]) and solve, over intervals [t, t+T],
%       x(t)'P_j x(t) - x(t+T)'P_j x(t+T) = int x'Qx + u'Ru dtau
%     then K_{j+1} = R^-1 B' P_j.
% (c) Off-policy IRL (Jiang and Jiang, 2012): data come ONCE from a fixed
%     stabilising behaviour policy u = -K_b x + e. For each target K_j,
%       x(t+T)'P_j x(t+T) - x(t)'P_j x(t) =
%         -int x'(Q + K_j'R K_j) x dtau + 2 int (u + K_j x)' R K_{j+1} x dtau
%     is linear in (P_j, K_{j+1}); neither A nor B is used.
clear; close all; rng(4);

A = [0.5 1.5; 2 -2]; B = [1; 4]; Q = eye(2); R = 1; n = 2;
[Pstar, ~, Kstar] = care(A, B, Q, R);
K0 = [1 1];                                  % stabilising: eig(A - B K0) = {-1.5 +- ...}
assert(all(real(eig(A - B*K0)) < 0));
dt = 1e-3; T = 0.05; nInt = round(T/dt); x0 = [5; -5];

%% (a) on-policy
K = K0; onK = K; onErr = [];
for j = 1:20
    nEq = 40; Phi = zeros(nEq, 3); y = zeros(nEq, 1); x = x0;
    for k = 1:nEq
        b0 = qb(x); c = 0;
        for s = 1:nInt
            e  = 0.1*rand;                               % probing noise in [0, 0.1]
            u  = -K*x + e;
            xn = rk4(@(z) A*z + B*u, x, dt);
            c  = c + dt/2*(x'*Q*x + xn'*Q*xn) + dt*R*u^2;
            x  = xn;
        end
        Phi(k,:) = (b0 - qb(x))'; y(k) = c;
        if mod(k, 8) == 0, x = 5*randn(n,1); end
    end
    P = s2m(Phi \ y); Kn = R \ (B'*P);
    onK(end+1,:) = Kn; onErr(end+1) = norm(P - Pstar); %#ok<SAGROW>
    fprintf('on-policy  j = %2d  K = [%8.4f %8.4f]  |P - P*| = %.2e\n', j, Kn, onErr(end));
    if norm(Kn - K) < 1e-4, K = Kn; break; end
    K = Kn;
end
Kon = K;

%% (c) off-policy: one batch of behaviour data, then iterate offline
Kb = K0; nEq = 300; x = x0; t = 0;
D = struct('db', {}, 'Ixx', {}, 'Iux', {});
for k = 1:nEq
    S.db = qb(x); S.Ixx = zeros(n); S.Iux = zeros(1,n);
    for s = 1:nInt
        e  = 0.1*rand + 0.5*sin(7*t) + 0.5*sin(13*t);   % probing noise
        u  = -Kb*x + e;
        xn = rk4(@(z) A*z + B*u, x, dt);
        S.Ixx = S.Ixx + dt/2*(x*x' + xn*xn');
        S.Iux = S.Iux + dt/2*u*(x + xn)';
        x = xn; t = t + dt;
    end
    D(k).db = qb(x) - S.db; D(k).Ixx = S.Ixx; D(k).Iux = S.Iux;
    if mod(k, 40) == 0, x = 5*randn(n,1); end
end
K = K0; offK = K; offErr = [];
for j = 1:20
    Phi = zeros(nEq, 3 + n); y = zeros(nEq, 1);
    for k = 1:nEq
        Phi(k,:) = [D(k).db', -2*R*(D(k).Iux + K*D(k).Ixx)];
        y(k) = -trace((Q + K'*R*K)*D(k).Ixx);
    end
    th = Phi \ y; P = s2m(th(1:3)); Kn = th(4:5)';
    offK(end+1,:) = Kn; offErr(end+1) = norm(P - Pstar); %#ok<SAGROW>
    fprintf('off-policy j = %2d  K = [%8.4f %8.4f]  |P - P*| = %.2e\n', j, Kn, offErr(end));
    if norm(Kn - K) < 1e-4, K = Kn; break; end
    K = Kn;
end
Koff = K;
fprintf('CARE:                K* = [%8.4f %8.4f]\n', Kstar);

%% closed loops after learning (noise removed)
[t1, x1] = ode45(@(t, z) (A - B*Kon)*z, 0:0.01:5, x0);
[t2, x2] = ode45(@(t, z) (A - B*Koff)*z, 0:0.01:5, x0);
figure('Position', [100 100 700 520]);
subplot(2,2,1); semilogy(1:numel(onErr), onErr, '-o', 'LineWidth', 1.2); grid on;
xlabel('Iteration j'); ylabel('$\|P_j - P^*\|$', 'Interpreter', 'latex'); title('On-policy IRL');
subplot(2,2,2); semilogy(1:numel(offErr), offErr, '-s', 'LineWidth', 1.2); grid on;
xlabel('Iteration j'); ylabel('$\|P_j - P^*\|$', 'Interpreter', 'latex'); title('Off-policy IRL');
subplot(2,2,3); plot(t1, x1, 'LineWidth', 1.3); grid on; xlabel('Time (s)'); legend('x_1', 'x_2');
title('States, on-policy gain');
subplot(2,2,4); plot(t2, x2, 'LineWidth', 1.3); grid on; xlabel('Time (s)'); legend('x_1', 'x_2');
title('States, off-policy gain');


% Save the figure
if ~exist('figures', 'dir'), mkdir('figures'); end
exportgraphics(gcf, fullfile('figures', 'irl_on_off_policy.png'), 'Resolution', 150);

function b = qb(x),  b = [x(1)^2; 2*x(1)*x(2); x(2)^2]; end
function P = s2m(p), P = [p(1) p(2); p(2) p(3)]; end
function x = rk4(f, x, h)
k1 = f(x); k2 = f(x + h/2*k1); k3 = f(x + h/2*k2); k4 = f(x + h*k3);
x = x + h/6*(k1 + 2*k2 + 2*k3 + k4);
end
