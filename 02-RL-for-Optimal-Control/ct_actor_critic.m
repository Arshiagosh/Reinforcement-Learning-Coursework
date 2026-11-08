%% Actor-critic (integral policy iteration) for a continuous-time nonlinear system
% HW8, Q1. Plant (input-affine):
%   x1' = x2
%   x2' = -x1 (pi/2 + atan(5 x1)) - 5 x1^2 / (2 (1 + 25 x1^2)) + 4 x2 + 3 u
% Cost int x'Qx + u'Ru with Q = diag(0, 1), R = 1.
% This example has a known optimum, V*(x) = x1^2 (pi/2 + atan(5 x1)) + x2^2 and
% u*(x) = -3 x2, so the learned weights can be checked exactly:
%   critic V = Wc' phi(x), phi = [x1 x2 x1x2 x1^2 x2^2 x1^2 atan(5x1) x1^3]
%          -> ideal Wc = [0 0 0 pi/2 1 1 0]
%   actor  u = Wa sigma(x), sigma = x2  -> ideal Wa = -3.
% Continuous-time version of the discrete actor-critic: at iteration j
%   critic: Wc' (phi(x(t)) - phi(x(t+T))) = int_t^{t+T} x'Qx + u'Ru dtau,  u = Wa_j x2
%           (least squares over measured intervals; f is never used)
%   actor:  Wa_{j+1} = argmin sum ( Wa sigma(x) + 1/2 R^-1 g' dphi(x)' Wc )^2
%           (least squares over the visited states; uses g only)
clear; close all; rng(3);

Q = diag([0 1]); R = 1; g = [0; 3];
f = @(x) [x(2); -x(1)*(pi/2 + atan(5*x(1))) - 5*x(1)^2/(2*(1 + 25*x(1)^2)) + 4*x(2)];
Wstar = [0 0 0 pi/2 1 1 0]';
dt = 1e-3; T = 0.05; nInt = round(T/dt); nEq = 80;
Wa = -2;            % initial admissible actor: x2 needs gain below -4/3 to be stable
WaHist = Wa; WcHist = zeros(7,1);
for j = 1:12
    Phi = zeros(nEq, 7); y = zeros(nEq,1); S = []; G = [];
    x = 2*rand(2,1) - 1;
    for k = 1:nEq
        p0 = phi(x); c = 0;
        for s = 1:nInt
            u  = Wa*x(2);
            xn = rk4(@(z) f(z) + g*(Wa*z(2)), x, dt);
            un = Wa*xn(2);
            c  = c + dt/2*(x'*Q*x + R*u^2 + xn'*Q*xn + R*un^2);
            x  = xn;
        end
        Phi(k,:) = (p0 - phi(x))'; y(k) = c;
        S(end+1,1) = x(2); G(end+1,:) = -0.5/R*g'*dphi(x)'; %#ok<AGROW>
        if mod(k, 4) == 0, x = 2*rand(2,1) - 1; end   % new trial: keep the data rich
    end
    Wc = Phi \ y;                               % critic (policy evaluation)
    WaN = S \ (G*Wc);                           % actor (policy improvement)
    fprintf('j = %2d  Wa = %8.5f  |Wc - Wc*| = %.2e\n', j, WaN, norm(Wc - Wstar));
    WaHist(end+1) = WaN; WcHist(:,end+1) = Wc; %#ok<SAGROW>
    if abs(WaN - Wa) < 1e-4, Wa = WaN; break; end
    Wa = WaN;
end
fprintf('Wc    = [%s]\nideal = [%s]\n', num2str(Wc', '%8.4f'), num2str(Wstar', '%8.4f'));

% Closed loop with the learned actor from x(0) = [1; -1]
[tc, xc] = ode45(@(t, z) f(z) + g*(Wa*z(2)), 0:0.01:8, [1; -1]);
it = 0:numel(WaHist)-1;
figure('Position', [100 100 640 640]);
subplot(3,1,1); plot(it(2:end), WcHist(:,2:end)', '-o', 'LineWidth', 1.2); grid on;
xlabel('Iteration j'); ylabel('W_c'); title('Critic weights');
legend('x_1', 'x_2', 'x_1x_2', 'x_1^2', 'x_2^2', 'x_1^2atan(5x_1)', 'x_1^3', 'Location', 'eastoutside');
subplot(3,1,2); plot(it, WaHist, '-o', 'LineWidth', 1.2); grid on; hold on;
plot(it([1 end]), [-3 -3], 'k--'); xlabel('Iteration j'); ylabel('W_a'); title('Actor weight (dashed: -3)');
subplot(3,1,3); plot(tc, xc, tc, Wa*xc(:,2), 'LineWidth', 1.2); grid on;
xlabel('Time (s)'); legend('x_1', 'x_2', 'u'); title('Closed loop with the learned actor');


% Save the figure
if ~exist('figures', 'dir'), mkdir('figures'); end
exportgraphics(gcf, fullfile('figures', 'ct_actor_critic.png'), 'Resolution', 150);

function p = phi(x)
x1 = x(1); x2 = x(2);
p = [x1; x2; x1*x2; x1^2; x2^2; x1^2*atan(5*x1); x1^3];
end
function G = dphi(x)
x1 = x(1); x2 = x(2); a = atan(5*x1);
G = [1 0; 0 1; x2 x1; 2*x1 0; 0 2*x2; 2*x1*a + 5*x1^2/(1 + 25*x1^2) 0; 3*x1^2 0];
end
function x = rk4(f, x, h)
k1 = f(x); k2 = f(x + h/2*k1); k3 = f(x + h/2*k2); k4 = f(x + h*k3);
x = x + h/6*(k1 + 2*k2 + 2*k3 + k4);
end
