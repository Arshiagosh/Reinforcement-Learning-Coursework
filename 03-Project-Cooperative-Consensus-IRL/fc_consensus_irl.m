function res = fc_consensus_irl(example, varargin)
%FC_CONSENSUS_IRL Model-free off-policy IRL for the fully cooperative
%consensus problem of continuous-time multi-agent systems.
%
%   res = fc_consensus_irl('five')    % Wang & Li (2020), Sec. V-A.1, Fig. 1 graph
%   res = fc_consensus_irl('twenty')  % Wang & Li (2020), Sec. V-A.2, Fig. 4 graph
%   res = fc_consensus_irl('linear')  % linear check: the optimum is known (CARE)
%
%   Implements Algorithm 3 of H. Wang and M. Li, "Model-Free Reinforcement
%   Learning for Fully Cooperative Consensus Problem of Nonlinear Multiagent
%   Systems", IEEE TNNLS, 2020. Data are collected once with an exploratory
%   behaviour input; policy iteration then solves the off-policy integral
%   Bellman equation (20) for the critic and all actor weights at once, by
%   least squares over many intervals [t, t+T]. No model (f, g) is used in
%   the learning step; the model is only used to simulate the agents.
%
%   Options (name/value): 'seed' (1), 'nRuns' (60), 'Tdata' (3), 'T' (0.05),
%   'amp' (0.4), 'maxIter' (60), 'tol' (1e-4), 'plot' (true), 'Tsim' (16),
%   'x0spread' (0), 'x0range' (0.6).
%
%   Data runs: as in step 1 of Algorithm 3, the same networked system is
%   excited with different behaviour inputs. Each run starts at
%   P.x0 + x0spread*randn (x0spread = 0: every run starts at P.x0 and only
%   the sinusoids differ). A negative x0spread draws x0 uniformly in
%   [-x0range, x0range] instead. The value of the nonlinear agents depends
%   on x and not only on delta, so a critic V(delta) fitted on random
%   starting states does not match the cost from P.x0 well (17% error in
%   the five-agent example, against 0.3% with data from P.x0).

opt = struct('seed', 1, 'nRuns', 60, 'Tdata', 3, 'T', 0.05, 'amp', 0.4, ...
             'x0range', 0.6, 'x0spread', 0, 'maxIter', 60, 'tol', 1e-4, ...
             'plot', true, 'Tsim', 16);
for k = 1:2:numel(varargin), opt.(varargin{k}) = varargin{k+1}; end
if nargin < 1, example = 'five'; end
rng(opt.seed);
P = setup(example);
N = P.N; n = 2; dt = 1e-3; nInt = round(opt.T/dt);
for i = 1:N, P.na(i) = numel(actor_basis(zeros(n,N), zeros(n,N), i, P)); end

%% 1) Collect data once, with an exploratory behaviour input u = sum of sinusoids
D = struct('dphi', {}, 'Qint', {}, 'uPsi', {}, 'PsiPsi', {});
for r = 1:opt.nRuns
    if opt.x0spread >= 0
        x = P.x0 + opt.x0spread*randn(n,N);
    else
        x = (rand(n,N) - 0.5)*2*opt.x0range;
    end
    w = 0.5 + 10*rand(N,6); ph = 2*pi*rand(N,6); t = 0;
    for k = 1:round(opt.Tdata/opt.T)
        phi0 = critic_basis(delta(x,P), x, P);
        S.Qint = 0;
        for i = 1:N, S.uPsi{i} = zeros(P.na(i),1); S.PsiPsi{i} = zeros(P.na(i)); end
        for s = 1:nInt
            u  = opt.amp*sum(sin(w*t + ph), 2)';        % held over the step
            d0 = delta(x,P);
            xn = rk4(@(xx) dynamics(xx, u, P), x, dt);
            d1 = delta(xn,P);
            S.Qint = S.Qint + dt*0.25*(P.Qfun(d0) + P.Qfun(d1));   % trapezoid
            for i = 1:N
                p0 = actor_basis(d0, x, i, P); p1 = actor_basis(d1, xn, i, P);
                S.uPsi{i}   = S.uPsi{i}   + dt*P.R(i)*u(i)*0.5*(p0 + p1);
                S.PsiPsi{i} = S.PsiPsi{i} + dt*P.R(i)*0.5*(p0*p0' + p1*p1');
            end
            x = xn; t = t + dt;
        end
        D(end+1).dphi = phi0 - critic_basis(delta(x,P), x, P); %#ok<AGROW>
        D(end).Qint = S.Qint; D(end).uPsi = S.uPsi; D(end).PsiPsi = S.PsiPsi;
    end
end
M = numel(D); hc = numel(D(1).dphi); nW = hc + sum(P.na);

%% 2) Policy iteration on the stored data (Algorithm 3 with the LS form of (23))
Wa = cell(N,1); for i = 1:N, Wa{i} = zeros(P.na(i),1); end   % u^0 = 0 is admissible
Wc = zeros(hc,1); hist = []; Whist = [];
for it = 1:opt.maxIter
    A = zeros(M, nW); b = zeros(M,1);
    for m = 1:M
        row = D(m).dphi'; pik = D(m).Qint;                % V(t) - V(t+T),  int Q/2
        for i = 1:N
            eta = D(m).uPsi{i}' - Wa{i}'*D(m).PsiPsi{i};   % int (u_i - u_i^k) R psi_i'
            row = [row, -eta]; %#ok<AGROW>
            pik = pik + 0.5*Wa{i}'*D(m).PsiPsi{i}*Wa{i};   % int u_i^k' R u_i^k / 2
        end
        A(m,:) = row; b(m) = pik;
    end
    % delta = (L kron I) x lives in a subspace, so some basis columns are
    % linearly dependent: take the minimum-norm least-squares solution.
    W = pinv(A, 1e-6*norm(A))*b;
    dW = norm(W(1:hc) - Wc); Wc = W(1:hc); off = hc;
    for i = 1:N
        new = W(off+1:off+P.na(i)); off = off + P.na(i);
        dW = dW + norm(new - Wa{i}); Wa{i} = new;
    end
    hist(end+1) = dW; Whist(:,end+1) = W; %#ok<AGROW>
    if dW < opt.tol, break; end
end
res.Wc = Wc; res.Wa = Wa; res.iterations = numel(hist); res.dW = hist; res.Whist = Whist;
res.residual = norm(A*W - b)/norm(b); res.P = P; res.nEquations = M;

%% 3) Closed-loop test from P.x0 with the learned actors (no noise)
tt = 0:dt:opt.Tsim; nt = numel(tt);
x = P.x0; J = zeros(1,nt); Vh = zeros(1,nt); dl = zeros(n*N, nt); U = zeros(N, nt); Jc = 0;
for k = 1:nt
    d = delta(x,P); dl(:,k) = d(:); Vh(k) = Wc'*critic_basis(d, x, P);
    u = zeros(1,N); for i = 1:N, u(i) = Wa{i}'*actor_basis(d, x, i, P); end
    U(:,k) = u'; J(k) = Jc;
    Jc = Jc + dt*(0.5*P.Qfun(d) + 0.5*sum(P.R(:)'.*u.^2));
    x = rk4(@(xx) dynamics(xx, u, P), x, dt);
end
x = P.x0; J0 = 0;                                        % same start, u = 0
for k = 1:nt, J0 = J0 + dt*0.5*P.Qfun(delta(x,P)); x = rk4(@(xx) dynamics(xx, zeros(1,N), P), x, dt); end
res.t = tt; res.delta = dl; res.u = U; res.Vhat = Vh; res.J = J; res.Jzero = J0;
res.consensusTime = tt(find(max(abs(dl),[],1) > 0.01, 1, 'last'));
fprintf('[%s] %d intervals, %d PI iterations, LS residual %.3f\n', example, M, res.iterations, res.residual);
fprintf('       Vhat(delta(0)) = %.4f, J = %.4f (u = 0 gives %.4f), |delta| < 0.01 after %.2f s\n', ...
        Vh(1), J(end), J0, res.consensusTime);
if isfield(P, 'A')      % linear check: compare with the centralised CARE optimum
    Pr = care(kron(eye(N), P.A), kron(P.L, P.B), eye(n*N), diag(P.R));
    d0 = delta(P.x0, P); res.Vstar = 0.5*d0(:)'*Pr*d0(:);
    fprintf('       optimal V*(delta(0)) from CARE = %.4f\n', res.Vstar);
end
if opt.plot, plot_results(res, example); end
end

%% ------------------------------------------------------------------------
function P = setup(example)
switch example
    case {'five', 'linear'}
        P.N = 5; E = zeros(5);                 % Fig. 1: ring 1->2->3->4->5->1 plus 3->1
        E(2,1) = 1; E(3,2) = 1; E(4,3) = 1; E(5,4) = 1; E(1,5) = 1; E(1,3) = 1;
        P.x0 = [0.2 -0.4 0.5 0.1 -0.3; 0.3 0.1 -0.2 0.4 -0.1];
        P.R = 10*ones(5,1);
        if strcmp(example, 'five')
            P.Qfun = @(d) 0.5*d(1,1)^4 + 0.3*d(2,1)^2 + 0.25*sin(d(1,2))^2 + 2*d(2,2)^2 ...
                        + 2*d(1,3)^4 + 0.3*d(2,3)^2 + 2*d(1,4)^4 + 0.6*d(2,4)^2 + d(1,5)^4 + d(2,5)^2;
        else
            P.A = [0 1; -1 -1]; P.B = [1; 0.5]; P.Qfun = @(d) sum(d(:).^2);
        end
    case 'twenty'
        P.N = 20; E = zeros(20);               % Fig. 4: two directed 10-rings joined 8<->11
        for k = 1:10
            E(mod(k,10)+1, k) = 1; E(10 + mod(k,10)+1, 10 + k) = 1;
        end
        E(11,8) = 1; E(8,11) = 1;
        P.x0 = 0.5*[sin(1:20); cos(1.7*(1:20))];
        P.R = 200*ones(20,1); P.Qfun = @(d) sum(d(:).^2);
    otherwise
        error('Unknown example "%s".', example);
end
P.E = E; P.L = diag(sum(E,2)) - E;
P.mask = kron(double((E + E' + eye(P.N)) > 0), ones(2));   % neighbour pairs
for i = 1:P.N, P.out{i} = find(E(:,i))'; end              % agents that listen to i
P.quartic = strcmp(example, 'five');
end

function d = delta(x, P)
d = x*P.L';                 % column i: sum_j e_ij (x_i - x_j), eq. (2)
end

function phi = critic_basis(d, ~, P)
% All quadratic terms of the error components; quartic terms when Q itself
% is quartic (five-agent example). The value function couples agents that
% are not neighbours, so neighbour-only products are not enough: in the
% twenty-agent example they left a 7.7% gap between V(delta(0)) and J.
v = d(:); [I, J] = find(triu(ones(numel(v)))); phi = v(I).*v(J);
if P.quartic, phi = [phi; v.^4; (d(1,:).*d(2,:))'.^2]; end
end

function psi = actor_basis(d, x, i, P)
% Agent i uses its own error and the errors of the agents that listen to it,
% as the optimal policy (7) does. The sin(x_i1) copy lets the actor follow
% the input gain of the nonlinear agents; it uses agent i's own state only.
v = reshape(d(:, [i, P.out{i}]), [], 1);
if isfield(P, 'A'), psi = v; else, psi = [v; sin(x(1,i))*v]; end
end

function dx = dynamics(x, u, P)
if isfield(P, 'A'), dx = P.A*x + P.B*u; return; end
x1 = x(1,:); x2 = x(2,:);                                  % eq. (36)
dx = [-x1 + x2 + 2*x2.^3;
      -0.5*(x1 + x2) + 0.5*x2.*(1 + 2*x2.^2).*sin(x1).^2 + sin(x1).*u];
end

function x = rk4(f, x, h)
k1 = f(x); k2 = f(x + h/2*k1); k3 = f(x + h/2*k2); k4 = f(x + h*k3);
x = x + h/6*(k1 + 2*k2 + 2*k3 + k4);
end

function plot_results(res, example)
N = res.P.N; t = res.t;
figure('Name', [example ' errors'], 'Position', [100 100 640 480]);
for c = 1:2
    subplot(2,1,c); plot(t, res.delta(c:2:end, :), 'LineWidth', 1.2); grid on;
    ylabel(sprintf('\\delta_i(%d)', c)); if c == 2, xlabel('Time (s)'); end
end
subplot(2,1,1); title(sprintf('Local neighbourhood errors, %d agents', N));
figure('Name', [example ' value'], 'Position', [760 100 560 320]);
plot(t, res.Vhat, t, res.J, 'LineWidth', 1.5); grid on; xlabel('Time (s)');
legend('$\hat V(\delta(t))$', '$J(t)=\int_0^t r\,d\tau$', 'Interpreter', 'latex', 'Location', 'east');
title('Critic estimate vs. accumulated cost');
end
