%% Reproduce the three examples and save the figures
% Runs the linear check, the five-agent example (paper Sec. V-A.1) and the
% twenty-agent example (Sec. V-A.2). The twenty-agent run uses 60 data runs
% (180 s of data) because its critic has 820 quadratic terms.
clear; close all;
if ~exist('figures', 'dir'), mkdir('figures'); end
cases = {'linear', {}; 'five', {}; 'twenty', {'nRuns', 60}};
for c = 1:size(cases, 1)
    name = cases{c, 1};
    res = fc_consensus_irl(name, cases{c, 2}{:});
    figs = findobj('Type', 'figure');
    for f = figs'
        tag = strrep(get(f, 'Name'), ' ', '_');
        exportgraphics(f, fullfile('figures', [tag '.png']), 'Resolution', 150);
    end
    close all;
    results.(name) = res; %#ok<STRNU>
end
