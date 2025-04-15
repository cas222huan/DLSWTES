%%
clc;
clear;
alg = 'dl_wo_tpw'; % sw_tes % swdtes
file_lst = strcat('data/lst_', alg, '.mat');
file_rmse = strcat('data/rmse_bins_', alg, '.mat');
load(file_lst); % load('lst_swdtes.mat'); load('lst_wo_tpw.mat'); 
load(file_rmse); % load('rmse_swdtes.mat'); load('rmses_dl_wo_tpw.mat');

x = lst_true; %lst_true; 
y = lst_prd; %lst_swdtes; % lst_dl_noisy;

% Add an extra row and col
rmses(:,end+1) = 0;
rmses(end+1,:) = 0;

set(0, 'DefaultAxesFontName', 'Times New Roman');
set(0, 'DefaultTextFontName', 'Times New Roman');
font_size = 12;
set(0, 'DefaultAxesFontSize', font_size);
set(0, 'DefaultTextFontSize', font_size);

%%
figure;
fig = gcf;
fig.Position = [0 0 954 343]; % [left bottom width height]
t = tiledlayout(1,2, 'TileSpacing', 'compact', 'Padding', 'compact');

nexttile;
ax = gca;

DensScat(x, y, 'logDensity', true, 'nBin_x', 2000, 'nBin_y', 2000, ...
    'ColorMap', 'jet', 'ColorBar', false, 'TargetAxes', ax, 'AxisType', 'normal');

xlabel('True LST (K)');
ylabel('Estimated LST (K)');

% calculate value ranges
min_x = min(x);
max_x = max(x);
min_y = min(y);
max_y = max(y);
min_xy = min([min_x min_y]);
max_xy = max([max_x max_y]);

% set lims and plot 1:1 line
max_v = max_xy + (max_xy-min_xy) * 0.1;
min_v = min_xy - (max_xy-min_xy) * 0.1;

hold on;
plot([min_v, max_v], [min_v, max_v], 'LineWidth', 1.5, 'Color', 'black', 'LineStyle','--');

xlim([min_v, max_v]);
ylim([min_v, max_v]);

% calculate accuracy metrics
rmse = sqrt(mean((y-x).^2));
bias = mean(y-x);
str = sprintf('(a)\nRMSE = %.2f\nbias = %.2f\nN = %d', rmse, bias, length(x));
text(0.05, 0.84, str, 'Units', 'normalized');

% set ticks
ticks = 170:30:350;
set(gca, 'XTick', ticks, 'YTick', ticks);
ax.Box = 'on';

cb = colorbar('Location', 'eastoutside');
cb.Ticks = cb.Limits;
cb.TickLabels = {'Sparse', 'Dense'};
cb.Label.String = 'Density';
cb.Label.Position = [2.5 0.6929 0];
cb.FontSize = font_size;


nexttile;
ax = gca;
h = pcolor(rmses);
% h.EdgeColor = '#A7A7A8';
h.LineWidth = 0.7;
colormap(ax, slanCM('rainbow'));

for i = 1:size(rmses, 1)-1
    for j = 1:size(rmses, 2)-1
        if ~isnan(rmses(i,j))
            text(j+0.5, i+0.5, sprintf('%.2f', rmses(i,j)), 'HorizontalAlignment', 'center', ...
                'VerticalAlignment', 'middle', 'FontWeight', 'bold', 'Color', 'black', 'FontSize', font_size);
        end
    end
end

xlabel('LST (K)');
ylabel('PWV (cm)');
text(0.03, 0.94, '(b)', 'Units', 'normalized');

lst_edges = 180:20:350;
tick_labels = [num2cell(lst_edges), 350];
xticks(1:10);
xticklabels(tick_labels);
yticklabels(0:7);

% axis equal;
c = colorbar;
c.Label.String = 'RMSE (K)';
c.FontSize = font_size;
c.TickLabels = sprintfc('%.1f', c.Ticks);

%%
save_name = strcat('../figs/lst_', alg, '.tif');
exportgraphics(gcf, save_name, 'Resolution', 600);