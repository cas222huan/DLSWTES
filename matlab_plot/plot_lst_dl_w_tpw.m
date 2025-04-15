%%
clc;
clear;
load('data/lst_dl_w_tpw.mat');
lsts = [lst_dl; lst_dl_noisy];
qas = [qa_dl; qa_dl_noisy];

%%
figure;
fig = gcf;
fig.Position = [0 0 849 361]; % [left bottom width height]
t = tiledlayout(1,2, 'TileSpacing', 'compact', 'Padding', 'compact');

titles = ["DL-SW-TES without uncertainties", "DL-SW-TES with uncertainties"];

set(0, 'DefaultAxesFontName', 'Times New Roman');
set(0, 'DefaultTextFontName', 'Times New Roman');
font_size = 13;
set(0, 'DefaultAxesFontSize', font_size);
set(0, 'DefaultTextFontSize', font_size);

for i=1:2
    nexttile;
    ax = gca;
    
    qa = qas(i,:);
    x = lst_true(qa==1);
    y = lsts(i, qa==1);

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

    % add text
    % text(0.9, 0.08, "("+char(96+i)+")", 'Units', 'normalized');
    
    % calculate accuracy metrics
    rmse = sqrt(mean((y-x).^2));
    bias = mean(y-x);
    str = sprintf('RMSE = %.2f\nbias = %.2f\nN = %d', rmse, bias, length(x));
    text(0.05, 0.85, str, 'Units', 'normalized');

    % set colorbar
    if i==2
        cb = colorbar('Location', 'eastoutside');
        cb.Ticks = cb.Limits;
        cb.TickLabels = {'Sparse', 'Dense'};
        cb.Label.String = 'Density';
        cb.Label.Position(1) = 3;
        cb.FontSize = font_size;
    end
    
    % set ticks
    ticks = 170:30:350;
    set(gca, 'XTick', ticks, 'YTick', ticks);
    
    title("("+char(96+i)+") "+titles(i), 'FontWeight','bold');
    ax.Box = 'on';
end

%%
exportgraphics(gcf, '../figs/lst_dl_w_tpw.tif', 'Resolution', 600);