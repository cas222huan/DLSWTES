%%
clc;
clear;
alg = 'sw_tes'; % 'dl_wo_tpw' 'dl_w_tpw'
file = strcat('data/rad_', alg, '.mat');
load(file);

%%
figure;
fig = gcf;
fig.Position = [402.3333 124.3333 1.0253e+03 710]; %[675 535 1025 711]; % [left bottom width height]
t = tiledlayout(2,3, 'TileSpacing', 'compact', 'Padding', 'compact');

set(0, 'DefaultAxesFontName', 'Times New Roman');
set(0, 'DefaultTextFontName', 'Times New Roman');
font_size = 13;
set(0, 'DefaultAxesFontSize', font_size);
set(0, 'DefaultTextFontSize', font_size);

bands = [2, 4, 5];
tick_gap_1 = 5;
tick_gap_2 = 2;

for i=1:6
    nexttile;
    ax = gca;
    
    if i < 4
        x = Lg_true(:, i);
        y = Lg_prd(:,i); %Lg_dl(:, i);
        DensScat(x, y, 'logDensity', true, 'nBin_x', 2000, 'nBin_y', 2000, ...
            'ColorMap', 'jet', 'ColorBar', false, 'TargetAxes', ax, 'AxisType', 'normal');
        
        xlabel('True L^{grd} (W m^{-2} sr^{-1} µm^{-1})');
        ylabel('Estimated L^{grd} (W m^{-2} sr^{-1} µm^{-1})');
        title("("+char(96+i)+") " + 'ECOSTRESS Band ' + string(bands(i)), 'FontWeight','bold');
        
    else
        x = Ld_true(:, i-3);
        y = Ld_prd(:,i-3); % Ld_dl(:, i-3);
        DensScat(x, y, 'logDensity', true, 'nBin_x', 600, 'nBin_y', 600, ...
            'ColorMap', 'jet', 'ColorBar', false, 'TargetAxes', ax, 'AxisType', 'normal');
        xlabel('True L^{atm↓} (W m^{-2} sr^{-1} µm^{-1})');
        ylabel('Estimated L^{atm↓} (W m^{-2} sr^{-1} µm^{-1})');
        title("("+char(96+i)+") " + 'ECOSTRESS Band ' + string(bands(i-3)), 'FontWeight','bold');
        
    end

    % colormap(slanCM('jet'));
    
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
    str = sprintf('RMSE = %.3f\nbias = %.3f\nN = %d', rmse, bias, length(x));
    text(0.05, 0.83, str, 'Units', 'normalized', 'FontSize', 13);

    % set colorbar
    if i==5
        cb = colorbar('Location', 'southoutside');
        cb.Ticks = cb.Limits;
        cb.TickLabels = {'Sparse', 'Dense'};
        cb.Label.String = 'Density';
        cb.Label.Position(2) = -1;
        cb.FontSize = font_size;
    end
    
    % set ticks
    if i < 4
        tick_min = floor(min_xy/tick_gap_1)*tick_gap_1;
        tick_max = ceil(max_xy/tick_gap_1)*tick_gap_1;
        ticks = tick_min:tick_gap_1:tick_max;
    else
        tick_min = floor(min_xy/tick_gap_2)*tick_gap_2;
        tick_max = ceil(max_xy/tick_gap_2)*tick_gap_2;
        ticks = tick_min:tick_gap_2:tick_max;
    end

    set(gca, 'XTick', ticks, 'YTick', ticks);
    
    ax.Box = 'on';
end

%%
save_name = strcat('../figs/rad_', alg, '.tif');
exportgraphics(gcf, save_name, 'Resolution', 600);