%% plot_edge_area_adc_compare_byconfig_batch.m
% ========================================================================
% Batch plotting of vessel cross-sectional area and paravascular CSF ADC
% pulsatility curves across cardiac phases.
%
% Supports single-scan and dual-scan (scan-rescan reproducibility) modes.
% Config files can be .json (from vessel_config_editor.html) or .m (legacy).
%
% ---- How to use ----
%   1. Create config:
%      - Use vessel_config_editor.html → saves case_config_<NAME>.json
%      - Or write case_config_<NAME>.m manually (legacy)
%   2. Set CASE_LIST below to the case names you want to process
%   3. Run this script in MATLAB
%
% ---- Config format (JSON, from vessel_config_editor) ----
%   {
%     "case_id": "250701",
%     "base_dir": "/v/ai/nobackup/cni/chang_data/250701_960",
%     "scans": {
%       "s1": {
%         "name": "0701",
%         "mat_file": ".../paravascular_adc_v2.mat",
%         "output_folder": ".../layout_preview",
%         "segments_of_interest": [
%           {"seg_id": "EP_01—EP_02", "name": "M2_R", "trim_start": 5, "trim_end": 5}
%         ]
%       },
%       "s2": { ... }    % optional, for scan-rescan comparison
%     }
%   }
%
% ---- Input (.mat files, from Python step2/step3) ----
%   Each scan's mat_file contains:
%     - area_all  : (N_points x 25)  cross-sectional area per phase
%     - adc_all   : (N_points x 25)  paravascular CSF ADC per phase
%     - seg_ids   : (N_points x 1)   segment ID string for each point
%     - cl_mm     : (N_points x 3)   centerline positions (mm)
%     - pi_val    : (N_points x 1)   pulsatility index per point
%
% ---- Output (per case) ----
%   Figures (saved to output folder as PNG):
%     <vessel>_B_mean.png   — dual-Y axis: area (left) + ADC (right) mean curves
%     <vessel>_C_grid.png   — 2x2 grid:
%                              top-left:  area individual traces (color = proximal→distal)
%                              top-right: area mean ± std
%                              bot-left:  ADC individual traces
%                              bot-right: ADC mean ± std
%
%   Summary (.mat):
%     pi_summary_<scan_name>.mat  containing:
%       pi_results  — struct array with per-segment:
%                      .display_name, .seg_id_1, .seg_id_2
%                      .pi_area_1/2, .pi_adc_1/2   (pulsatility index)
%                      .area_max/min/mean_1/2       (area statistics)
%                      .adc_max/min/mean_1/2        (ADC statistics)
%       pi_meta     — case metadata (names, mode, timestamp)
%
%   Output folder location:
%     Single scan:  <case_dir>/single_<scan_name>/
%     Dual scan:    <case_dir>/compare_<scan1>_vs_<scan2>/
%
% ---- Dependencies ----
%   - load_json_config.m  (for JSON config support, in same folder)
%   - MATLAB R2016b+      (for jsondecode)
% ========================================================================

clc; clear; close all;

%% ====== User config ======
CASE_LIST = {
    '0202s1'
    '0413s2'
};

% JSON config folder (optional - if configs are not on MATLAB path)
JSON_CONFIG_DIR = '';   % e.g. '/v/ai/nobackup/cni/chang_data/configs'

SAVE_PNG  = true;

n_phases    = 25;
MAX_LINES   = 30;
ZERO_AS_NAN = true;

COLOR_AREA_1 = [0.92 0.35 0.25];
COLOR_AREA_2 = [0.55 0.08 0.05];
COLOR_ADC_1  = [0.30 0.65 0.95];
COLOR_ADC_2  = [0.05 0.25 0.60];

GRAD_AREA_1_LO = [0.99 0.78 0.70]; GRAD_AREA_1_HI = [0.85 0.20 0.15];
GRAD_AREA_2_LO = [0.80 0.40 0.35]; GRAD_AREA_2_HI = [0.40 0.05 0.03];
GRAD_ADC_1_LO  = [0.75 0.88 0.98]; GRAD_ADC_1_HI  = [0.20 0.55 0.90];
GRAD_ADC_2_LO  = [0.40 0.55 0.80]; GRAD_ADC_2_HI  = [0.02 0.15 0.45];

SHOW_STD_IN_BMEAN = false;
SHOW_STD_IN_GRID  = true;
ALPHA_FILL_GRID   = 0.18;
%% ======================


%% ====== Main loop ======
for iCase = 1:length(CASE_LIST)
    CASE_NAME = CASE_LIST{iCase};
    fprintf('\n################################################################\n');
    fprintf('##  [%d/%d]  Case: %s\n', iCase, length(CASE_LIST), CASE_NAME);
    fprintf('################################################################\n');

    %% --- Load config (JSON or .m) ---
    case_cfg = load_case_config(CASE_NAME, JSON_CONFIG_DIR);
    if isempty(case_cfg)
        warning('Config not found for %s — skipping.', CASE_NAME);
        continue;
    end

    if ~isfield(case_cfg.scans, 's1')
        warning('Config must contain scan s1 — skipping %s.', CASE_NAME);
        continue;
    end
    scan1_cfg = case_cfg.scans.s1;

    has_s2 = isfield(case_cfg.scans, 's2');
    if has_s2
        scan2_cfg = case_cfg.scans.s2;
        fprintf('Mode: DUAL scan (%s vs %s)\n', scan1_cfg.name, scan2_cfg.name);
    else
        scan2_cfg = [];
        fprintf('Mode: SINGLE scan (%s)\n', scan1_cfg.name);
    end

    %% --- Load mat files ---
    fprintf('\nLoading scan1: %s\n', scan1_cfg.mat_file_adc);
    if ~exist(scan1_cfg.mat_file_adc, 'file')
        warning('scan1 mat not found — skipping %s.', CASE_NAME);
        continue;
    end
    D1 = load(scan1_cfg.mat_file_adc);
    area_all_1 = D1.area_all; adc_all_1 = D1.adc_all; seg_ids_1 = D1.seg_ids;
    if ZERO_AS_NAN, area_all_1(area_all_1 == 0) = NaN; end

    if has_s2
        fprintf('Loading scan2: %s\n', scan2_cfg.mat_file_adc);
        if ~exist(scan2_cfg.mat_file_adc, 'file')
            warning('scan2 mat not found — skipping %s.', CASE_NAME);
            continue;
        end
        D2 = load(scan2_cfg.mat_file_adc);
        area_all_2 = D2.area_all; adc_all_2 = D2.adc_all; seg_ids_2 = D2.seg_ids;
        if ZERO_AS_NAN, area_all_2(area_all_2 == 0) = NaN; end
    end

    phase_axis = 1:n_phases;

    %% --- Output folder ---
    [scan1_results_dir, ~, ~] = fileparts(scan1_cfg.output_folder);
    [scan1_step2_parent, ~, ~] = fileparts(scan1_results_dir);
    [scan1_output_parent, ~, ~] = fileparts(scan1_step2_parent);
    [case_dir, ~, ~] = fileparts(scan1_output_parent);

    if has_s2
        out_folder = fullfile(case_dir, sprintf('compare_%s_vs_%s', scan1_cfg.name, scan2_cfg.name));
    else
        out_folder = fullfile(case_dir, sprintf('single_%s', scan1_cfg.name));
    end
    if SAVE_PNG && ~exist(out_folder, 'dir'), mkdir(out_folder); end
    fprintf('Output folder: %s\n', out_folder);

    %% --- Match segments ---
    seg_table_1 = scan1_cfg.segments_of_interest;
    if has_s2
        seg_table_2 = scan2_cfg.segments_of_interest;
        display_names_2 = seg_table_2(:, 2);
    end

    n_picked = size(seg_table_1, 1);
    n_done = 0;

    pi_results = struct('display_name', {}, ...
        'seg_id_1', {}, 'seg_id_2', {}, ...
        'n_pts_1', {}, 'n_pts_2', {}, ...
        'pi_area_1', {}, 'pi_area_2', {}, ...
        'pi_adc_1', {}, 'pi_adc_2', {}, ...
        'area_max_1', {}, 'area_min_1', {}, 'area_mean_1', {}, ...
        'area_max_2', {}, 'area_min_2', {}, 'area_mean_2', {}, ...
        'adc_max_1', {}, 'adc_min_1', {}, 'adc_mean_1', {}, ...
        'adc_max_2', {}, 'adc_min_2', {}, 'adc_mean_2', {});

    %% --- Edge loop (same as original, abbreviated) ---
    for k = 1:n_picked
        seg1_id      = seg_table_1{k, 1};
        display_name = seg_table_1{k, 2};
        trim1_start  = seg_table_1{k, 3};
        trim1_end    = seg_table_1{k, 4};

        if has_s2
            k2 = find(strcmp(display_names_2, display_name), 1);
            if isempty(k2)
                fprintf('SKIP [%d/%d] %s: not in scan2\n', k, n_picked, display_name);
                continue;
            end
            seg2_id = seg_table_2{k2, 1};
            trim2_start = seg_table_2{k2, 3};
            trim2_end = seg_table_2{k2, 4};
        end

        fprintf('\nEdge [%d/%d] %s  scan1: %s\n', k, n_picked, display_name, seg1_id);

        [area1, adc1, n1] = get_trimmed(area_all_1, adc_all_1, seg_ids_1, seg1_id, trim1_start, trim1_end);
        if isempty(area1), fprintf('  WARNING: no data, skip\n'); continue; end

        if has_s2
            [area2, adc2, n2] = get_trimmed(area_all_2, adc_all_2, seg_ids_2, seg2_id, trim2_start, trim2_end);
            if isempty(area2), fprintf('  WARNING: scan2 no data, skip\n'); continue; end
        end

        mu_a1 = mean(area1,1,'omitnan'); sd_a1 = std(area1,0,1,'omitnan');
        mu_d1 = mean(adc1,1,'omitnan');  sd_d1 = std(adc1,0,1,'omitnan');
        [pi_a1, area_max_1, area_min_1, area_mean_1] = pulsatility(mu_a1);
        [pi_d1, adc_max_1, adc_min_1, adc_mean_1] = pulsatility(mu_d1);

        if has_s2
            mu_a2 = mean(area2,1,'omitnan'); sd_a2 = std(area2,0,1,'omitnan');
            mu_d2 = mean(adc2,1,'omitnan');  sd_d2 = std(adc2,0,1,'omitnan');
            [pi_a2, area_max_2, area_min_2, area_mean_2] = pulsatility(mu_a2);
            [pi_d2, adc_max_2, adc_min_2, adc_mean_2] = pulsatility(mu_d2);
        end

        % Accumulate results
        pi_results(end+1).display_name = display_name;
        pi_results(end).seg_id_1 = seg1_id; pi_results(end).n_pts_1 = n1;
        pi_results(end).pi_area_1 = pi_a1; pi_results(end).pi_adc_1 = pi_d1;
        pi_results(end).area_max_1 = area_max_1; pi_results(end).area_min_1 = area_min_1;
        pi_results(end).area_mean_1 = area_mean_1;
        pi_results(end).adc_max_1 = adc_max_1; pi_results(end).adc_min_1 = adc_min_1;
        pi_results(end).adc_mean_1 = adc_mean_1;

        if has_s2
            pi_results(end).seg_id_2 = seg2_id; pi_results(end).n_pts_2 = n2;
            pi_results(end).pi_area_2 = pi_a2; pi_results(end).pi_adc_2 = pi_d2;
            pi_results(end).area_max_2 = area_max_2; pi_results(end).area_min_2 = area_min_2;
            pi_results(end).area_mean_2 = area_mean_2;
            pi_results(end).adc_max_2 = adc_max_2; pi_results(end).adc_min_2 = adc_min_2;
            pi_results(end).adc_mean_2 = adc_mean_2;
        else
            pi_results(end).seg_id_2 = ''; pi_results(end).n_pts_2 = 0;
            pi_results(end).pi_area_2 = NaN; pi_results(end).pi_adc_2 = NaN;
            pi_results(end).area_max_2 = NaN; pi_results(end).area_min_2 = NaN;
            pi_results(end).area_mean_2 = NaN;
            pi_results(end).adc_max_2 = NaN; pi_results(end).adc_min_2 = NaN;
            pi_results(end).adc_mean_2 = NaN;
        end

        safe_name = regexprep(display_name, '[\\/:*?"<>|]', '_');
        title_str = strrep(display_name, '_', '\_');

        % ==================== Figure 1: dual Y mean ====================
        fig1 = figure('Position', [100 100 720 500], 'Visible', 'off');
        ax1 = axes(fig1); hold(ax1, 'on');

        yyaxis(ax1, 'left');
        h_a1 = plot(ax1, phase_axis, mu_a1, '-o', 'Color', COLOR_AREA_1, 'LineWidth', 2.2, 'MarkerFaceColor', COLOR_AREA_1, 'MarkerSize', 4.5);
        if has_s2
            h_a2 = plot(ax1, phase_axis, mu_a2, '--o', 'Color', COLOR_AREA_2, 'LineWidth', 2.2, 'MarkerFaceColor', COLOR_AREA_2, 'MarkerSize', 4.5);
        end
        ylabel(ax1, 'Cross-sectional area (mm^2)'); ax1.YColor = COLOR_AREA_2;

        yyaxis(ax1, 'right');
        h_d1 = plot(ax1, phase_axis, mu_d1, '-s', 'Color', COLOR_ADC_1, 'LineWidth', 2.2, 'MarkerFaceColor', COLOR_ADC_1, 'MarkerSize', 4.5);
        if has_s2
            h_d2 = plot(ax1, phase_axis, mu_d2, '--s', 'Color', COLOR_ADC_2, 'LineWidth', 2.2, 'MarkerFaceColor', COLOR_ADC_2, 'MarkerSize', 4.5);
        end
        ylabel(ax1, 'Paravascular CSF ADC'); ax1.YColor = COLOR_ADC_2;
        xlabel(ax1, 'Phase'); xlim(ax1, [1 n_phases]); grid(ax1, 'on');

        if has_s2
            legend(ax1, [h_a1 h_a2 h_d1 h_d2], {['Area ' scan1_cfg.name], ['Area ' scan2_cfg.name], ['ADC ' scan1_cfg.name], ['ADC ' scan2_cfg.name]}, 'Location', 'best');
            title(ax1, sprintf('%s | %s vs %s', title_str, scan1_cfg.name, scan2_cfg.name));
        else
            legend(ax1, [h_a1 h_d1], {['Area ' scan1_cfg.name], ['ADC ' scan1_cfg.name]}, 'Location', 'best');
            title(ax1, sprintf('%s | %s', title_str, scan1_cfg.name));
        end

        if SAVE_PNG
            exportgraphics(fig1, fullfile(out_folder, [safe_name '_B_mean.png']), 'Resolution', 150);
        end
        close(fig1);

        % ==================== Figure 2: 2x2 grid ====================
        li1 = sample_idx(n1, MAX_LINES);
        fig2 = figure('Position', [100 100 1200 720], 'Visible', 'off');

        % C1: area traces
        ax_c1 = subplot(2,2,1); hold on;
        cmap_a1 = make_gradient(GRAD_AREA_1_LO, GRAD_AREA_1_HI, length(li1));
        for kk = 1:length(li1)
            plot(phase_axis, area1(li1(kk),:), '-', 'Color', [cmap_a1(kk,:) 0.7], 'LineWidth', 0.9);
        end
        if has_s2
            li2 = sample_idx(n2, MAX_LINES);
            cmap_a2 = make_gradient(GRAD_AREA_2_LO, GRAD_AREA_2_HI, length(li2));
            for kk = 1:length(li2), plot(phase_axis, area2(li2(kk),:), '--', 'Color', [cmap_a2(kk,:) 0.7], 'LineWidth', 0.9); end
        end
        ylabel('Area (mm^2)'); title('Area traces'); xlim([1 n_phases]); grid on;

        % C2: area mean±std
        ax_c2 = subplot(2,2,2); hold on;
        if SHOW_STD_IN_GRID, fill_patch(ax_c2, phase_axis, mu_a1, sd_a1, COLOR_AREA_1, ALPHA_FILL_GRID); end
        plot(phase_axis, mu_a1, '-o', 'Color', COLOR_AREA_1, 'LineWidth', 2, 'MarkerFaceColor', COLOR_AREA_1, 'MarkerSize', 4);
        if has_s2
            if SHOW_STD_IN_GRID, fill_patch(ax_c2, phase_axis, mu_a2, sd_a2, COLOR_AREA_2, ALPHA_FILL_GRID); end
            plot(phase_axis, mu_a2, '--o', 'Color', COLOR_AREA_2, 'LineWidth', 2, 'MarkerFaceColor', COLOR_AREA_2, 'MarkerSize', 4);
        end
        ylabel('Area (mm^2)'); title('Area mean +/- std'); xlim([1 n_phases]); grid on;

        % C3: ADC traces
        ax_c3 = subplot(2,2,3); hold on;
        cmap_d1 = make_gradient(GRAD_ADC_1_LO, GRAD_ADC_1_HI, length(li1));
        for kk = 1:length(li1), plot(phase_axis, adc1(li1(kk),:), '-', 'Color', [cmap_d1(kk,:) 0.7], 'LineWidth', 0.9); end
        if has_s2
            cmap_d2 = make_gradient(GRAD_ADC_2_LO, GRAD_ADC_2_HI, length(li2));
            for kk = 1:length(li2), plot(phase_axis, adc2(li2(kk),:), '--', 'Color', [cmap_d2(kk,:) 0.7], 'LineWidth', 0.9); end
        end
        xlabel('Phase'); ylabel('CSF ADC'); title('ADC traces'); xlim([1 n_phases]); grid on;

        % C4: ADC mean±std
        ax_c4 = subplot(2,2,4); hold on;
        if SHOW_STD_IN_GRID, fill_patch(ax_c4, phase_axis, mu_d1, sd_d1, COLOR_ADC_1, ALPHA_FILL_GRID); end
        plot(phase_axis, mu_d1, '-o', 'Color', COLOR_ADC_1, 'LineWidth', 2, 'MarkerFaceColor', COLOR_ADC_1, 'MarkerSize', 4);
        if has_s2
            if SHOW_STD_IN_GRID, fill_patch(ax_c4, phase_axis, mu_d2, sd_d2, COLOR_ADC_2, ALPHA_FILL_GRID); end
            plot(phase_axis, mu_d2, '--o', 'Color', COLOR_ADC_2, 'LineWidth', 2, 'MarkerFaceColor', COLOR_ADC_2, 'MarkerSize', 4);
        end
        xlabel('Phase'); ylabel('CSF ADC'); title('ADC mean +/- std'); xlim([1 n_phases]); grid on;

        if has_s2
            sgtitle(sprintf('%s | %s: %d pts | %s: %d pts', title_str, scan1_cfg.name, n1, scan2_cfg.name, n2));
        else
            sgtitle(sprintf('%s | %s: %d pts', title_str, scan1_cfg.name, n1));
        end

        if SAVE_PNG
            exportgraphics(fig2, fullfile(out_folder, [safe_name '_C_grid.png']), 'Resolution', 150);
        end
        close(fig2);

        n_done = n_done + 1;
    end

    %% --- PI summary ---
    fprintf('\nCase %s: %d/%d edges done.\n', CASE_NAME, n_done, n_picked);
    if ~isempty(pi_results)
        fprintf('%-10s | %10s %10s\n', 'Edge', 'PI_area', 'PI_adc');
        for r = 1:length(pi_results)
            rr = pi_results(r);
            fprintf('%-10s | %10.3f %10.3f\n', rr.display_name, rr.pi_area_1, rr.pi_adc_1);
        end

        pi_meta.case_name = CASE_NAME;
        pi_meta.scan1_name = scan1_cfg.name;
        pi_meta.mode = 'single';
        if has_s2, pi_meta.scan2_name = scan2_cfg.name; pi_meta.mode = 'dual'; end
        pi_mat_file = fullfile(out_folder, sprintf('pi_summary_%s.mat', scan1_cfg.name));
        save(pi_mat_file, 'pi_results', 'pi_meta');
        fprintf('Saved: %s\n', pi_mat_file);
    end
end

fprintf('\nAll %d cases done.\n', length(CASE_LIST));


%% ===================== Helper: load config (JSON or .m) =====================

function cfg = load_case_config(case_name, json_dir)
% Try JSON first, then .m
    cfg = [];

    % Try JSON
    json_name = ['case_config_' case_name '.json'];
    if ~isempty(json_dir)
        json_path = fullfile(json_dir, json_name);
    else
        json_path = json_name;  % look in current dir / MATLAB path
    end

    if exist(json_path, 'file')
        cfg = load_json_config(json_path);
        fprintf('Loaded JSON config: %s\n', json_path);
        return;
    end

    % Try .m
    m_func = ['case_config_' case_name];
    if exist(m_func, 'file') == 2
        cfg = feval(m_func);
        fprintf('Loaded .m config: %s\n', m_func);
        return;
    end

    warning('No config found for %s (tried .json and .m)', case_name);
end


%% ===================== Helper functions =====================

function [area_t, adc_t, n_pts] = get_trimmed(area_all, adc_all, seg_ids, seg_id, n_trim_start, n_trim_end)
    mask = strcmp(seg_ids, seg_id);
    if ~any(mask), area_t=[]; adc_t=[]; n_pts=0; return; end
    area_seg = area_all(mask,:); adc_seg = adc_all(mask,:);
    i_start = n_trim_start + 1; i_end = size(area_seg,1) - n_trim_end;
    if i_start > i_end, area_t=[]; adc_t=[]; n_pts=0; return; end
    area_t = area_seg(i_start:i_end,:); adc_t = adc_seg(i_start:i_end,:);
    n_pts = size(area_t,1);
end

function [pi_val, v_max, v_min, v_mean] = pulsatility(mu)
    v = mu(~isnan(mu));
    if isempty(v), pi_val=NaN; v_max=NaN; v_min=NaN; v_mean=NaN; return; end
    v_max=max(v); v_min=min(v); v_mean=mean(v);
    if v_mean<=0, pi_val=NaN; else, pi_val=(v_max-v_min)/v_mean; end
end

function idx = sample_idx(n, maxn)
    if n<=maxn, idx=1:n; else, idx=round(linspace(1,n,maxn)); end
end

function fill_patch(ax, x, mu, sg, col, alpha_val)
    xf=[x,fliplr(x)]; yf=[mu+sg,fliplr(mu-sg)];
    fill(ax, xf, yf, col, 'FaceAlpha', alpha_val, 'EdgeColor', 'none');
end

function cmap = make_gradient(c_lo, c_hi, n)
    if n<=1, cmap=c_hi(:)'; return; end
    t=linspace(0,1,n)'; cmap=(1-t)*c_lo + t*c_hi;
end