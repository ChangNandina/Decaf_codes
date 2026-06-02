function cfg = case_config_1211_reg1104()
% 1211_reg1104 case 的配置（两次扫描）
% 配套 plot_edge_area_adc.m / plot_edge_area_adc_compare_byconfig.m 使用
%
% 顺序约定: s1 = 第一次扫描 (1104), s2 = 第二次扫描 (1211)

case_dir = 'Z:\nobackup\cni\chang_data\1211_reg1104';

% ── 扫描 1 (第一次, 1104) ─────────────────────────
cfg.scans.s1.name          = '1104';
cfg.scans.s1.mat_file_adc  = fullfile(case_dir, 'output_1104', ...
    'results\step2_multilabel\paravascular_adc_v2.mat');
cfg.scans.s1.output_folder = fullfile(case_dir, 'output_1104', ...
    'results\step2_multilabel\layout_preview_trim');
% segment 表: { seg_id_in_mat, display_name, trim_start, trim_end }
cfg.scans.s1.segments_of_interest = {
    'EP_01—EP_02',  'M2_R',  5,  5;
    'EP_04—EP_06',  'M1_R',  3,  3;
    'EP_07—EP_14',  'A1_R',  3,  3;
    'EP_20—EP_28',  'A1_L',  3,  3;
    'EP_05—EP_11',  'P2_R',  3,  3;
    'EP_23—EP_31',  'P2_L',  9,  3;
    %'EP_13—EP_15',  'P1_R',  3,  3;
    %'EP_17—EP_23',  'P1_L',  9,  3;
    'EP_30—EP_32',  'M1_L',  3,  3;
    'EP_34—EP_36',  'M2_L',  3,  10;
};

% ── 扫描 2 (第二次, 1211) ──────────────────────────
% 注意: 同一条血管在两次扫描里的编号可能不一样, trim 也可能不一样
cfg.scans.s2.name          = '1211';
cfg.scans.s2.mat_file_adc  = fullfile(case_dir, 'output_1211', ...
    'results\step2_multilabel\paravascular_adc_v2.mat');
cfg.scans.s2.output_folder = fullfile(case_dir, 'output_1211', ...
    'results\step2_multilabel\layout_preview_trim');
cfg.scans.s2.segments_of_interest = {
    'EP_01—EP_02',  'M2_R',  5,  5;
    'EP_04—EP_06',  'M1_R',  5,  5;
    'EP_07—EP_16',  'A1_R',  3,  3;
    %'EP_21—EP_29',  'A1_L',  3,  3;seg fail
    'EP_05—EP_13',  'P2_R',  3,  3;
    'EP_25—EP_32',  'P2_L',  9,  5;
    %'EP_31—EP_33',  'M1_L',  3,  3;%no adc
    'EP_35—EP_37',  'M2_L',  3,  10;
};  

end