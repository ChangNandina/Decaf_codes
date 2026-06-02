function cfg = load_json_config(json_path)
% LOAD_JSON_CONFIG  Read a JSON config file from vessel_config_editor.html
% and return a struct compatible with case_config_*.m format.
%
% ---- Usage ----
%   cfg = load_json_config('case_config_250701.json');
%
% ---- JSON input format (saved by vessel_config_editor.html) ----
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
%       "s2": { ... }   % optional
%     }
%   }
%
% ---- Output struct (same as case_config_*.m) ----
%   cfg.scans.s1.name                     — scan display name (e.g. '0701')
%   cfg.scans.s1.mat_file_adc             — path to .mat with area/adc data
%   cfg.scans.s1.output_folder            — path for figure output
%   cfg.scans.s1.segments_of_interest     — cell array {seg_id, name, trim_start, trim_end; ...}
%
% ---- Notes ----
%   - Paths with forward slashes are auto-converted to backslashes on Windows
%   - Requires MATLAB R2016b+ (jsondecode)
%   - Called automatically by plot_batch_jsoncompat.m when .json config exists

    raw = jsondecode(fileread(json_path));
    cfg = struct();

    scan_names = fieldnames(raw.scans);
    for i = 1:length(scan_names)
        sn = scan_names{i};
        src = raw.scans.(sn);

        cfg.scans.(sn).name          = src.name;
        cfg.scans.(sn).mat_file_adc  = fix_path(src.mat_file);
        cfg.scans.(sn).output_folder = fix_path(src.output_folder);

        % Convert segments array to cell array: {seg_id, name, trim_start, trim_end; ...}
        segs = src.segments_of_interest;
        if isempty(segs)
            cfg.scans.(sn).segments_of_interest = {};
            continue;
        end

        n_seg = length(segs);
        seg_cell = cell(n_seg, 4);
        for j = 1:n_seg
            if isstruct(segs)
                s = segs(j);
            else
                s = segs{j};
            end
            seg_cell{j, 1} = s.seg_id;
            seg_cell{j, 2} = s.name;
            seg_cell{j, 3} = s.trim_start;
            seg_cell{j, 4} = s.trim_end;
        end
        cfg.scans.(sn).segments_of_interest = seg_cell;
    end

    fprintf('Loaded JSON config: %s\n', json_path);
    fprintf('  Scans: %s\n', strjoin(scan_names, ', '));
    for i = 1:length(scan_names)
        sn = scan_names{i};
        n = size(cfg.scans.(sn).segments_of_interest, 1);
        fprintf('  %s: %s, %d segments\n', sn, cfg.scans.(sn).name, n);
    end
end


function p = fix_path(p)
% Convert forward slashes to system separator if on Windows
    if ispc
        p = strrep(p, '/', '\');
    end
end