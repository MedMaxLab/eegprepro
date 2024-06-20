
function [EEG,L, obj_info] = preprocess_single_file(L, obj_info, data_info, params_info, ...
    path_info, template_info, save_info, verbose)
    % FUNCTION: preprocess_single_file
    %
    % Description: Preprocesses a single EEG data file based on specified parameters.
    %
    % Syntax:
    %   [EEG, L] = preprocess_single_file(L, obj_info, data_info, params_info, 
    %                                     path_info, template_info, save_info, verbose)
    %
    % Input:
    %   - L: Structure containing information about channel locations.
    %   - obj_info: Structure containing information about the EEG data file.
    %   - data_info: Structure containing general information about the dataset.
    %   - params_info: Structure containing preprocessing parameters.
    %   - path_info: Structure specifying paths for different components.
    %   - template_info: Structure containing template information.
    %   - save_info: Structure specifying the data saving options.
    %   - verbose: Boolean setting the verbosity level.
    %
    % Output:
    %   - EEG: EEG data structure after preprocessing.
    %   - L: Updated structure containing information about channel locations.
    %   - obj_info: Updated structure containing information about the EEG data file.
    %
    % Notes:
    %   - This function performs various preprocessing steps on a single EEG data file,
    %     including loading, importing events, managing channel locations, renaming 
    %     channels, removing NaN values, removing channels, removing segments,
    %     removing baseline, resampling, filtering, importing or generating channel
    %     locations, rereferencing, performing Independent Component Analysis (ICA),
    %     and applying Artifact Subspace Reconstruction (ASR).
    %
    % Author: [Andrea Zanola, Federico Del Pup]
    % Date: [25/01/2024]

    if nargin <8
        verbose = false;
    end
    
    %% Load raw file
    if verbose
        [EEG] = import_data(obj_info.raw_filename, obj_info.raw_filepath, verbose);
    else
        cmd2run = "import_data(obj_info.raw_filename, obj_info.raw_filepath, verbose);";
        [ ~, EEG] = evalc(cmd2run);
    end

    %% Overwrite Reference
    [EEG, obj_info] = get_reference(EEG, data_info, obj_info, verbose);

    %% Trye to Import Event
    if isempty(EEG.event) && ~isempty(obj_info.event_filename)
        try
            if verbose
                [EEG] = pop_importevent(EEG,'event', obj_info.event_filename, ...
                    'timeunit',1/EEG.srate);
            else
                cmd2run = "pop_importevent(EEG,'event', " + ...
                    "obj_info.event_filename,'timeunit',1/EEG.srate);";
                [~, EEG] = evalc(cmd2run);
            end
        catch
            if verbose
                warning('EVENT FILE PRESENT BUT NOT IMPORTED DUE TO ERROR.');
            end
        end
    elseif isempty(obj_info.event_filename)
        if verbose
            disp('NO EVENTFILE PRESENT.');
        end
    end

    %% Start Preprocessing Steps if EEG is loaded
    if ~isempty(EEG)
        EEG.history = ['--- PREVIOUS HISTORY ---' EEG.history newline ...
            '--- PREPROCESSED WITH BIDS-ALIGN ---'];
        EEG.history = [EEG.history newline 'DATASET: ' data_info.dataset_name];
        EEG.history = [EEG.history newline 'IMPORT DATA: ' obj_info.raw_filepath ...
            filesep obj_info.raw_filename];
    
        %% CHANLOCS MANAGMENT
        %(1) CHANLOCS MANAGMENT
        if  isempty(EEG.chanlocs) && isempty(obj_info.electrodes_filename) ...
                && isempty(obj_info.channels_filename)
            error('IMPOSSIBLE TO LOAD CHANNEL LOCATION');
        end
    
        %% Rename the labels of the channels according to the channels filename
        if ~isempty(obj_info.channels_filename) && ...
                ~isequal(obj_info.channel_location_filename, 'loaded')
            T = readtable(obj_info.channels_filename,'FileType','text');
            for i=1:EEG.nbchan
                EEG.chanlocs(i).labels = char(T{i,1});
            end
            EEG.history = [EEG.history newline ...
                'RENAME CHANNELS NAMES ACCORDING TO: ' obj_info.channels_filename];
        end

        %% Manually remove NaN 
        M = any(isnan(EEG.data));
        if any(M)
            EEG.data = EEG.data(:,not(M));
            EEG.history = [EEG.history newline 'REMOVED DATA WITH NANS: ' ...
                num2str(sum(M)) ' time stpes.'];
            if verbose
                warning('NANS ARE PRESENT IN THE RECORDED FILE');
            end
        end
    
        %% Remove Channels
        [EEG] = prepstep_removechannels(EEG, data_info, params_info, verbose);

        %% Import or generate channel location  
        %It is important to remove the channels first, because the
        %following function eventually uses create_chan_loc.m; here
        %conversion files are used, but non-physiological channels are not
        %present.
        [EEG, L, channel_location_file_extension, B] = load_channel_location(EEG, ...
            data_info, obj_info, L, template_info, verbose);

        %% Remove initial and final part of the recording
        if params_info.prep_steps.rmsegments
            params_info2 = params_info;
            idx1 = find(sum(EEG.data==0,1)==size(EEG.data,1));
            idx2 = idx1(idx1>(size(EEG.data,2)-320));
            if ~isempty(idx2)
                idx2 = idx2(1);
                if length(EEG.event)>1
                    maxlat = min(ceil(EEG.event(end).latency)+656, size(EEG.data,2) );
                    if size(EEG.data,2) < ceil(EEG.event(end).latency)+656
                        disp('final trial trial too short')
                    end
                    if idx2 < maxlat
                        idx2 = maxlat;
                    end
                end
                params_info2.dt_i = 0;
                params_info2.dt_f = (EEG.times(end) - EEG.times(idx2))/1000;
                disp(['removed from sample ' num2str(idx2)])
                [EEG] = prepstep_removesegments(EEG, params_info2, verbose);
                EEG.event = EEG.event(~strcmp({EEG.event.type},'boundary'));
            end
        end

        [EEG] = prepstep_removesegments(EEG, params_info, verbose);
    
        %% Remove baseline 
        [EEG] = prepstep_removebaseline(EEG, params_info, verbose);

        %% Resampling 
        [EEG] = prepstep_resampling(EEG, data_info, params_info, obj_info, verbose);

        %% Filtering
        [EEG] = prepstep_filtering(EEG, params_info, verbose);

        %% ICA and ICAREJ
        % ICA DECOMP
        [EEG] = prepstep_ICA(EEG, params_info, true, verbose);

        % ICA REJ
        if isfield(EEG,'reject')
            if isfield(EEG.reject,'gcompreject')
                EEG.reject.gcompreject = [];
            end
        end
        [EEG] = prepstep_ICArejection(EEG, params_info, verbose);

        %% 1˚ ASR channel correction
        nchan_preASR = EEG.nbchan;
        [EEG] = prepstep_1ASR(EEG, params_info, verbose);

        %% 2˚ ASR windows removal
        [EEG] = prepstep_2ASR(EEG, params_info, verbose);

        %% Interpolation
        [EEG] = prepstep_interpolation(EEG, params_info, B, nchan_preASR, verbose);
        
        %% Rereferecing
        [EEG] = rereference(EEG, data_info, params_info, obj_info, ...
            channel_location_file_extension, B, verbose);

        %% Warning if file after aggressive ASR, still have values over threshold
        if max(abs(EEG.data),[],'all') > params_info.th_reject 
            if verbose
                warning(['EEG FILE STILL GREATER THAN ' ...
                    num2str(params_info.th_reject) ' uV']);
            end
        end

        %% FINAL ICA DECOMPOSITION
        [EEG] = prepstep_ICA(EEG, params_info, false, verbose);

        %% Save the .set file 
        if save_info.save_set
            EEG.history = [EEG.history newline 'SAVE .SET FILE: ' ...
                path_info.set_folder obj_info.set_preprocessed_filename]; 
            
            if verbose
                pop_saveset( EEG, 'filename', obj_info.set_preprocessed_filename, ...
                    'filepath', path_info.set_folder);
            else
                cmd2run = "pop_saveset( EEG, 'filename', " + ...
                          "obj_info.set_preprocessed_filename, "+ ...
                          " 'filepath', path_info.set_folder);";
                evalc(cmd2run);
            end       
        end
    end
end
