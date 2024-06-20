
function [EEG, DATA_STRUCT] = preprocess_dataset(dataset_info, save_info, ...
    params_info, path_info, selection_info, verbose)
    % FUNCTION: preprocess_dataset
    %
    % Description: Preprocesses a dataset of EEG recordings, extracts relevant
    % information, and saves the preprocessed data to a template-based data
    % structure or matrix format.
    %
    % Syntax:
    %   [EEG, DATA_STRUCT] = preprocess_dataset(dataset_info, save_info, params_info,
    %                                           path_info, selection_info, verbose)
    %
    % Input:
    %   - dataset_info: Structure containing dataset-specific information.
    %   - save_info: Structure specifying the data saving options.
    %   - params_info: Structure containing preprocessing parameters.
    %   - path_info: Structure containing necessary paths.
    %   - selection_info: Structure containing selection parameters.
    %   - verbose: Boolean setting the verbosity level. False will suppress
    %             warnings and other disp print.
    %
    % Output:
    %   - EEG: EEG data structure after preprocessing.
    %   - DATA_STRUCT: Structure containing preprocessed data information.
    %
    % Notes:
    %   - This function preprocesses all the EEG data for a specified dataset,
    %     including loading participant and diagnostic files, extracting channel
    %     information, interpolating missing channels, and saving the data.
    %
    % Author: [Andrea Zanola, Federico Del Pup]
    % Date: [25/01/2024]
    
    %% SET VERBOSE
    if nargin < 6
        verbose =  false;
    end
    
    
    %% Load Dataset Informations
    [data_info, path_info, template_info, T] = load_info(dataset_info, path_info, ...
        params_info, save_info, verbose);

    %% Extract selected patients
    cd(path_info.dataset_path);

    % Check if subject selection is required
    % if selection_info.select_subjects && isempty(T)
    %     error('UNABLE TO SELECT SUBJECTS WITH NO PARTICIPANT FILE LOADED');
    % 
    % else

    if xor(isempty(selection_info.label_name),isempty(selection_info.label_value))
        error("ONE BETWEEN LABEL NAME AND LABEL VALUE IS EMPTY.");
    end

    if selection_info.select_subjects && ~isempty(selection_info.label_name) && ~isempty(selection_info.label_value)
        if ~isempty(T)
            idx_column = linspace(1,width(T),width(T));
            I = idx_column(ismember(T.Properties.VariableNames, selection_info.label_name));
            subj_list = T.(1); % first column always ID
            all_subj_list = subj_list; % to avoid errors if T is empty (no participant file)
            
            mask = strcmp(T.(I), selection_info.label_value);                                     
            subj_list = subj_list(mask,:);
            Tr = T(mask,:);
        else
            error('UNABLE TO SELECT SUBJECTS BY GROUP WITH NO PARTICIPANT FILE LOADED');
        end
        
    else
        d = dir(pwd);
        dfolders = d([d(:).isdir]);

        %exclude some folders
        untouchable_folders = {'.','..','code','stimuli','derivatives','sourcedata','.datalad'};
        dfolders = dfolders(~ismember({dfolders(:).name},[untouchable_folders path_info.diagnostic_folder_name]));
        subj_list = {dfolders.name};
        %if isempty(T)
        all_subj_list = subj_list; % to avoid errors if T is empty (no participant file)
        %else
        %    all_subj_list = T.participant_id;
        %end
        Tr = T;
    end

    % Select files on the basis of number of subjects
    if selection_info.select_subjects
        [vec_subj, subj_list] = get_elements(subj_list, selection_info.sub_i, ...
            selection_info.sub_f, selection_info.subjects_totake,'SUBJECTS', verbose);
    else
        vec_subj = 1:1:length(subj_list);
    end
    
    %% Check Channel/Electrodes file name 
    obj_info.check_ch0_root = dir([path_info.dataset_path '*_channels.tsv']);
    obj_info.check_el0_root = dir([path_info.dataset_path '*_electrodes.tsv']);
    if length(obj_info.check_ch0_root)>1
        error(['MULTIPLE CHANNEL FILES PRESENT IN ' path_info.dataset_path]);
    end
    if length(obj_info.check_el0_root)>1
        error(['MULTIPLE ELECTRODES FILES PRESENT IN ' path_info.dataset_path]);
    end

    L = [];
    counts = [0,0,0];

    %% Preprocess all subjects
    for j=vec_subj
        counts(1)=counts(1)+1;
        if verbose
            fprintf([' \t\t\t\t\t\t\t\t --- ' data_info.dataset_name ...
                ' - PROCESSING SUBJECT: ' num2str(counts(1)) '/' ...
                num2str(length(vec_subj)) ' ---\n']);
        end

        subject_name   = subj_list{j};
        jj = find(strcmp(all_subj_list, subject_name));
        subject_folder = [path_info.dataset_path subject_name];
        cd(subject_folder);

        %% Load subject information
        if ~isempty(Tr)
            O = find(strcmp(subject_name,Tr{:,1}));
            if ~isempty(O)
                subj_info = table2struct(Tr(O,:));
            else
                if verbose
                    warning('SUBJECT FOLDER NAME NOT FOUND IN THE PARTICIPANT FILE.');
                end
                subj_info = [];
            end
        else
            if verbose
                warning('PARTICIPANT FILE NOT PROVIDED, SUBJECT INFO NOT LOADED.');
            end
            subj_info = [];
        end

        %% Check Channel/Electrodes file name 
        obj_info.check_ch1_root = dir('*_channels.tsv');
        obj_info.check_el1_root = dir('*_electrodes.tsv');
        if length(obj_info.check_ch1_root)>1
            error(['MULTIPLE CHANNEL FILES PRESENT IN ' subject_folder]);
        end
        if length(obj_info.check_el1_root)>1
            error(['MULTIPLE ELECTRODES FILES PRESENT IN ' subject_folder]);
        end
    
        %% Extract selected sessions
        d         = dir(pwd);
        dfolders  = d([d(:).isdir]);
        sess_list = dfolders(~ismember({dfolders(:).name},{'.','..'}));
        all_sess_list = {sess_list(1:end).name};

        if selection_info.select_subjects
            % Select files on the basis of number of sessions
            [vec_sess, sess_list] = get_elements(sess_list, selection_info.ses_i, ...
                selection_info.ses_f, selection_info.session_totake, ...
                'SESSIONS', verbose);

        else
            vec_sess = 1:1:length(sess_list);
        end

        %% Preprocess all sessions for a subject
        counts(2) = 0;
        for k = vec_sess
            counts(2)=counts(2)+1;
            if verbose
                fprintf([' \t\t\t\t\t\t\t\t --- ' data_info.dataset_name ...
                    '- SESSION PROCESSED: ' num2str(counts(2)) ...
                    '/' num2str(length(vec_sess)) ' ---\n']);
            end
            session_name           = sess_list(k).name;
            kk = find(strcmp(all_sess_list, session_name));

            subject_session_folder = [subject_folder filesep ...
                session_name filesep 'eeg' filesep];
            cd(subject_session_folder);
    
            %% Check Channel/Electrodes file name 
            obj_info.check_ch2_root = dir('*_channels.tsv');
            obj_info.check_el2_root = dir('*_electrodes.tsv');

            %% Extract selected objects
            obj_info.raw_filepath  = pwd;
            obj_list     = dir(['*' data_info.eeg_file_extension]);
            obj_list = obj_list([6 7 8 9 10 11 12 13 14 1 2 3 4 5]);
            all_obj_list = {obj_list(1:end).name};

            if selection_info.select_subjects
                % Select files on the basis of number of sessions
                [vec_obj, obj_list] = get_elements(obj_list, selection_info.obj_i, ...
                    selection_info.obj_f, selection_info.task_totake, ...
                    'OBJECTS', verbose);
            else
                vec_obj = 1:1:length(obj_list);
            end

            %% Preprocess all files of a session -----------------------
            counts(3) = 0;
            for i=vec_obj
                counts(3)=counts(3)+1;
                if verbose
                    fprintf([' \t\t\t\t\t\t\t\t --- ' data_info.dataset_name ...
                        '- OBJECT PROCESSED: ' num2str(counts(3)) ...
                        '/' num2str(length(vec_obj)) ' ---\n']);
                end
                obj_info.raw_filename              = obj_list(i).name;
                ii = find(strcmp(all_obj_list, obj_list(i).name));

                obj_info.preprocessed_filename     =  ...
                    [int2str(data_info.dataset_number_reference) '_' ...
                    int2str(jj) '_' int2str(kk) '_' int2str(ii)];
                
                obj_info.set_preprocessed_filename = ...
                    [obj_info.preprocessed_filename data_info.eeg_file_extension];
                
                if verbose
                    fprintf([' \t\t\t\t\t\t\t\t --- PREPROCESSING: ' ...
                        subject_name '/' session_name '/' obj_list(i).name ...
                        ' ' num2str(i) '/' num2str(length(vec_obj)) ' ---\n']);
                end
                % Extract right electrodes, channels and event filenames
                [obj_info] = extract_filenames(obj_info, path_info, data_info, template_info);

              %% Run preprocess
                [EEG, L, obj_info] = preprocess_single_file(L, obj_info, data_info, ...
                    params_info, path_info, template_info, save_info, verbose);
                
              %% Save data to template (and interpolation)
                if save_info.save_data && ~isempty(EEG)
                    obj_info.mat_preprocessed_filename = ...
                    [path_info.mat_folder obj_info.preprocessed_filename '.mat'];
                    [EEG, DATA_STRUCT] = save_data_totemplate(EEG, obj_info, ...
                        template_info, save_info, path_info, data_info, ...
                        params_info, subj_info, verbose);
                    
                    % Export events
                    if ~isempty(EEG.event) && save_info.save_marker
                        eeg_eventtable(EEG,'exportFile',[path_info.output_csv_path ...
                            obj_info.preprocessed_filename '.csv'],'dispTable',false);
                    end
                else
                    DATA_STRUCT = [];
                end
                
            end
        end
    end

end

