% ==========================================================
% CODE TO SPLIT ALL EEGMMI EEG FILES INSIDE A FOLDER AND 
% CREATE A SET OF MAT FILES WITH EEG TRIAL PLUS THEIR LABELS
% ==========================================================
%
% ------------------------- NOTE 1 -------------------------
% Remember to change the following path according to your 
% dataset location and desired output path.
% In the output path, a folder with the same name as the 
% one with all EEGs will be created. Insisde this folder,
% other folders will be created, the first with all left
% MM task and the other with the right MM task
% ------------------------- NOTE 2 -------------------------
% This code works if data inside the path2data folder were 
% preprocessed with the BIDSAlign library. They must be 
% set files with name 
% {dataset_id}_{subject_id}_{session_id}_{object_id}.set .
%
% For example, 17_1_1_1.set will refer to the first 
%              session of the first subject
%
% Inside the output folder, the same structure will be used
% but the object ID will become the trial number.-
%
% For example, 17_1_1_15.set, will refer to the first 
%              session of the first subject,
%              seconds from 56 to 60 of the original file
%
% To better exploit the name structure, original trial number
% will be put in session id (from 1 to 14) while the specific
% performed task will become a trial. 
% 
% Moreover, in a third folder with name "mat_files", all
% splits will be stored in a .mat files having a struct
% with fields data and label
% ----------------------------------------------------------

%  0: eyes open
%  1: eyes closed, 
%  2: move left hand, 
%  3: move right hand, 
%  4: image left hand,
%  5: image right hand, 
%  6: move both fists, 
%  7: move both feet, 
%  8: image both fist,
%  9: image both feet, 
% 10: trial rest.


% "BASE1T0": "baseline (eyes open)",
% "BASE2T0": "baseline (eyes closed)",
% "TASK1T0": "Task 1 rest ",
% "TASK1T1": "onset of the real motion of opening and closing the left fist when target appears on left side of screen",
% "TASK1T2": "onset of the real motion of opening and closing the right fist when target appears on right side of screen",
% "TASK2T0": "Task 2 rest",
% "TASK2T1": "onset of the imagined motion of opening and closing the left fist when target appears on left side of screen",
% "TASK2T2": "onset of the imagined motion of opening and closing the right fist when target appears on right side of screen",
% "TASK3T0": "Task 3 rest",
% "TASK3T1": "onset of the real motion of opening and closing both fists when target appears on the top side of screen",
% "TASK3T2": "onset of the real motion of opening and closing both feet when target appears on the bottom side of screen",
% "TASK4T0": "Task 4 rest",
% "TASK4T1": "onset of the imagined motion of opening and closing both fists when target appears on the top side of screen",
% "TASK4T2": "onset of the imagined motion of opening and closing both feet when target appears on the bottom side of screen"

clc
close all
clear

% set here the dataset code and preprocessing pipeline. They will be used to create a
% folder with name dataset_code + pipeline inside the outputpath
pipeline = '_ICASR';
dataset_code = 'ds004362';

% ------------------------------------------------------------
% in the following path DO NOT INCLUDE the last ( "/" or "\" )
% ------------------------------------------------------------

% path to the dataset preprocessed with BIDSAlign (files in .set with events)
path2data = '/home/pnc/Documents/_set_preprocessed';
root_path2data = [path2data filesep];
path2data = [root_path2data dataset_code pipeline];

% output path, used to create the output folders and save the splitted files
outputpath = '/home/pnc/Documents/_mat_splitted';
root_outputpath = [outputpath filesep];
outputpath = [root_outputpath dataset_code pipeline];

% path to the dataset preprocessed with BIDSAlign (files in .mat to actually split)
path2mat = '/home/pnc/Documents/_mat_preprocessed';
root_path2mat = [path2mat filesep];
path2mat = [path2mat filesep dataset_code pipeline];


% START CODING PART. FROM HERE EVERYTHING SHOULD WORK WITHOUT ADDITIONS OR MODIFICATIONS 

% get all file names 
file_names = get_dataset_file_list( root_path2data, [dataset_code pipeline], '.set' );

% create output folders if they not exist
if ~isfolder( [root_outputpath dataset_code 'EO' pipeline])
    mkdir([root_outputpath dataset_code 'EO' pipeline])
end
if ~isfolder( [root_outputpath dataset_code 'EC' pipeline])
    mkdir([root_outputpath dataset_code 'EC' pipeline])
end
if ~isfolder( [root_outputpath dataset_code 'LH' pipeline])
    mkdir([root_outputpath dataset_code 'LH' pipeline])
end
if ~isfolder( [root_outputpath dataset_code 'RH' pipeline])
    mkdir([root_outputpath dataset_code 'RH' pipeline])
end
if ~isfolder( [root_outputpath dataset_code 'LHI' pipeline])
    mkdir([root_outputpath dataset_code 'LHI' pipeline])
end
if ~isfolder( [root_outputpath dataset_code 'RHI' pipeline])
    mkdir([root_outputpath dataset_code 'RHI' pipeline])
end
if ~isfolder( [root_outputpath dataset_code 'BH' pipeline])
    mkdir([root_outputpath dataset_code 'BH' pipeline])
end
if ~isfolder( [root_outputpath dataset_code 'BF' pipeline])
    mkdir([root_outputpath dataset_code 'BF' pipeline])
end
if ~isfolder( [root_outputpath dataset_code 'BHI' pipeline])
    mkdir([root_outputpath dataset_code 'BHI' pipeline])
end
if ~isfolder( [root_outputpath dataset_code 'BFI' pipeline])
    mkdir([root_outputpath dataset_code 'BFI' pipeline])
end
if ~isfolder( [root_outputpath dataset_code 'RE' pipeline])
    mkdir([root_outputpath dataset_code 'RE' pipeline])
end

% iterate over all files
for i = 1 : size(file_names,1)

    % extract subject and session
    raw_filename = file_names(i).name;
    raw_filepath = file_names(i).folder;
    raw_filename_split = strsplit(raw_filename, '_');
    subject = str2double(raw_filename_split{2});
    session = raw_filename_split{end};
    session = str2double(session(1:end-4));
    [~, EEG_full] = evalc( "pop_loadset('filename',raw_filename, 'filepath',raw_filepath);");
    EEG_MAT = load([path2mat filesep raw_filename(1:end-4) '.mat']);

    srate = EEG_full.srate;
    if subject == 88 || subject == 92 || subject == 100
        disp(['-- SKIPPED -- FILE ' num2str(i) ': Subject ' num2str(subject) ...
        ', Session ' num2str(session) ', Sampling Rate ' num2str(srate)])
        continue
    end
    disp(['SPLITTING FILE ' num2str(i) ': Subject ' num2str(subject) ...
        ', Session ' num2str(session) ', Sampling Rate ' num2str(srate)])

    if session == 1
        % full record of rest data with eyes open don't need to be splitted
        new_filename = [strjoin(raw_filename_split(1:2), '_') '_' num2str(session) '_1.mat'];
        EEG_MAT.DATA_STRUCT.label_group = 0;
        EEG_MAT.DATA_STRUCT.label_map = 'EO';
        save( [root_outputpath dataset_code 'EO' pipeline filesep new_filename], '-struct', 'EEG_MAT' );

    elseif session == 2
        % full record of rest data with eyes closed don't need to be splitted
        new_filename = [strjoin(raw_filename_split(1:2), '_') '_' num2str(session) '_1.mat'];
        EEG_MAT.DATA_STRUCT.label_group = 1;
        EEG_MAT.DATA_STRUCT.label_map = 'EC';
        save( [root_outputpath dataset_code 'EC' pipeline filesep new_filename], '-struct', 'EEG_MAT' );

    else
        % task records need to be splitted
        % usually alternates between rest and task data
        % each task is long exactly 4.1 seconds (656 samples)

        EEG_MAT_trial = EEG_MAT;
        for k = 1:(size(EEG_full.event,2))
            
            % get trial start and ending index
            if k==size(EEG_full.event,2)
                latencies = [floor(EEG_full.event(k).latency) size(EEG_full.data,2)];
            else
                latencies = [floor(EEG_full.event(k).latency) floor(EEG_full.event(k+1).latency)-1];
            end
            if (latencies(2)-latencies(1))< 655
                latencies = [latencies(1) min(latencies(2), size(EEG_full.data,2))];
            end
           
    
            % get label
            label = EEG_full.event(k).type;
            targetFolder = '';
            switch label
                case 'TASK1T0'
                    label = 10;
                    targetFolder = 'RE';
                case 'TASK1T1'
                    label = 2;
                    targetFolder = 'LH';
                case 'TASK1T2'
                    label = 3;
                    targetFolder = 'RH';
                case 'TASK2T0'
                    label = 10;
                    targetFolder = 'RE';
                case 'TASK2T1'
                    label = 4;
                    targetFolder = 'LHI';
                case 'TASK2T2'
                    label = 5;
                    targetFolder = 'RHI';
                case 'TASK3T0'
                    label = 10;
                    targetFolder = 'RE';
                case 'TASK3T1'
                    label = 6;
                    targetFolder = 'BH';
                case 'TASK3T2'
                    label = 7;
                    targetFolder = 'BF';
                case 'TASK4T0'
                    label = 10;
                    targetFolder = 'RE';
                case 'TASK4T1'
                    label = 8;
                    targetFolder = 'BHI';
                case 'TASK4T2'
                    label = 9;
                    targetFolder = 'BFI';
            end

            if (latencies(2)-latencies(1))< 655 && label ~=10
                disp('Subject has a too short final trial')
            end
            
            % set trial info
            EEG_MAT_trial.DATA_STRUCT.label_group = label;
            EEG_MAT_trial.DATA_STRUCT.label_map = targetFolder;
            EEG_MAT_trial.DATA_STRUCT.data = EEG_MAT.DATA_STRUCT.data(:, latencies(1):latencies(2));

            % create new_filename
            new_filename = [strjoin(raw_filename_split(1:2), '_') '_' num2str(session) '_' num2str(k) '.mat'];

            save( [root_outputpath dataset_code targetFolder ...
                pipeline filesep new_filename], '-struct', 'EEG_MAT_trial' );

        end
    end

end
