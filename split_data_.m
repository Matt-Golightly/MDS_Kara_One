addpath(genpath('...\KARA_ONE_Data'));

folders = {'P02', 'MM05', 'MM08', 'MM09', 'MM10', 'MM11', 'MM12', 'MM14'};
folders = [folders {'MM15', 'MM16', 'MM18', 'MM19', 'MM20', 'MM21'}];

data_path = 'G:\UWA_MDS\2021SEM1\Research_Project\KARA_ONE_Data';
output_data_path = [data_path '\' 'ImaginedSpeechData'];
ICA = false;

for j=1:length(folders)
    
    disp(['Computing EEG Data for folder ' folders{j}]);
    folder = [data_path '/' folders{j}];
    D = dir([folder '/*.set']);
    set_fn = [folder '/' D(1).name];
    
    kinect_folder = [folder '/kinect_data']; %Required for extracting prompts
    labels_fn = [kinect_folder '/labels.txt'];
    
    EEG = pop_loadset(set_fn);
    data = EEG.data;
    
    disp('Splitting the data');
    load([folder '/epoch_inds.mat']);
    
    thinking_mats = split_data(thinking_inds, data); %EEG active data segemnted by trial
    epoch_data.thinking_mats = thinking_mats;
    
    rest_mats = split_data(clearing_inds, data); %EEG rest data segemnted by trial
    epoch_data.rest_mats = rest_mats;
    
     % Getting the prompts.
    prompts = table2cell(readtable(labels_fn,'Delimiter',' ','ReadVariableNames',false));
     
    
    % Getting the channel locations
    channel_loc = extractfield(EEG.chanlocs, 'labels');
    
    
    % Creating and saving the struct.
    disp('saving the struct');
     %%
    EEG_Data = struct();
   
    EEG_Data.prompts = prompts;
    EEG_Data.activeEEG = thinking_mats;
    EEG_Data.restEEG = rest_mats;
    EEG_Data.Data = data;
    EEG_Data.chanlocs = channel_loc;
    
    
    
    %%
    output_folder = [output_data_path '/' folders{j}];
    if ~exist(output_folder, 'dir')
       mkdir(output_folder)
    end
    save([output_folder '/EEG_Data.mat'], 'EEG_Data');

end 

