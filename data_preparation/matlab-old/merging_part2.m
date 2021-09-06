% EEGprocuesan = 129X500X73
% fixEEGprocuesan = 73 X 500X129


clc
clear

x = dir('\\130.60.169.45\methlab\ETH_AS\preprocessed2')
subjects = {x.name};
subjects = {subjects{4:end-3}}';
clear x
cd('\\130.60.169.45\methlab\ETH_AS')


%%
all_trialinfoprosan = []

for subj = 1:100 %= 5 %186 - BA5 didnt work, 346- BY2
  
    
datapath = strcat('\\130.60.169.45\methlab\ETH_AS\preprocessed2\',subjects{subj});

 cd (datapath)
    
       if exist(strcat('trialinfoprosan.mat')) > 0
            datafile= strcat('trialinfoprosan.mat');
            load (datafile);
       end 
       
B = trialinfoprosan.cues;
A = all_trialinfoprosan;
all_trialinfoprosan = vertcat(A,B);
size(all_trialinfoprosan,1)
end

save('all_trialinfoprosan', 'all_trialinfoprosan', '-v7.3')