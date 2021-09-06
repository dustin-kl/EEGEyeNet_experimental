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
all_EEGprocuesan = []

for subj = 1:100%186 - BA5 didnt work, 346- BY2
  
    
datapath = strcat('\\130.60.169.45\methlab\ETH_AS\preprocessed2\',subjects{subj});

 cd (datapath)
    
       if exist(strcat('EEGprocuesan.mat')) > 0
            datafile= strcat('EEGprocuesan.mat');
            load (datafile);
       end 
       
final_EEGprocuesan = [];
for i = 1: size(EEGprocuesan.data,1)
    for ii = 1:size(EEGprocuesan.data,3)
        final_EEGprocuesan(ii,:,i) = EEGprocuesan.data(i,:,ii);
    end
end
all_EEGprocuesan = vertcat(all_EEGprocuesan ,final_EEGprocuesan);
size(all_EEGprocuesan,1);
end

save('all_EEGprocuesan', 'all_EEGprocuesan', '-v7.3')