load('all_EEGprocuesan.mat');


for i = 1: size(all_EEGprocuesan,2)
    for ii = 1:size(all_EEGprocuesan,3)
        final(:,ii,i) = all_EEGprocuesan(:,i,ii);
    end
end

cd('\\130.60.169.45\methlab\Neurometric\Antisaccades\code\eeglab14_1_2b')
eeglab;
close all

X.srate = 500
X.nbchan = 129
X.pnts = 500
X.trials = 1
X.xmin = 0
X.event = []
X.setname = []


for i = 1:size(final,1)
    X.data = final(i,:,:);
   
    downsamplEEG(i) = pop_resample(X,125);
end


 save('downsamplEEG', 'downsamplEEG', '-v7.3')

