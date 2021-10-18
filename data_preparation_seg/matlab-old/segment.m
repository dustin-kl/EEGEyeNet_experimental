clc
clear


cd('\\130.60.169.45\methlab\Neurometric\Antisaccades\code\eeglab14_1_2b')

eeglab;
close all

x = dir('\\130.60.169.45\methlab\ETH_AS\preprocessed2')
subjects = {x.name};
subjects = {subjects{4:end-3}}';
clear x
cd('\\130.60.169.45\methlab\ETH_AS')

   %%
for subj = 1:length(subjects) %186 - BA5 didnt work, 346- BY2
  
    
     datapath = strcat('\\130.60.169.45\methlab\ETH_AS\preprocessed2\',subjects{subj});

    cd (datapath)
    
       if exist(strcat('gip_',subjects{subj},'_AS_EEG.mat')) > 0
            datafile= strcat('gip_',subjects{subj},'_AS_EEG.mat');
            load (datafile)
        elseif exist(strcat('oip_',subjects{subj},'_AS_EEG.mat')) > 0
            datafile= strcat('oip_',subjects{subj},'_AS_EEG.mat');
            load (datafile)
       end 
       
       
       

%% Re-reference to average reference
       EEG = pop_reref(EEG,[]);
 
       %% triggers renaming
       countblocks = 1;
       for e = 1:length(EEG.event)
           if strcmp(EEG.event(e).type, 'boundary')
               countblocks = countblocks + 1;
               continue;
           end
           if countblocks == 2 || countblocks == 3 || countblocks == 4 % antisaccade blocks
               if strcmp(EEG.event(e).type,'10  ') % change 10 to 12 for AS
                   EEG.event(e).type = '12  ';
               elseif strcmp(EEG.event(e).type,'11  ')
                   EEG.event(e).type = '13  '; % change 11 to 13 for AS 
               end
               
               if strcmp(EEG.event(e).type,'40  ')
                   EEG.event(e).type = '41  ';
               end
               
           end
       end
       
       EEG.event(strcmp('boundary',{EEG.event.type})) = [];
       rmEventsIx = strcmp('L_fixation',{EEG.event.type});
       rmEv =  EEG.event(rmEventsIx);
       EEG.event(rmEventsIx) = [];
       EEG.event(1).dir = []; %left or right
       EEG.event(1).cond = [];%pro or anti
       %% rename EEG.event.type
       previous = '';
       for e = 1:length(EEG.event)
           if strcmp(EEG.event(e).type, 'L_saccade')
               if strcmp(previous, '10  ')
                   EEG.event(e).type = 'saccade_pro_left'
                   EEG.event(e).cond = 'pro';
                   EEG.event(e).dir = 'left';
                   %pro left
               elseif strcmp(previous, '11  ')
                   EEG.event(e).type = 'saccade_pro_right'
                   EEG.event(e).cond = 'pro';
                   EEG.event(e).dir = 'right';
               elseif strcmp(previous, '12  ')
                   EEG.event(e).type = 'saccade_anti_left'
                   EEG.event(e).cond = 'anti';
                   EEG.event(e).dir = 'left';
               elseif strcmp(previous, '13  ')
                   EEG.event(e).type = 'saccade_anti_right'
                   EEG.event(e).cond = 'anti';
                   EEG.event(e).dir = 'right';
               else
                   EEG.event(e).type = 'invalid';
               end
           end      
           if ~strcmp(EEG.event(e).type, 'L_fixation') ...
                   && ~strcmp(EEG.event(e).type, 'L_blink')
               previous = EEG.event(e).type;
           end
       end
      
%% remove everything from EEG.event which is not saccade or trigger
    
       tmpinv=find(strcmp({EEG.event.type}, 'invalid') | strcmp({EEG.event.type}, 'L_blink')) 
       EEG.event(tmpinv)=[]
       
       %% removing errors
       % if 10 and the sub didn't look left then error
       % pro left sac_start_x > sac_endpos_x --> correct condition
       tmperrsacc1=find(strcmp({EEG.event.type}, 'saccade_pro_left') & [EEG.event.sac_startpos_x]< [EEG.event.sac_endpos_x]);
       tmperr1=[tmperrsacc1 (tmperrsacc1-1)];
       EEG.event(tmperr1)=[];
       
       tmperrsacc2=find(strcmp({EEG.event.type}, 'saccade_anti_left') & [EEG.event.sac_startpos_x]> [EEG.event.sac_endpos_x]);
       tmperr2=[tmperrsacc2 (tmperrsacc2-1)];
       EEG.event(tmperr2)=[];
       
       tmperrsacc3=find(strcmp({EEG.event.type}, 'saccade_pro_right') & [EEG.event.sac_startpos_x]> [EEG.event.sac_endpos_x]);
       tmperr3=[tmperrsacc3 (tmperrsacc3-1)]
       EEG.event(tmperr3)=[];
       
       tmperrsacc4=find(strcmp({EEG.event.type}, 'saccade_anti_right') & [EEG.event.sac_startpos_x]< [EEG.event.sac_endpos_x]);
       tmperr4=[tmperrsacc4 (tmperrsacc4-1)];
       EEG.event(tmperr4)=[];
       
       %% amplitude too small
       tmperrsacc6=find(strcmp({EEG.event.type}, 'saccade_pro_right') ...
           & [EEG.event.sac_amplitude]<1.5)
       tmperrsacc7=find(strcmp({EEG.event.type}, 'saccade_pro_left') ...
           & [EEG.event.sac_amplitude]<1.5)
       tmperrsacc8=find(strcmp({EEG.event.type}, 'saccade_anti_left') ...
           & [EEG.event.sac_amplitude]<1.5)
       tmperrsacc9=find(strcmp({EEG.event.type}, 'saccade_anti_right') ...
           & [EEG.event.sac_amplitude]<1.5)
       tmperr69=[tmperrsacc6 (tmperrsacc6-1) tmperrsacc7 (tmperrsacc7-1) tmperrsacc8 (tmperrsacc8-1) tmperrsacc9 (tmperrsacc9-1)]
       EEG.event(tmperr69)=[];
       
       clear tmperrsacc1 tmperrsacc2 tmperrsacc3 tmperrsacc4 tmperrsacc6 tmperrsacc7 tmperrsacc8 tmperrsacc9
       
     
  
       %% delete cues where there was no saccade afterwards
%        tmperrcue10 = []
%        tmperrcue11 = []
%        
       %start with pro left cue 10
       tmperrcue10=  find(strcmp({EEG.event.type}, '10  ')) ;      
       for iii=1:length(tmperrcue10)
           pos = tmperrcue10(iii)
          if ~ (strcmp(EEG.event(pos+1).type , 'saccade_pro_left'))
       
        EEG.event(pos).type='missingsacc'; %cue
          end
       end  
       
       %%11
       tmperrcue11 =   find(strcmp({EEG.event.type}, '11  '))    ;   
       for iii=1:length(tmperrcue11)
           pos = tmperrcue11(iii)
          if ~ (strcmp(EEG.event(pos+1).type , 'saccade_pro_right'))
       
        EEG.event(pos).type='missingsacc'; %cue
          end
       end  
       
    tmpinv=find(strcmp({EEG.event.type}, 'missingsacc')) ;
    EEG.event(tmpinv)=[];
       
       
     
    
%% delete saccades and cues when the saccade comes faster than 100ms after cue
tmpevent=length(EEG.event)
saccpro=find(strcmp({EEG.event.type},'saccade_pro_right')==1 | strcmp({EEG.event.type},'saccade_pro_left')==1)% find rows where there is a saccade
saccanti=find(strcmp({EEG.event.type},'saccade_anti_right')==1 | strcmp({EEG.event.type},'saccade_anti_left')==1);%find rows where there is a saccade

for b=1:size(saccpro,2)
    
    if (EEG.event(saccpro(1,b)).latency-EEG.event(saccpro(1,b)-1).latency)<50 %50 because 100ms
        EEG.event(saccpro(b)).type='micro'; %saccade
        EEG.event(saccpro(b)-1).type = 'micro'; %cue
    end
end   
    
for b=1:size(saccanti,2)

    if (EEG.event(saccanti(b)).latency-EEG.event(saccanti(1,b)-1).latency)<50;
        EEG.event(saccanti(b)-1).type ='micro';
        EEG.event(saccanti(b)).type ='micro';
    end
    
end

    tmpinv=find(strcmp({EEG.event.type}, 'micro')) ;
    EEG.event(tmpinv)=[];





%%     epoching
EEGprocuesan= pop_epoch(EEG, {'10','11'}, [0, 1]);

%how many epochs
trialinfoprosan.epochs=size(EEGprocuesan.data, 3);

%% important

tmp=find(strcmp({EEGprocuesan.event.type}, '11  ') | strcmp({EEGprocuesan.event.type}, '10  '))
right= find(strcmp({EEGprocuesan.event(tmp).type},'11  ')==1);
left= find(strcmp({EEGprocuesan.event(tmp).type},'10  ')==1);

trialinfoprosan.cues  = nan(length(tmp),1);
trialinfoprosan.cues(left)= 0;
trialinfoprosan.cues(right)= 1;

%% save epoched data
if size(EEGprocuesan.data,3) ~= size(trialinfoprosan.cues,1) 
   error('this is bad')
end
     save EEGprocuesan EEGprocuesan

     save trialinfoprosan trialinfoprosan 
end
