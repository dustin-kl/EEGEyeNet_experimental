# configuration used by the training and evaluation methods
# let's keep it here to have a clean code on other methods that we try
import time
import os
config = dict()

##################################################################
##################################################################
############### BENCHMARK CONFIGURATIONS #########################
##################################################################
##################################################################
# 'LR_task' (dataset: 'antisaccade'):
# 'Direction_task' (dataset: 'dots' or 'processing_speed' or 'zuco'): dots = "Large Grid Dataset" and processing_speed = "Visual Symbol Search"
# 'Position_task' (dataset: 'dots' or 'zuco'):
# 'Age_task' (dataset: 'antisaccade): for age regression 
# 'Age_task_binary' (dataset: 'antisaccade'): for age classification into people under/over 35 years 
config['task'] = 'Position_task'
config['dataset'] = 'dots'
config['preprocessing'] = 'min'  # max or min
config['feature_extraction'] = True 
config['include_ML_models'] = False
config['include_DL_models'] = False
config['include_your_models'] = False
config['include_dummy_models'] = True

##################################################################
##################################################################
############### PATH CONFIGURATIONS ##############################
##################################################################
##################################################################
# Where experiment results are stored.
config['log_dir'] = './runs/'
# Path to training, validation and test data folders.
config['data_dir'] = './data/'
# Path of root
config['root_dir'] = '.'
# Retrain or load already trained
config['retrain'] = True
config['pretrained'] = False 
config['save_models'] = True
# If retrain is false we need to provide where to load the experiment files
config['load_experiment_dir'] = ''
# all_EEG_file should specify the name of the file where the prepared data is located (if emp
def build_file_name():
    all_EEG_file = config['task'] + '_with_' + config['dataset']
    all_EEG_file = all_EEG_file + '_' + 'synchronised_' + config['preprocessing']
    all_EEG_file = all_EEG_file + ('_hilbert.npz' if config['feature_extraction'] else '.npz')
    return all_EEG_file
config['all_EEG_file'] = build_file_name() # or use your own specified file name


##################################################################
##################################################################
############### MODELS CONFIGURATIONS ############################
##################################################################
##################################################################
# Specific to models now
config['framework'] = 'tensorflow' # pytorch or tensorflow 
config['learning_rate'] = 1e-5
config['early_stopping'] = True
config['patience'] = 20

##################################################################
##################################################################
############### HELPER VARIABLES #################################
##################################################################
##################################################################
config['trainX_variable'] = 'EEG'
config['trainY_variable'] = 'labels'


def create_folder():
    if config['retrain']:
        model_folder_name = str(int(time.time()))
        model_folder_name += '_' + config['task'] + '_' + config['dataset'] + '_' + config['preprocessing'] 
        model_folder_name += '_hilbert' if config['feature_extraction'] else ''
        model_folder_name += '_pretrained' if config['pretrained'] and config['include_DL_models'] else ''
        model_folder_name += f"_{config['framework']}" if config['include_DL_models'] else ''
        model_folder_name += f"_patience_{config['patience']}" if config['include_DL_models'] else ''
        model_folder_name += f"_lr_{config['learning_rate']}" if config['include_DL_models'] else ''
        model_folder_name += "_standML" if config['include_ML_models'] else ''
        model_folder_name += "_DUMMY" if config['include_dummy_models'] else ''

        config['model_dir'] = os.path.abspath(os.path.join(config['log_dir'], model_folder_name))
        config['checkpoint_dir'] = config['model_dir'] + '/checkpoint/'
        if not os.path.exists(config['model_dir']):
            os.makedirs(config['model_dir'])

        if not os.path.exists(config['checkpoint_dir']):
            os.makedirs(config['checkpoint_dir'])

        config['info_log'] = config['model_dir'] + '/' + 'info.log'
        config['batches_log'] = config['model_dir'] + '/' + 'batches.log'

    else:
        config['model_dir'] = config['log_dir'] + config['load_experiment_dir']
        config['checkpoint_dir'] = config['model_dir'] + 'checkpoint/'
        stamp = str(int(time.time()))
        config['info_log'] = config['model_dir'] + '/' + 'inference_info_' + stamp + '.log'
        config['batches_log'] = config['model_dir'] + '/' + 'inference_batches_' + stamp + '.log'