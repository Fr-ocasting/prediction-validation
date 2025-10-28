import pandas as pd
import re
import io
import os
import sys 

current_file_path = os.path.abspath(os.path.dirname(__file__))
parent_dir = os.path.abspath(os.path.join(current_file_path,'..','..'))
if parent_dir not in sys.path:
    sys.path.insert(0,parent_dir)

from experiences.contextual_data_integration.exp4_15min_h1_results import results as results_Exp4_15min_h1
from experiences.contextual_data_integration.exp4_15min_results import results  as results_Exp4_15min
from experiences.contextual_data_integration.exp4_results import results  as results_Exp4
from experiences.contextual_data_integration.C_netmob.exp6_subway_netmob_results import results as results_Exp6_subway
from experiences.contextual_data_integration.C_netmob.exp6_bike_netmob_results import results as results_Exp6_bike
from experiences.contextual_data_integration.exp1_subway_in_results import results  as results_Exp1_subway_in
from experiences.contextual_data_integration.exp1_subway_out_results import results  as results_Exp1_subway_out
folder_path = 'save/K_fold_validation/training_wo_HP_tuning'
metrics = ['rmse','mae','mase']

dic_exp_to_names = {
    'Exp1': 'subway_out_STAEformer',
    'Exp1_subway_in':'subway_in_STAEformer',
    'Exp1_subway_out':'subway_out_STAEformer',

    'Exp2': 'bike_out_STAEformer',
    'Exp2_rainy': 'bike_out_STAEformer',
    'Exp3': 'bike_out_STAEformer',

    'Exp3_bike_15min_h4': 'bike_out_STAEformer',

    'Exp4': 'bike_out_STAEformer',
    'Exp4_15min': 'bike_out_STAEformer',
    'Exp4_15min_h1': 'bike_out_STAEformer',

    'Exp6_subway_netmob': 'subway_in_STAEformer',
    'Exp6_bike_netmob': 'bike_out_STAEformer',
}

# A supprimer : 
dic_exp_to_h = {
    'Exp1': 4,
    'Exp2': 4,
    'Exp3': 1,
    'Exp3_bike_15min_h4': 4,
    'Exp4': 1,
    'Exp4_15min': 4,
    'Exp4_15min_h1': 1,
}
# -----


def find_baseline(exp_i,h=None):
    if exp_i == 'Exp1':
        return 'STAEformer_subway_out_calendar__e200_h4'
    elif exp_i == 'Exp1_subway_in':
        return f'STAEformer_subway_in_calendar__e80_h{h}'
    elif exp_i == 'Exp1_subway_out':
        return f'STAEformer_subway_out_calendar__e80_h{h}'
    elif exp_i == 'Exp2':
        return 'STAEformer_bike_out_calendar__e200_h4'
    elif exp_i == 'Exp3_bike_15min_h4':
        return 'STAEformer_bike_out_calendar__e80_h4'
    elif exp_i == 'Exp3':
        return 'STAEformer_bike_out_calendar__e200_h1'
    elif exp_i == 'Exp4_15min':
        return 'STAEformer_bike_out_calendar__e80_h4'
    elif exp_i == 'Exp4':
        return 'STAEformer_bike_out_calendar__e120_h1'
    elif exp_i == 'Exp4_15min_h1':
        return 'STAEformer_bike_out_calendar__e80_h1'
    elif exp_i == 'Exp6_subway_netmob':
        return f'STAEformer_subway_in_calendar__e150_h{h}'
    elif exp_i == 'Exp6_bike_netmob':
        return f'STAEformer_bike_out_calendar__e150_h{h}'
    else:
        raise NotImplementedError


re._pattern = r'STAEformer.*?bis'


dic_trials = {'Exp1_subway_in': [c[:-4] for c in list(set(re.findall(re._pattern, results_Exp1_subway_in)))],
            
            'Exp1_subway_out': [c[:-4] for c in list(set(re.findall(re._pattern, results_Exp1_subway_out)))],
    
    'Exp1':[
            'STAEformer_subway_out_calendar__e200_h4',
            'STAEformer_subway_out_subway_in_calendar_early_fusion_feature_extractor__e200_h4',
            'STAEformer_subway_out_subway_in_calendar_early_fusion_independant_embedding__e100_h4',
            'STAEformer_subway_out_subway_in_calendar_early_fusion_shared_embedding__e200_h4',
            'STAEformer_subway_out_subway_in_calendar_late_fusion_feature_extractor__e200_h4',
            'STAEformer_subway_out_subway_in_calendar_late_fusion_simple_embedding__e200_h4',
            'STAEformer_subway_out_subway_in_calendar_late_fusion_traffic_model_backbone__e100_h4'
                    ],
        
    'Exp2_rainy': ['STAEformer_bike_out_calendar__e200_h4',
            'STAEformer_bike_out_weather_calendar_late_fusion_feature_extractor__e200_h4',
            'STAEformer_bike_out_weather_calendar_late_fusion_s_proj_t_proj__e200_h4',
            'STAEformer_bike_out_weather_calendar_late_fusion_repeat_t_proj__e200_h4',
            'STAEformer_bike_out_weather_calendar_early_fusion_repeat_t_proj__e200_h4',
            'STAEformer_bike_out_weather_calendar_early_fusion_s_proj_t_proj__e200_h4',
            'STAEformer_bike_out_weather_calendar_early_fusion_feature_extractor__e200_h4'
                    ],

    'Exp2':['STAEformer_bike_out_calendar__e200_h4',
            'STAEformer_bike_out_weather_calendar_late_fusion_feature_extractor__e200_h4',
            'STAEformer_bike_out_weather_calendar_late_fusion_s_proj_t_proj__e200_h4',
            'STAEformer_bike_out_weather_calendar_late_fusion_repeat_t_proj__e200_h4',
            'STAEformer_bike_out_weather_calendar_early_fusion_repeat_t_proj__e200_h4',
            'STAEformer_bike_out_weather_calendar_early_fusion_s_proj_t_proj__e200_h4',
            'STAEformer_bike_out_weather_calendar_early_fusion_feature_extractor__e200_h4'
                    ],

                    
            'Exp3':[
                    'STAEformer_bike_out_calendar__e200_h1',
                    'STAEformer_bike_out_subway_out_weather_calendar_late_fusion_cross_attn_traffic_model_backbone__e200_h1',
                    'STAEformer_bike_out_subway_out_weather_calendar_early_fusion_s_proj_t_proj__e200_h1',
                    'STAEformer_bike_out_subway_out_weather_calendar_late_fusion_adp_query_cross_attn_traffic_model_backbone__e200_h1',
                    ],

            'Exp3_bike_15min_h4':[
                    'STAEformer_bike_out_subway_out_weather_calendar_late_fusion_adp_query_cross_attn_traffic_model_backbone__e200_h4',
                    'STAEformer_bike_out_subway_out_weather_calendar_late_fusion_cross_attn_traffic_model_backbone__e200_h4',
                    'STAEformer_bike_out_subway_out_weather_calendar_early_fusion_s_proj_t_proj__e200_h4',
                    'STAEformer_bike_out_calendar__e200_h4',
                    ],

            'Exp4': [c[:-4] for c in list(set(re.findall(re._pattern, results_Exp4)))],

            'Exp4_15min': [c[:-4] for c in list(set(re.findall(re._pattern, results_Exp4_15min)))],

            'Exp4_15min_h1': [c[:-4] for c in list(set(re.findall(re._pattern, results_Exp4_15min_h1)))],

            'Exp6_subway_netmob': [c[:-4] for c in list(set(re.findall(re._pattern, results_Exp6_subway)))],

            'Exp6_bike_netmob':  [c[:-4] for c in list(set(re.findall(re._pattern, results_Exp6_bike)))],
}


