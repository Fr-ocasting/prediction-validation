# Load saved results and apply it on rainy day : 
# ----------
# 1. For Each Target, for each config with, for each horizon, for each bis:
#       i. load datset, model, and trainer 
#       ii. load saved model weights 
# 2. Forward prediction, get desagregated metrics 

# 3. Get RMSE, MAE, MASE, MAPE on rainy day with the format:
#   f"{model_name}_{target_name}_{contextual_data}_{fusion_strategie}_{feature_extractor}_p{percentage}__e{epochs}h{horizon}_bis{bis}:   All Steps RMSE = {RMSE:.3f}, MAE = {MAE:.3f}, MASE = {MASE:.3f}, MAPE = {MAPE:.3f}"

# -- Config Init 
import os 
import sys 

current_path = os.path.abspath(os.path.dirname(__file__))
parent_dir = os.path.abspath(os.path.join(current_path, '..','..','..'))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from pipeline.Evaluation.accuracy_comparison import load_trainer_ds_from_2_trials,get_predict_real_and_inputs,get_model_args
from pipeline.Evaluation.accuracy_comparison import get_rainy_indices,plot_analysis_comparison_2_config,get_previous_and_prediction
import torch 
from pipeline.utils.metrics import evaluate_metrics

def get_dict_metrics_on_rainy_events(full_predict1,full_predict2,Y_true,X,args_init1,args_init2,ds2,ds1):
    h_idx = 1
    metric_list = ['rmse','mse','mae','mase','mape']
    previous,_,_,_ = get_previous_and_prediction(full_predict1,full_predict2,Y_true,X,h_idx)
    assert args_init1.horizon_step == args_init1.step_ahead, "Horizon step must be equal to step_ahead here"

    print("\nComparison on between models across all time-slots followed by comparison on Rainy Events Only")
    _,train_rainy_indices,_,total_indices = get_rainy_indices(args = args_init2,ds = ds1,training_mode = 'train')
    print(f"Number of rainy time-slots in the train set: {len(train_rainy_indices)}, i.e {len(train_rainy_indices)/total_indices*100:.2f} % of the train set")
    # ---- Plot Accuracy comparison on rainy moments only ----
    mask,rainy_indices,df_weather,total_indices = get_rainy_indices(args = args_init2,ds = ds2,training_mode = 'test')
    print(f"Number of rainy time-slots in the test set: {len(rainy_indices)}, i.e {len(rainy_indices)/total_indices*100:.2f} % of the test set\n")

    dates = mask[mask].index
    masked_index = mask.reset_index(drop=True)
    masked_index = masked_index[masked_index].index


    # --- Get Prediction on rainy time-slots:
    rainy_predict1 = torch.index_select(full_predict1,0,torch.tensor(masked_index).long())
    rainy_predict2 = torch.index_select(full_predict2,0,torch.tensor(masked_index).long())
    rainy_Y_true = torch.index_select(Y_true,0,torch.tensor(masked_index).long())
    rainy_previous = torch.index_select(previous,0,torch.tensor(masked_index).long())

    dic_metric1 = evaluate_metrics(rainy_predict1,rainy_Y_true,metrics = metric_list, previous = rainy_previous,horizon_step = h_idx)
    dic_metric1 = dict(dic_metric1.copy())
    dic_metric2 = evaluate_metrics(rainy_predict2,rainy_Y_true,metrics = metric_list, previous = rainy_previous,horizon_step = h_idx)
    dic_metric2 = dict(dic_metric2.copy())
    return dic_metric1,dic_metric2



dic_contextual_data = {'bike_out': [[],['weather','calendar']],
                    'subway_out': [[],['subway_in','calendar'],['subway_in','weather','calendar']]
                    # 'subway_out': [['subway_in','weather','calendar']]
                    }

dic_fusion_strategie = {('bike_out',()): '',
                        ('bike_out',('weather','calendar')):'early_fusion',
                        ('subway_out',()):'',
                        ('subway_out',('subway_in','calendar')):'early_fusion',
                        ('subway_out',('subway_in','weather','calendar')):'early_fusion'
                        }
             
dic_feature_extractor = {('bike_out',()): '',
                        ('bike_out',('weather','calendar')):'repeat_t_proj',
                        ('subway_out',()):'',
                        ('subway_out',('subway_in','calendar')):'shared_embedding',
                        ('subway_out',('subway_in','weather','calendar')):'shared_embedding_repeat_t_proj'
                        }


model_name = 'STAEformer'
epochs = 50
range_k = range(1,6)
training_mode = 'test'
modification = {'shuffle':False,
                'data_augmentation':False,
                'torch_compile': False,
                }
log = ''
for percentage in [5, 10, 15, 25, 50, 75, 100]:
    # for horizon in [1,4]:
    for horizon in [1]:
        # for target in ['bike_out','subway_out']:
        for target in ['subway_out']:
            # for contextual_data in dic_contextual_data[target]:
            for contextual_data in [['subway_in','weather','calendar']]:
                reversed_metric = False
                fusion_strategie = dic_fusion_strategie[(target,tuple(contextual_data))]
                feature_extractor = dic_feature_extractor[(target,tuple(contextual_data))]
                    
                
                if not('weather' in contextual_data):
                    contextual_data1 = ['weather','calendar'] if target == 'bike_out' else ['subway_in','weather','calendar']
                    fusion_strategie1 = dic_fusion_strategie[(target,tuple(contextual_data1))]
                    feature_extractor1 = dic_feature_extractor[(target,tuple(contextual_data1))]

                    if contextual_data == []:
                        # trial_id1 = f"{model_name}_{target}_{percentage}p__e{epochs}_h{horizon}_bis"
                        trial_id1 = f"{model_name}_{target}_ExpandingTrain{percentage}p__e{epochs}_h{horizon}_bis"
                    else:
                        # trial_id1 = f"{model_name}_{target}_{'_'.join(contextual_data)}_{fusion_strategie}_{feature_extractor}_{percentage}p__e{epochs}_h{horizon}_bis"
                        trial_id1 = f"{model_name}_{target}_{'_'.join(contextual_data)}_{fusion_strategie}_{feature_extractor}_ExpandingTrain{percentage}p__e{epochs}_h{horizon}_bis"

                    # trial_id2 = f"{model_name}_{target}_{'_'.join(contextual_data1)}_{fusion_strategie1}_{feature_extractor1}_{percentage}p__e{epochs}_h{horizon}_bis"
                    trial_id2 = f"{model_name}_{target}_{'_'.join(contextual_data1)}_{fusion_strategie1}_{feature_extractor1}_ExpandingTrain{percentage}p__e{epochs}_h{horizon}_bis"
                    reversed_metric = True 
                else:
                    # trial_id1 = f"{model_name}_{target}_{percentage}p__e{epochs}_h{horizon}_bis"
                    # trial_id2 = f"{model_name}_{target}_{'_'.join(contextual_data)}_{fusion_strategie}_{feature_extractor}_{percentage}p__e{epochs}_h{horizon}_bis"
                    # trial_id1 = f"{model_name}_{target}_ExpandingTrain{percentage}p__e{epochs}_h{horizon}_bis"
                    trial_id1 = f"{model_name}_{target}_subway_in_calendar_early_fusion_shared_embedding_ExpandingTrain{percentage}p__e{epochs}_h{horizon}_bis"
                    trial_id2 = f"{model_name}_{target}_{'_'.join(contextual_data)}_{fusion_strategie}_{feature_extractor}_ExpandingTrain{percentage}p__e{epochs}_h{horizon}_bis"

                # model_args,_,path_model_args,_ = get_model_args(save_folder_name = f'Exp5/{target}_{model_name}')
                model_args,_,path_model_args,_ = get_model_args(save_folder_name = f'Exp5_ExpandingTrain/{target}_{model_name}')
                ds1,ds2,args_init1,args_init2 = None, None, None, None

                # for k in range_k:
                for k in range(1,2):
                    trial_id1_updated = f"{trial_id1}{k}_f5"
                    trial_id2_updated = f"{trial_id2}{k}_f5"

                    trainer1,trainer2,ds1,ds2,args_init1,args_init2 = load_trainer_ds_from_2_trials(trial_id1_updated,trial_id2_updated,modification = modification,
                                                                                                    model_args=model_args,
                                                                                                    path_model_args=path_model_args,
                                                                                                    ds1_init=ds1,ds2_init=ds2,
                                                                                                    args_init1=args_init1,args_init2=args_init2,
                                                                                                    )
                                                                                                    

                    full_predict1,full_predict2,Y_true,X = get_predict_real_and_inputs(trainer1,trainer2,ds1,ds2,training_mode=training_mode)

                    globals()[f"trainer1_bis{k}"] = trainer1
                    globals()[f"trainer2_bis{k}"] = trainer2
                    globals()[f"ds1_bis{k}"] = ds1
                    globals()[f"ds2_bis{k}"] = ds2
                    globals()[f"full_predict1_bis{k}"] = full_predict1
                    globals()[f"full_predict2_bis{k}"] = full_predict2

                    dic_metric1,dic_metric2 = get_dict_metrics_on_rainy_events(globals()[f"full_predict1_bis{k}"],globals()[f"full_predict2_bis{k}"],Y_true,X,args_init1,args_init2,ds2,ds1)

                    # Keep track on metric from model 1
                    if reversed_metric:
                        dic_metric2 = dict(dic_metric1.copy())
                    RMSE = dic_metric2['rmse_all']
                    MAE = dic_metric2['mae_all']
                    MASE = dic_metric2['mase_all']
                    MAPE = dic_metric2['mape_all']
                    # if feature_extractor == '':
                    #     log += f"{model_name}_{target}_{percentage}p__e{epochs}_h{horizon}_bis{k}:   All Steps RMSE = {RMSE:.3f}, MAE = {MAE:.3f}, MASE = {MASE:.3f}, MAPE = {MAPE:.3f}\n"
                    # else:
                    #     log += f"{model_name}_{target}_{'_'.join(contextual_data)}_{fusion_strategie}_{feature_extractor}_{percentage}p__e{epochs}_h{horizon}_bis{k}:   All Steps RMSE = {RMSE:.3f}, MAE = {MAE:.3f}, MASE = {MASE:.3f}, MAPE = {MAPE:.3f}\n"
                    if feature_extractor == '':
                        log += f"{model_name}_{target}_ExpandingTrain{percentage}p__e{epochs}_h{horizon}_bis{k}:   All Steps RMSE = {RMSE:.3f}, MAE = {MAE:.3f}, MASE = {MASE:.3f}, MAPE = {MAPE:.3f}\n"
                    else:
                        log += f"{model_name}_{target}_{'_'.join(contextual_data)}_{fusion_strategie}_{feature_extractor}_ExpandingTrain{percentage}p__e{epochs}_h{horizon}_bis{k}:   All Steps RMSE = {RMSE:.3f}, MAE = {MAE:.3f}, MASE = {MASE:.3f}, MAPE = {MAPE:.3f}\n"
                print(log)