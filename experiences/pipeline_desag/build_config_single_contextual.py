import itertools

class ConfigBuilder(object):
    def __init__(self,target_data,contextual_dataset_names,dataset_for_coverage,model_name,horizons,freq,REPEAT_TRIAL,SANITY_CHECKER,compilation_modification):
        self.target_data = target_data
        self.model_name = model_name
        self.contextual_dataset_names = contextual_dataset_names
        self.dataset_for_coverage = dataset_for_coverage
        self.horizons = horizons
        self.freq = freq
        self.REPEAT_TRIAL = REPEAT_TRIAL
        self.SANITY_CHECKER = SANITY_CHECKER
        self.compilation_modification = compilation_modification

    def build_config_single_contextual(self, dic_configs,
                                            possible_target_kwargs,
                                            config_backbone_model,
                                            contextual_dataset_names,
                                            possible_contextual_kwargs,
                                            netmob_preprocessing_kwargs=None
                                            ):
        all_configs = self.get_all_combinations(possible_contextual_kwargs)

        for config in all_configs:
            contextual_dataset_names = list([item[0] for item in config])
            dic_contextual_kwargs = {ds: possible_contextual_kwargs[ds][fusion][feature_extractor] for ds, fusion, feature_extractor in config}
            # --- Feed 'dic_configs' with same configuration for all horizons and repetitions: 
            for horizon in self.horizons:
                for n_bis in range(1,self.REPEAT_TRIAL+1): # range(1,6):
                    dataset_names =  [self.target_data] +list(dic_contextual_kwargs.keys())+ ['calendar']

                    L_fusion_type = [item[1] for item in config]
                    L_feature_extractor_type = [item[2] for item in config]

                    str_fusion_extractor_type = '_'.join([f"{f}_{e}" for f,e in zip(L_fusion_type,L_feature_extractor_type)])    

                    name_i = f"{self.model_name}_{'_'.join(dataset_names)}_{str_fusion_extractor_type}"
                    name_i_end = f"_e{config_backbone_model['epochs']}_h{horizon}_bis{n_bis}"
                    name_i = f"{name_i}_{name_i_end}"

                    config_i =  {'target_data': self.target_data,
                                'dataset_names': dataset_names,
                                'model_name': self.model_name,
                                'dataset_for_coverage': self.dataset_for_coverage,
                                'freq': self.freq,
                                'horizon_step': horizon,
                                'step_ahead': horizon,
                                'target_kwargs' : {self.target_data: possible_target_kwargs[self.target_data]},
                                'contextual_kwargs' : dic_contextual_kwargs,
                                'denoising_names':[],
                                'bool_sanity_checker' : self.SANITY_CHECKER 
                                } 
                
                    config_i = self.smoothing(config_i,netmob_preprocessing_kwargs,contextual_dataset_names)
                    config_i.update(config_backbone_model)
                    config_i.update(self.compilation_modification)
                    dic_configs[name_i] = config_i
            # ----

        return dic_configs
    


    def get_all_combinations(self,possible_contextual_kwargs):
        # Extraction des triplets (dataset, fusion, feature_extractor) par dataset
        ds_triplets = {
            ds: [(ds, f, e) for f, extractors in fusions.items() for e in extractors]
            for ds, fusions in possible_contextual_kwargs.items()
        }
        
        datasets = list(ds_triplets.keys())
        K_total = len(datasets)
        all_configs = []
        
        # Itération sur la taille du sous-ensemble de datasets (1 à K)
        for r in range(1, K_total + 1):
            for selected_ds in itertools.combinations(datasets, r):
                pools = [ds_triplets[ds] for ds in selected_ds]
                for combo in itertools.product(*pools):
                    all_configs.append(list(combo))
                    
        return all_configs

    def smoothing(self,config_i,netmob_preprocessing_kwargs,contextual_dataset_names):
        if (netmob_preprocessing_kwargs is not None) and 'netmob_POIs' in contextual_dataset_names:
            if 'denoiser_kwargs' in netmob_preprocessing_kwargs.keys():
                config_i.update({'denoising_names':['netmob_POIs'],
                                'denoiser_names':["exponential"],   # ['median'], ['exponential'], ['savitzky_golay']         # un seul filtre
                                'denoiser_kwargs': netmob_preprocessing_kwargs['denoiser_kwargs']}
                                )
        return config_i
            



    # def clean_contextual_kwargs(self,contextual_dataset_names,contextual_kwargs_i,weather_contextual_kwargs):
    #     ''' Build contextual kwargs as a dictionnary where all the configuration are the same'''
    #     contextual_kwargs ={'subway_out':contextual_kwargs_i,
    #                         'subway_in':contextual_kwargs_i,
    #                         'subway_in_subway_out':contextual_kwargs_i,
    #                         'weather':weather_contextual_kwargs,
    #                         'netmob_POIs':contextual_kwargs_i,
    #                         'bike_in':contextual_kwargs_i,
    #                         'bike_out':contextual_kwargs_i
    #                         }
        
    #     if 'weather' not in contextual_dataset_names:
    #         contextual_kwargs.pop('weather',None)  
    #     if 'subway_in' not in contextual_dataset_names:
    #         contextual_kwargs.pop('subway_in',None)  
    #     if 'subway_out' not in contextual_dataset_names:
    #         contextual_kwargs.pop('subway_out',None)
    #     if 'subway_in_subway_out' not in contextual_dataset_names:
    #         contextual_kwargs.pop('subway_in_subway_out',None) 
    #     if 'netmob_POIs' not in contextual_dataset_names:
    #         contextual_kwargs.pop('netmob_POIs',None)
    #     if 'bike_in' not in contextual_dataset_names:
    #         contextual_kwargs.pop('bike_in',None)
    #     if 'bike_out' not in contextual_dataset_names:
    #         contextual_kwargs.pop('bike_out',None)
    #     return contextual_kwargs

   
