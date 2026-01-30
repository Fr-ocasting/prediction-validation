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
                                            weather_contextual_kwargs,
                                            netmob_preprocessing_kwargs=None
                                            ):
        
        for fusion_type, config_contextual_kwargs in possible_contextual_kwargs.items():
            for feature_extractor_type, contextual_kwargs_i in config_contextual_kwargs.items():

                # --- Load contextual kwargs : 
                contextual_kwargs = self.clean_contextual_kwargs(contextual_dataset_names,contextual_kwargs_i,weather_contextual_kwargs)

                # --- Feed 'dic_configs' with same configuration for all horizons and repetitions: 
                for horizon in self.horizons:
                    for n_bis in range(1,self.REPEAT_TRIAL+1): # range(1,6):
                        dataset_names =  [self.target_data] +contextual_dataset_names+ ['calendar']
                        name_i = f"{self.model_name}_{'_'.join(dataset_names)}_{fusion_type}_{feature_extractor_type}"
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
                                    'contextual_kwargs' : contextual_kwargs,
                                    'denoising_names':[],
                                    'bool_sanity_checker' : self.SANITY_CHECKER 
                                    } 
                    
                        config_i = self.smoothing(config_i,netmob_preprocessing_kwargs,contextual_dataset_names)
                        config_i.update(config_backbone_model)
                        config_i.update(self.compilation_modification)
                        dic_configs[name_i] = config_i
                # ----

        return dic_configs

    def clean_contextual_kwargs(self,contextual_dataset_names,contextual_kwargs_i,weather_contextual_kwargs):
        ''' Build contextual kwargs as a dictionnary where all the configuration are the same'''
        contextual_kwargs ={'subway_out':contextual_kwargs_i,
                            'subway_in':contextual_kwargs_i,
                            'subway_in_subway_out':contextual_kwargs_i,
                            'weather':weather_contextual_kwargs,
                            'netmob_POIs':contextual_kwargs_i
                            }
        
        if 'weather' not in contextual_dataset_names:
            contextual_kwargs.pop('weather',None)  
        if 'subway_in' not in contextual_dataset_names:
            contextual_kwargs.pop('subway_in',None)  
        if 'subway_out' not in contextual_dataset_names:
            contextual_kwargs.pop('subway_out',None)
        if 'subway_in_subway_out' not in contextual_dataset_names:
            contextual_kwargs.pop('subway_in_subway_out',None) 
        if 'netmob_POIs' not in contextual_dataset_names:
            contextual_kwargs.pop('netmob_POIs',None)
        return contextual_kwargs
    
    def smoothing(self,config_i,netmob_preprocessing_kwargs,contextual_dataset_names):
        if (netmob_preprocessing_kwargs is not None) and 'netmob_POIs' in contextual_dataset_names:
            if 'denoiser_kwargs' in netmob_preprocessing_kwargs.keys():
                config_i.update({'denoising_names':['netmob_POIs'],
                                'denoiser_names':["exponential"],   # ['median'], ['exponential'], ['savitzky_golay']         # un seul filtre
                                'denoiser_kwargs': netmob_preprocessing_kwargs['denoiser_kwargs']}
                                )
        return config_i
        

   
