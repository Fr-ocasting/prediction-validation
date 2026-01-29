class BaselineConfigBuilder(object):
    def __init__(self,target_data,contextual_dataset_names, model_name,horizons,freq,REPEAT_TRIAL,SANITY_CHECKER,compilation_modification,add_name_save):
        self.target_data = target_data
        self.contextual_dataset_names = contextual_dataset_names
        self.model_name = model_name
        self.horizons = horizons
        self.freq = freq
        self.REPEAT_TRIAL = REPEAT_TRIAL
        self.SANITY_CHECKER = SANITY_CHECKER
        self.compilation_modification = compilation_modification
        self.add_name_save = add_name_save


    def build_config_single_contextual(self, dic_configs, possible_target_kwargs,config_backbone_model):
        contextual_kwargs ={}
        for horizon in self.horizons:
            for n_bis in range(1,self.REPEAT_TRIAL+1): # range(1,6):
                dataset_names =  [self.target_data] + ['calendar']
                name_i = f"{self.model_name}_{'_'.join(dataset_names)}{self.add_name_save}"
                name_i_end = f"_e{config_backbone_model['epochs']}_h{horizon}_bis{n_bis}"
                name_i = f"{name_i}_{name_i_end}"

                config_i =  {'target_data': self.target_data,
                            'dataset_names': dataset_names,
                            'model_name': self.model_name,
                            'dataset_for_coverage': [self.target_data]+self.contextual_dataset_names,
                            'freq': self.freq,
                            'horizon_step': horizon,
                            'step_ahead': horizon,
                            'target_kwargs' : {self.target_data: possible_target_kwargs[self.target_data]},
                            'contextual_kwargs' : contextual_kwargs,
                            'denoising_names':[],
                            'bool_sanity_checker' : self.SANITY_CHECKER
                            } 
                config_i.update(config_backbone_model)
                config_i.update(self.compilation_modification)

                dic_configs[name_i] = config_i
        return dic_configs