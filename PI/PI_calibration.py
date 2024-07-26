import numpy as np
import torch 
from utils.utilities import get_higher_quantile


class Calibrator(object):
    def __init__(self,alpha,device):
        super(Calibrator,self).__init__()
        self.device = device
        self.alpha = alpha
        self.str_info = ''

    def get_prediction(self,trainer):
        trainer.training_mode = 'cal'
        Preds,Y_cal,T_cal = trainer.loop_epoch()
        self.Preds = Preds
        self.Y_cal = Y_cal
        self.T_cal = T_cal
    
    def get_lower_upper_bands(self):
        # Adapt dimensions 
        if self.Preds.dim() == 2: self.Preds = self.Preds.unsqueeze(1)
            
        # get lower and upper band
        if self.Preds.size(-1) == 2: self.lower_q,self.upper_q = self.Preds[...,0].unsqueeze(-1),self.Preds[...,1].unsqueeze(-1)   # The Model return ^q_l and ^q_u associated to x_b
        elif self.Preds.size(-1) == 1: self.lower_q,self.upper_q = self.Preds,self.Preds 
        else:raise ValueError(f"Shape of model's prediction: {self.Preds.size()}. Last dimension should be 1 or 2.")
        # ...

    def unormalize(self,trainer):
        # unormalized lower and upper band  
        self.lower_q = trainer.dataset.normalizer.unormalize_tensor(inputs = self.lower_q,feature_vect = True ) # ,device = self.args.device
        self.upper_q  = trainer.dataset.normalizer.unormalize_tensor(inputs = self.upper_q,feature_vect = True ) # , device = self.args.device
        self.Y_cal = trainer.dataset.normalizer.unormalize_tensor(inputs = self.Y_cal, feature_vect = True ) # ,device = self.args.device
        # ...
    
    def get_conformity_scores(self,conformity_scores_type):
        # Get Confority scores: 
        if conformity_scores_type == 'max_residual':
            self.conformity_scores = torch.max(self.lower_q-self.Y_cal,self.Y_cal-self.upper_q).to(self.device) # Element-wise maximum        #'max(lower_q-y_b,y_b-upper_q)' is the quantile regression error function

        if conformity_scores_type == 'max_residual_plus_middle':
            self.str_info = self.str_info+ "\n|!| Conformity scores computation is not based on 'max(ql-y, y-qu)'"
            self.conformity_scores = torch.max(self.lower_q-self.Y_cal,self.Y_cal-self.upper_q) + ((self.lower_q>self.Y_cal)(self.upper_q<self.Y_cal))*(self.upper_q - self.lower_q)/2  # Element-wise maximum        #'max(lower_q-y_b,y_b-upper_q)' is the quantile regression error function
        # ...

    def get_quantile_tensor(self,quantile_method):
        if quantile_method == 'classic':
            self.get_conformalized_quantile()
        elif quantile_method == 'compute_quantile_by_class':
            self.get_conformalized_quantile_by_time_slots_class()
        else:
            raise NotImplementedError(f"Quantile method {quantile_method} is not still implemented")


    def get_conformalized_quantile(self):
        quantile_order = torch.Tensor([np.ceil((1 - self.alpha)*(self.Y_cal.size(0)+1))/self.Y_cal.size(0)]).to(self.device)
        #Q = torch.quantile(self.conformity_scores, quantile_order, dim = 0).to(self.device) #interpolation = 'higher'
        self.Q = get_higher_quantile(self.conformity_scores,quantile_order,device = self.device)


    def get_conformalized_quantile_by_time_slots_class(self):
        dic_label2Q = {}
        nb_label_with_quantile_1 = 0
        for label in self.T_cal.unique():
            indices = torch.nonzero(self.T_cal == label,as_tuple = True)[0]
            quantile_order = torch.Tensor([np.ceil((1 - self.alpha)*(indices.size(0)+1))/indices.size(0)]).to(self.device)  # Quantile for each class, so the quantile order is different as each class has a different length
            quantile_order = min(torch.Tensor([1]).to(self.device),quantile_order)
            if quantile_order == 1: 
                nb_label_with_quantile_1 +=1
                #print(f"label {label} has only {indices.size(0)} elements in his class. We then use quantile order = 1")
            conformity_scores_i = self.conformity_scores[indices]
            scores_counts = conformity_scores_i.size(0)
            Q_i = get_higher_quantile(conformity_scores_i,quantile_order,device = self.device)
            #Q_i = torch.quantile(conformity_scores_i, quantile_order, dim = 0)#interpolation = 'higher'
            dic_label2Q[label.item()]= {'Q': Q_i,'count':scores_counts}

        self.str_info = self.str_info+ f"\nProportion of label with quantile order set to 1: {'{:.1%}'.format(nb_label_with_quantile_1/len(self.T_cal.unique()))}"
        print(self.str_info)
        self.Q = dic_label2Q