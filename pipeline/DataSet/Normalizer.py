import torch
import numpy as np 
from packaging.version import Version

class Normalizer(object):
    def __init__(self,reference = None,minmaxnorm = False, standardize = False, dims = None):
        self.minmaxnorm = minmaxnorm
        self.standardize = standardize
        self.dims = dims
        reshaped_inputs = self.reshape_input(reference,dims)
        self.get_stats(reshaped_inputs)  # Get Min, Max, Mean, Std 

    def reshape_input(self,inputs,dims):
        # Design Permutation tuple: 
        int_dims = [dim if dim>=0 else inputs.dim()+dim for dim in dims ]   
        int_dims = sorted(int_dims)
        remaining_dims = [dim for dim in np.arange(inputs.dim()) if not(dim in int_dims)] 
        permutations = remaining_dims+int_dims
        self.permutations = permutations
        
        #Permute 
        permuted_inputs = inputs.permute(tuple(permutations))
        self.permuted_size = permuted_inputs.size()
        
        # Reshape (flattening 'input' through dimension 'dims')
        reshape = tuple([permuted_inputs.size(k) for k,_ in enumerate(remaining_dims)]+[-1]) 
        reshaped_inputs = permuted_inputs.reshape(reshape)

        self.reshaped_inputs_dim =  reshaped_inputs.size()
        return(reshaped_inputs)

    def get_stats(self,inputs: torch.Tensor): #,dims: tuple
        ''' Return Min, Max, Mean and Std of inputs through the choosen dimension 'dims' (which have been flattened)'''
        if (not(hasattr(self,'mini'))):
            self.mini = inputs.min(-1).values  
            #self.mini = inputs.min(dims).values  
        if (not(hasattr(self,'maxi'))): 
            self.maxi = inputs.max(-1).values
            #self.maxi = inputs.max(dims).values
        if (not(hasattr(self,'mean'))):
            self.mean= inputs.mean(-1)
            #self.mean = inputs.mean(dims)
        if (not(hasattr(self,'std'))): 
            self.std = inputs.std(-1)
            #print('self.std: ',self.std)
            #self.std = inputs.std(dims) 

    def repeat_stats_tensor(self,X,S, feature_vect = False):
        '''
        According to argument 'dims', reshape and repeat tensor S to match dimension with X.

        args
        ----
        X : feature vector              >>>> torch.randn(T',N,C,H,W,L)
        I : Train input                 >>>> torch.randn(T,N,C,H,W)
        S : statistics (mini,mean...)   >>>> torch.randn(N,H)
        dims : dimension for which we have aggregated >>>> [0,2,4]  # cause we removed T,C,W from 'Train input'
        '''
        S = S.to(X)
        reshaped_vector, repeat_vector = [1]*X.dim(),[1]*X.dim()

        # Dépend de si c'est un Feature Vector (pour lequel on a ajouté une dimension L), ou un Input (comme train_input)
        conj_dims = [x for x in np.arange(X.dim()-1) if not x in self.dims] if feature_vect else [x for x in np.arange(X.dim()) if not x in self.dims]

        #Design re-shaping:
        for k,c in enumerate(conj_dims):
            reshaped_vector[c] = S.size(k)

        # Design repeating:
        for k,c in enumerate(X.size()):
            if reshaped_vector[k] == 1:
                repeat_vector[k] = c

        # Apply reshaped and repeat:
        reshaped_S = S.reshape(tuple(reshaped_vector))
        reshaped_S = reshaped_S.repeat(tuple(repeat_vector))
        return(reshaped_S)

    def transform(self,inputs: torch.Tensor, reverse: bool = False,feature_vect: bool = False):

        # MinMax Normalization
        if self.minmaxnorm:
            stacked_mini = self.repeat_stats_tensor(inputs,self.mini,feature_vect)
            stacked_maxi = self.repeat_stats_tensor(inputs,self.maxi,feature_vect)

            #stacked_mini = torch.stack([self.mini]*self.reshaped_inputs_dim[-1],-1)
            #stacked_maxi = torch.stack([self.maxi]*self.reshaped_inputs_dim[-1],-1)

            if reverse:
                return((inputs*(stacked_maxi-stacked_mini) + stacked_mini))
            else: 
                output_with_nan_and_inf = (inputs - stacked_mini)/(stacked_maxi-stacked_mini)  # Sometimes issues when divided by 0
                return(self.tackle_nan_inf_values(output_with_nan_and_inf))
        # ...
            
        # Z-Standardization 
        elif self.standardize:
            stacked_mean = self.repeat_stats_tensor(inputs,self.mean,feature_vect)
            stacked_std = self.repeat_stats_tensor(inputs,self.std,feature_vect)

            #stacked_mean = torch.stack([self.mean]*self.reshaped_inputs_dim[-1],-1)
            #stacked_std = torch.stack([self.std]*self.reshaped_inputs_dim[-1],-1)

            if reverse:
                return(inputs*stacked_std + stacked_mean)
            else: 
                output_with_nan_and_inf = (inputs - stacked_mean)/(stacked_std)  # Sometimes issues when divided by 0
                return(self.tackle_nan_inf_values(output_with_nan_and_inf)) 
        # ...

        else:
            raise ValueError('Standardization method has not been precised. Set minamxnorm = True or standardize = True')


    def tackle_nan_inf_values(self,output_with_nan_and_inf):
        '''For each channel and each station, we can have some issues when the minimum from Training Set is equal to its Maximum. We then can't normalize the dataset and set the values to 0. '''
        regular_values_set_to_0 =  torch.isinf(output_with_nan_and_inf).sum()
        Values_with_normalization_issues = (torch.isnan(output_with_nan_and_inf) + torch.isinf(output_with_nan_and_inf)).sum()
        if (regular_values_set_to_0 > 0) or (Values_with_normalization_issues>0):
            print('Values with issues: ','{:.3%}'.format(Values_with_normalization_issues.item()/output_with_nan_and_inf.numel() ))
            print('Regular Values that we have to set to 0: ','{:.3%}'.format(regular_values_set_to_0.item()/output_with_nan_and_inf.numel() ))
        if Version(torch.__version__) >= Version("2.0.0"):
            output = torch.nan_to_num(output_with_nan_and_inf,0,0,0)  # Set 0 when devided by maxi - mini = 0 (0 when Nan, 0 when +inf, 0 when -inf
        else:
            output = output_with_nan_and_inf.clone()
            output[torch.isnan(output)] = 0
            output[torch.isinf(output)] = 0
            output[output == float('inf')] = 0
            output[output == float('-inf')] = 0
        return(output)
    

    def unormalize_tensor(self,inputs: torch.Tensor, feature_vect: bool = False):
        unormalized_tensor = self.normalize_tensor(inputs, reverse=True,feature_vect = feature_vect)
        return(unormalized_tensor)

    def normalize_tensor(self,tensor: torch.Tensor,reverse: bool =False,feature_vect: bool = False):
        '''
        args 
        -----
        inputs : n-dimension torch Tensor
        dims :  dimension through which we want to retrieve min/max or mean/std
        minmaxnorm : MinMax-Normalization if True
        standardize: Z-standardization if True 

        Examples:
            inputs = torch.randn(8,4,2,3,6)
            dims = [0,-1,-2]
            minmaxnorm  = True

            output is a Tensor object whose 'tensor' attribute is normalized (or unormalized)
            it returns the minmax-normalization of 'inputs' through dimensions 0,4,3. 
        '''
        normalized_tensor = self.transform(tensor,reverse,feature_vect)
        return(normalized_tensor)