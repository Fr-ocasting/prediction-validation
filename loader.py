import torch
from torch.utils.data import Dataset,DataLoader



class calib_prop_splitter(object):
    '''
    args 
    -----
    U : Tensor [B,N,*]  (feature vect)
    U_target : Tensor [B,N,output_dim]  (target)
    contextual_tensor : Dictionnary of all the contextual tensor : {'weather' : weather_tensor, 'calendar' : calendar_tensor, 'netmob': netmob_tensor}  ...
    calib_prop : proportion of calibration set within  training set
    '''

    def __init__(self,U,U_target,contextual_tensors,calib_prop):
        super(calib_prop_splitter,self).__init__()
        self.U = U
        self.U_target = U_target
        self.contextual_tensors = contextual_tensors    
        self.calib_prop = calib_prop

        self.get_attr_limits_proper_calib()
        self.split_proper_calib()


    def get_attr_limits_proper_calib(self):
        ''' Generate the random indices for Calibration set and Proper set '''
        indices = torch.randperm(self.U.size(0)) 
        split = int(self.U.size(0)*self.calib_prop)

        self.indices_cal  = indices[split:]
        self.indices_train = indices[:split]

    def split_proper_calib(self):
        ''' Split the training set in Proper and Calibration set '''
        # Proper Set
        self.proper_set_x = self.U[self.indices_train]
        self.proper_contextual = {name_ds: tensor[self.indices_train] for name_ds,tensor in self.contextual_tensors.items()}
        self.proper_set_y = self.U_target[self.indices_train]

        # Calib Set : 
        self.calib_set_x = self.U[self.indices_cal]
        self.calib_contextual = {name_ds: tensor[self.indices_cal] for name_ds,tensor in self.contextual_tensors.items()}
        self.calib_set_y = self.U_target[self.indices_cal]


class CustomDataLoder(object):
    ## DataLoader Classique pour le moment, puis on verra pour faire de la blocked cross validation
    '''
    args
    -----
    '''
    def __init__(self,U,U_target,contextual_tensor,args, shuffle):
        super().__init__()
        self.dataloader = {}
 
        self.U = U
        self.U_target = U_target
        self.contextual_tensor = contextual_tensor

        self.shuffle = shuffle

        # Hyper-Parameters
        self.calib_prop = args.calib_prop
        self.batch_size = args.batch_size 

        self.num_workers = args.num_workers 
        self.persistent_workers = args.persistent_workers 
        self.pin_memory = args.pin_memory 
        self.prefetch_factor = args.prefetch_factor 
        self.drop_last = args.drop_last 
        # ...

    def call_dataloader(self,training_mode):
        inputs = CustomDataset(self.U,self.U_target,self.contextual_tensor) 
        # sampler = torch.utils.data.distributed.DistributedSampler(train_dataset,num_replicas=idr_torch.size,rank=idr_torch.rank,shuffle= ...)
        self.dataloader = DataLoader(inputs, 
                                    batch_size=(self.U.size(0) if training_mode=='cal' else self.batch_size),  # batch_size
                                    shuffle = self.shuffle,  # if hasattr, then already shuffled 
                                    #sampler=sampler,
                                    num_workers=self.num_workers,
                                    persistent_workers=self.persistent_workers,
                                    pin_memory=self.pin_memory,
                                    prefetch_factor=self.prefetch_factor,
                                    drop_last=self.drop_last
                                    ) 
    


class DictDataLoader(object):

    def __init__(self, train_tuple, valid_tuple, test_tuple,args):
        super(DictDataLoader,self).__init__()

        self.train_tuple = train_tuple
        self.valid_tuple = valid_tuple
        self.test_tuple = test_tuple
        self.calib_prop = args.calib_prop
        self.args = args

    def load_train(self):
        # Load Train:
        U,U_target,contextual_tensors = self.train_tuple
        if self.calib_prop is not None :
            splitter = calib_prop_splitter(U,U_target,contextual_tensors,self.calib_prop)
            train_loader = CustomDataLoder(splitter.proper_set_x,splitter.proper_set_y,splitter.proper_contextual,self.args, shuffle = False)  #already shuffled 
            train_loader.call_dataloader('train')

            calib_loader = CustomDataLoder(splitter.calib_set_x,splitter.calib_set_y,splitter.calib_contextual,self.args, shuffle = False)  #already shuffled 
            calib_loader.call_dataloader('cal')
            return(train_loader,calib_loader)

        else:
            train_loader = CustomDataLoder(U,U_target,contextual_tensors,self.args, shuffle = True) 
            train_loader.call_dataloader('train')
            return(train_loader,None)
        # ...

    def load_valid(self):
        # Load Valid:
        if self.valid_tuple is not None:
            U,U_target,contextual_tensors = self.valid_tuple
            valid_loader = CustomDataLoder(U,U_target,contextual_tensors,self.args, shuffle = False)
            valid_loader.call_dataloader('valid')
        else:
            valid_loader = None
        return(valid_loader)
        # ...

    def load_test(self):
        # Load Test: 
        if self.test_tuple is not None:
            U,U_target,contextual_tensors = self.test_tuple
            test_loader = CustomDataLoder(U,U_target,contextual_tensors,self.args, shuffle = False)
            test_loader.call_dataloader('test')
        else:
            test_loader = None
        return(test_loader)
        # ...

    def get_dictdataloader(self):
        train_loader,calib_loader = self.load_train()
        valid_loader = self.load_valid()
        test_loader = self.load_test()

        if self.calib_prop is not None:
            return dict(train = train_loader.dataloader,
                        valid = valid_loader.dataloader if valid_loader is not None else None,
                        test = test_loader.dataloader if test_loader is not None else None,
                        cal = calib_loader.dataloader
                        )     
        
        else: 
            return dict(train = train_loader.dataloader,
                        valid = valid_loader.dataloader if valid_loader is not None else None,
                        test = test_loader.dataloader if test_loader is not None else None
                        )
        
class CustomDataset(Dataset):
    def __init__(self,X,Y,contextual):
        self.X = X
        self.Y = Y
        if len(contextual) > 0: self.contextual = contextual 
        
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self,idx):
        #T_data = [t[idx] for t in self.T]
        if hasattr(self,'contextual'):
            Contextual_data = tuple(contextual_tensor[idx] for _,contextual_tensor in self.contextual.items())
            return self.X[idx], self.Y[idx], Contextual_data
        else:
            return self.X[idx], self.Y[idx]



if __name__ == '__main__':
    import argparse 

    T = 200   # Period coverage: T time-steps
    N = 40 # Nb Spatial-Unit 
    batch_size = 16
    calib_prop = 0.5

    # Load args 
    parser = argparse.ArgumentParser()
    for key, default in zip(['calib_prop','batch_size','num_workers','persistent_workers','pin_memory','prefetch_factor','drop_last'],[calib_prop,batch_size,1,True,True,2,False]):
        parser.add_argument(f'--{key}', type=type(default), default=default)
    args = parser.parse_args(args=[]) 
    # ...


    train_subset = torch.arange(100)
    valid_subset = torch.arange(110,160)
    test_subset = torch.arange(170,200)

    # Trafic Subway Data:
    L = 8
    subway_X = torch.randn(T,N,L)
    subway_Y = torch.randn(T,N,1)


    # NetMob Data:
    C,H,W = 2,6,6
    netmob = torch.randn(T,N,C,H,W)

    # Calendar-Class:
    calendar = torch.randint(0,6,(T,)) # Calendar class between 0 and 6, Vector of dimension T 

    # Train, Valid, Test split : 
    train_tuple = subway_X[train_subset],subway_Y[train_subset], dict(netmob = netmob[train_subset], calendar = calendar[train_subset])
    valid_tuple = subway_X[valid_subset],subway_Y[valid_subset], dict(netmob = netmob[valid_subset], calendar = calendar[valid_subset])
    test_tuple =  subway_X[test_subset],subway_Y[test_subset], dict(netmob = netmob[test_subset], calendar = calendar[test_subset])

    # Load DictDataLoader: 
    DictDataLoader_object = DictDataLoader(train_tuple, valid_tuple, test_tuple,args)
    dict_dataloader = DictDataLoader_object.get_dictdataloader()

    train_loader = dict_dataloader['train']
    x_b,y_b,contextual_data_b  = next(iter(train_loader)) #x_b,y_b,*contextual_data_b

    print(x_b.size(),y_b.size(), contextual_data_b[0].size(),contextual_data_b[1].size())