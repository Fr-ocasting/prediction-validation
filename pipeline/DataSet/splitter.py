
from pipeline.DataSet.dataset import TensorDataset

class SplitterTrainValidTest(object):
    def __init__(self,data,
                 train_indices = None , valid_indices = None, test_indices = None,
                 first_train = None, last_train = None, first_valid = None, last_valid = None, first_test = None, last_test = None, 
                 minmaxnorm = False,standardize = False):
        super(SplitterTrainValidTest,self).__init__()
        self.data = data
        self.minmaxnorm = minmaxnorm
        self.standardize = standardize

        if train_indices is not None : self.train_indices = train_indices
        if valid_indices is not None : self.valid_indices = valid_indices
        if test_indices is not None : self.test_indices = test_indices

        if first_train is not None : self.first_train = first_train
        if last_train is not None : self.last_train = last_train

        if first_valid is not None : self.first_valid = first_valid
        if last_valid is not None : self.last_valid = last_valid

        if first_test is not None : self.first_test = first_test
        if last_test is not None : self.last_test = last_test

        self.split_data()

    def split_data(self):
        # Split Data within 3 groups:
        if hasattr(self,'train_indices'):
            self.data_train = self.data[self.train_indices] 
            self.data_valid = self.data[self.valid_indices] if self.valid_indices is not None else None
            self.data_test = self.data[self.test_indices] if self.test_indices is not None else None
        elif hasattr(self,'first_train'):
            self.data_train = self.data[self.first_train:self.last_train]
            self.data_valid = self.data[self.first_valid:self.last_valid] if hasattr(self,'first_valid') else None
            self.data_test = self.data[self.first_test:self.last_test]   if hasattr(self,'first_test') else None
        else: 
            raise ValueError("Neither 'train_indices' nor 'first_train' attribute has been designed ")
        

    def split_normalize_tensor_datasets(self,normalizer = None):
        '''Load TensorDataset (train_dataset) object from data_train.
        Define TensorDataset object from valid (valid_dataset) and test (test_dataset). 
        Associate statistics from train dataset to valid and test dataset
        Normalize them according to their statistics 
        '''
        train_dataset = TensorDataset(self.data_train, normalized = False, normalizer=normalizer)


        if hasattr(self,'first_valid'): 
            valid_dataset = TensorDataset(self.data_valid, normalized = False, normalizer=normalizer)
        else:
            valid_dataset = None

        if hasattr(self,'first_test'):  
            test_dataset = TensorDataset(self.data_test,normalized = False, normalizer=normalizer)
        else : 
            test_dataset = None

        if normalizer is not None:
            train_dataset.normalize(feature_vect = True)
            if hasattr(self,'first_valid'): 
                valid_dataset.normalize(feature_vect = True)

            if hasattr(self,'first_test'): 
                test_dataset.normalize(feature_vect = True)

        
        return(train_dataset,valid_dataset,test_dataset)

