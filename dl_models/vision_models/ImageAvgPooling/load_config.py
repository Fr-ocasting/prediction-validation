import argparse

'''
Have to define :
>> 'args' which represent all the hyperparamer linked to the feature extractor model
>> get_config() which return 'args', a NameSpace containing the hyperparameter of feature extractor model updated thank to parameters from the backbone model and DATA_TO_PREDICT
'''

parser = argparse.ArgumentParser(description='ImageAvgPooling')
parser.add_argument('--out_dim', type=int, default=None,
                    help='out_put Temporal dimension. Is not supposed to be changed from the input temporal dimension. Have to be initialized here, but set later in the code')

args = parser.parse_args(args=[])



def get_config(H,W,L):
    args.out_dim = L
    return(args)
