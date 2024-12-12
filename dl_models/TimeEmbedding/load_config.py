import argparse

parser = argparse.ArgumentParser(description='TimeEmbedding')

parser.add_argument('--embedding_dim', type=int, default=3, choices=[3,4,8],
                    help='Embedding Dimension')

parser.add_argument('--multi_embedding', type=bool, default=False, choices=[True,False],
                    help='If True then the embedding vector will be different for each spatial-units')

parser.add_argument('--calendar_class', type=int, default=2, choices=[0,1,2,3],
                    help='Is the identifiant of the type of clustering done on contextual calendar information. 0: no embedding ; 1: personnal clustering ; 2: No clustering; 3: clustering per hour')
parser.add_argument('--TE_transfer', type=bool, default=False, choices=[True,False],
                    help='If True then use a pre-trained embedding model.')

parser.add_argument('--position', type=str, default='input', choices=['input','output'],
                    help='Position of time_embedding module : before or after the core model.')

parser.add_argument('--type_calendar', type=str, default='tuple', choices=['tuple','unique_long_embedding'],
                    help='if unique_long_embedding then embedding for a single long vector. If tuple then embedding of each element of the tuple.')

parser.add_argument('--specific_lr', type=bool, default=False, choices=[True,False],
                    help='If True then design a specific lr (more significant) for the embedding module, to force the training.')

parser.add_argument('--concatenation_late', type=bool, default=True, 
                    choices = [True,False],
                    help='If True then concatenation of extracted feature just before the output module of the backbone model')

parser.add_argument('--concatenation_early', type=bool, default=False, 
                    choices = [True,False],
                    help='If True then concatenation of extracted feature with the inputs at the begining of the backbone model.')
                    
                       
args = parser.parse_args(args=[])
