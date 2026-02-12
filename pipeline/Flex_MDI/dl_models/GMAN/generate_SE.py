
import numpy as np
import networkx as nx
import os
import sys
import torch 
import pickle
current_file_path = os.path.abspath(os.path.dirname(__file__))
ROOT = os.path.abspath(os.path.join(current_file_path,'..'))
if ROOT not in sys.path:
    sys.path.insert(0,ROOT)
from constants.paths import ABS_PATH_PACKAGE,FOLDER_PATH
import pipeline.Flex_MDI.dl_models.GMAN.node2vec as node2vec
import pandas as pd
p = 2
q = 1
num_walks = 100
walk_length = 80
window_size = 10
iter_word2vec = 1000






def read_graph(edgelist):
    G = nx.read_edgelist(
        edgelist, nodetype=int, data=(('weight',float),),
        create_using=nx.DiGraph())
    return G


    
benchmark_datasets = ['PEMS04','PEMS07','PEMS08','PEMS03','PEMS-BAY', 'PeMS04_flow','PeMS07_flow','PeMS08_flow','PeMS03_flow','PeMS-BAY_flow']

# Added function to load or generate SE using node2vec
def load_SE_GMAN(dataset,args):
    save_folder_path = f'{FOLDER_PATH}/{dataset.target_data}/adj'
    SE_file = f'{save_folder_path}/SE_GMAN_nHead{args.num_heads}_dimHead{args.head_dim}_{args.adj_type}.txt'
    args.SE_file = SE_file

    if not os.path.exists(SE_file):
        from gensim.models import Word2Vec

        def learn_embeddings(walks, dimensions, output_file):
            walks = [list(map(str, walk)) for walk in walks]
            model = Word2Vec(
                walks, vector_size = dimensions, window = 10, min_count=0, sg=1,
                workers = 8, epochs = iter_word2vec)
            model.wv.save_word2vec_format(output_file)
            return

        if args.adj_type in ['dist','adj']:
            if args.target_data in benchmark_datasets:
                is_directed = True
            elif args.target_data in ['subway_in','subway_out','bike_in','bike_out']:
                is_directed = False
            else:
                raise ValueError(f"Dataset {args.target_data} not recognized for adj_type {args.adj_type}.")
        elif args.adj_type == 'corr':
            is_directed = False
        else:
            raise ValueError(f"Adj type {args.adj_type} not recognized.")



        Adj_file = f'{save_folder_path}/{args.adj_type}'
        
        if os.path.exists(f'{Adj_file}.csv'):
            df = pd.read_csv(f'{Adj_file}.csv',index_col = 0)
            nx_G = nx.from_pandas_adjacency(df, create_using=nx.DiGraph())

        elif 'pkl' in dataset.adj_mx_path  and os.path.exists(dataset.adj_mx_path):
            with open(dataset.adj_mx_path, 'rb') as f:
                nx_G = pickle.load(f)
                nx_G = nx.from_pandas_adjacency(pd.DataFrame(nx_G), create_using=nx.DiGraph())

        elif 'txt' in dataset.adj_mx_path  and os.path.exists(dataset.adj_mx_path):
            nx_G = read_graph(dataset.adj_mx_path)  

        elif os.path.exists(f'{Adj_file}.txt'):
            Adj_file = f'{Adj_file}.txt'
            nx_G = read_graph(Adj_file)
        else:
            raise ValueError(f"Not any Adj file found in {os.listdir(save_folder_path)}.")
        print('Graph loaded',len(nx_G.nodes()),len(nx_G.edges()))
        G = node2vec.Graph(nx_G, is_directed, p, q)
        G.preprocess_transition_probs()
        walks = G.simulate_walks(num_walks, walk_length)
        dimensions = args.num_heads * args.head_dim
        print('walks generated',len(walks))
        print('dimensions',dimensions)
        print('SE_file',SE_file)
        learn_embeddings(walks, dimensions, SE_file)
    
    # Load SE
    f = open(args.SE_file, mode = 'r')
    print('Loading Spatial Embedding from',args.SE_file)
    lines = f.readlines()
    temp = lines[0].split(' ')
    num_vertex, dims = int(temp[0]), int(temp[1])
    SE = torch.zeros((num_vertex, dims), dtype=torch.float32)
    for k,line in enumerate(lines[1 :]):
        temp = line.split(' ')
        try: 
            index = int(temp[0])
        except:
            index = k
        SE[index] = torch.tensor([float(ch) for ch in temp[1:]])

    return SE








