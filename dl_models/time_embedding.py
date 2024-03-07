import torch
import torch.nn as nn

class TimeEmbedding(nn.Module):
    def __init__(self,Embedding_dims,C_outs,c_out_embedding = 6):
        super(TimeEmbedding, self).__init__()
        
        self.relu = nn.ReLU()
        self.flatten = nn.Flatten()
        
        L_embedding = []
        L_dense = []
        for embedding_dim,C_out in zip(Embedding_dims,C_outs):
            L_embedding.append(nn.Embedding(embedding_dim,C_out))
            L_dense.append(nn.Linear(C_out,2*C_out))
        
        self.Denses = L_dense
        self.Embeddings = L_embedding  
        self.dense1 =  nn.Linear(sum([emb_dim*(c_out*2) for emb_dim,c_out in zip (Embedding_dims,C_outs)]),256)
        self.dense2 =  nn.Linear(256,32)
        self.dense3 =  nn.Linear(32,c_out_embedding)
        
    def forward(self,dic_encodded_time):
        flattened_list = []
        for t,emb,dense in zip(list(dic_encodded_time.values()),self.Embeddings, self.Denses):
            t = torch.LongTensor(t).reshape(1,len(t))
            t = emb(t)
            t = dense(t)
            t = self.relu(t)
            t = self.flatten(t)
            flattened_list.append(t)
        z_vector = torch.cat(flattened_list,dim=1)
        z = self.dense1(z_vector)
        z = self.relu(self.dense2(z))
        z = self.relu(self.dense3(z))          
        return(z)