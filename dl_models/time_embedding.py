import torch
import torch.nn as nn


def elt2word_indx(elt,Encoded_dims):
    # Eventuellement, on peut faire -1 sur chaque 'elt'. 
        # Car minute de 0 à 3
        # heure de 0 à 23
        # Jour de 0 à 6
    Encoded_dims_tensor = torch.tensor([1] + Encoded_dims[:-1])
    product_dimension = Encoded_dims_tensor.cumprod(dim=0)
    bijection = torch.dot(torch.Tensor(elt).float(),product_dimension.float())  
    word_indx = torch.cuda.LongTensor([bijection.item()]) if torch.cuda.is_available() else torch.LongTensor([bijection.item()])
    return(word_indx)

class TimeEmbedding(nn.Module):
    def __init__(self,Encoded_dims,embedding_dim,embedding_with_dense_layer = True):
        super(TimeEmbedding, self).__init__()
        self.nb_words = int(torch.prod(torch.Tensor(Encoded_dims)).item())
        self.embedding_with_dense_layer = embedding_with_dense_layer

        if embedding_with_dense_layer:
            self.embedding = nn.Linear(self.nb_words,embedding_dim)
        else: 
            self.embedding = nn.Embedding(self.nb_words,embedding_dim)
        self.Encoded_dims = Encoded_dims

    def forward(self,elt): 
        if type(elt) == tuple: 
            word_indx = elt2word_indx(elt,self.Encoded_dims)
        else:
            word_indx = elt

        if self.embedding_with_dense_layer:
            one_hot_encodding_matrix = nn.functional.one_hot(word_indx.long().squeeze(),num_classes =self.nb_words).float()
            z = self.embedding(one_hot_encodding_matrix)
        else: 
            z = self.embedding(word_indx)
        return(z)

if False : 
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
            z = self.relu(self.dense1(z_vector))
            z = self.relu(self.dense2(z))
            z = self.dense3(z)         
            return(z)