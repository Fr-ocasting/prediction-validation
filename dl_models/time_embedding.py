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
    def __init__(self,nb_words_embedding,embedding_dim,type_calendar,mapping_tensor,embedding_with_dense_layer = True, n_embedding = 1):
        super(TimeEmbedding, self).__init__()
        self.nb_words = nb_words_embedding
        self.embedding_with_dense_layer = embedding_with_dense_layer
        self.type_calendar = type_calendar
        self.mapping_tensor = mapping_tensor
        self.n_embedding = n_embedding

        if self.type_calendar == 'tuple':
            nb_embeddings = mapping_tensor.size(1)
            self.dic_sizes = [mapping_tensor[:,i].max().item() +1 for i in range(nb_embeddings) if mapping_tensor[:,i].max().item() > 0]
            self.Embedding_dims = [max(int(dic_size/2), 1) for dic_size in self.dic_sizes]


            # [Emb_hour, Emb_jour, Emb_minute]
            # For each embedding: 
            # embedding = nn.Linear(dic_size,emb_dim*n_embedding)

            # Example : 
            # Emb_hour: 
            # dic_size = 7
            # emb_dim = 3
            # n_embedding = 40
            # [3,3,3,3,3...... 40 fois ... 3,3,3]

            self.embedding = nn.ModuleList([nn.Linear(dic_size,emb_dim*n_embedding) for dic_size,emb_dim in zip(self.dic_sizes,self.Embedding_dims)])
            #self.output1 = nn.Linear(sum(Embedding_dims),embedding_dim*2)
            self.output1 = nn.Linear(sum(self.Embedding_dims),int(sum(self.Embedding_dims)/2))
            #self.output2 = nn.Linear(embedding_dim*2,embedding_dim) 
            self.output2 = nn.Linear(int(sum(self.Embedding_dims)/2),embedding_dim) 
            self.relu = nn.ReLU()

        elif self.type_calendar == 'unique_long_embedding' : 
            if embedding_with_dense_layer:
                self.embedding = nn.Linear(self.nb_words,embedding_dim)
            else: 
                self.embedding = nn.Embedding(self.nb_words,embedding_dim)

        else:
            raise NotImplementedError(f"args.type_calendar '{self.type_calendar}' has not been implemented")

    def forward(self,elt): 
        if self.type_calendar == 'tuple':
            elt = self.mapping_tensor[elt.long()].to(elt)
            concat_z = torch.Tensor().to(elt)
            for i,emb_layer in enumerate(self.embedding):
                if len(elt.size()) == 1:    # When there is no batch, but just a  single element
                    elt = elt.unsqueeze(0) 
                elt_i = elt[:,i].long().squeeze()
                one_hot_encodding_matrix = nn.functional.one_hot(elt_i,num_classes =self.dic_sizes[i]).to(elt).float()
                emb_vector = emb_layer(one_hot_encodding_matrix)
                emb_vector = emb_vector.reshape(elt.size(0),self.n_embedding,self.Embedding_dims[i])
                concat_z = torch.cat([concat_z,emb_vector],dim=-1) # [B,N_stations,embedding_dim*len(self.dic_sizes)]

 
            z = self.output2(self.relu(self.output1(concat_z)))  # [B, N_stations, Z]

        if self.type_calendar == 'unique_long_embedding':
            if self.embedding_with_dense_layer:
                one_hot_encodding_matrix = nn.functional.one_hot(elt.long().squeeze(),num_classes =self.nb_words).to(elt).float()
                z = self.embedding(one_hot_encodding_matrix)
            else: 
                z = self.embedding(elt)
        return(z)
    


class TE_adder(nn.Module):
    def __init__(self,model,args,args_embedding = None,dic_class2rpz = None):
        super(TE_adder, self).__init__()
        self.model = model
        self.args = args
        if args_embedding is not None:
            #mapping_tensor = torch.tensor([(week[0], time[0][0], time[0][1], bank_holiday) for _, (week, time, bank_holiday) in sorted(dic_class2rpz.items())]).to(args.device)
            mapping_tensor = torch.tensor([(week[0], time[0][0], time[0][1]) for _, (week, time) in sorted(dic_class2rpz.items())]).to(args.device)
            self.multi_embedding = args.multi_embedding
            self.Tembedding = TimeEmbedding(args_embedding.nb_words_embedding,args_embedding.embedding_dim,args.type_calendar,mapping_tensor, n_embedding= args.num_nodes if self.multi_embedding else 1)
            self.Tembedding_position = args_embedding.position
            self.N_repeat = 1 if self.multi_embedding else args.num_nodes
        
    
    #def __getattr__(self, name):
    #    # Redirige tous les appels d'attributs non trouvés vers l'attribut 'self.model'
    #    return getattr(self.model, name)

    def forward(self,x, time_elt = None):
        if len(x.size())<4:
            x = x.unsqueeze(1)
        B,C,N,L = x.size()
    
        if time_elt is not None:
            if self.Tembedding_position == 'input':
                time_elt = self.Tembedding(time_elt)   # [B,1] -> [B,embedding_dim*N_station]  
                if not(self.multi_embedding):
                    time_elt = time_elt.repeat(1,self.N_repeat*C,1)
                time_elt = time_elt.reshape(B,C,N,-1)   # [B,N_station*embedding_dim] -> [B,C,embedding_dim,N]
                x = torch.cat([x,time_elt],dim = -1)

        if self.args.model_name == 'DCRNN':
            x = self.model(x,labels=None)
        else:
            x = self.model(x)         

        return(x)
        
            
        