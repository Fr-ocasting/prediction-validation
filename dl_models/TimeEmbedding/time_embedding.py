import torch
import torch.nn as nn

import sys 
import os 
# Get Parent folder : 
current_path = os.getcwd()
parent_dir = os.path.abspath(os.path.join(current_path, '..','..'))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from dl_models.vision_models.VariableSelectionNetwork.VariableSelectionNetwork import GRN

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
    def __init__(self,args_embedding,n_vertex,x_input_size,dropout): #n_vertex
        super(TimeEmbedding, self).__init__()
        self.embedding_dim = args_embedding.embedding_dim
        self.dic_sizes = args_embedding.dic_sizes
        self.embedding_dim_calendar_units = args_embedding.embedding_dim_calendar_units
        self.embedding = nn.ModuleList([nn.Embedding(dic_size,emb_dim) for dic_size,emb_dim in zip(self.dic_sizes,self.embedding_dim_calendar_units)])
        
        input_dim = sum(self.embedding_dim_calendar_units)
        self.n_vertex = n_vertex
        if args_embedding.out_h_dim is not None: 
            out_h_dim = args_embedding.out_h_dim
        else:
            out_h_dim = input_dim//2

        if False: 
            self.output1 = nn.Linear(input_dim,out_h_dim)

            if args_embedding.multi_embedding:
                self.output2 = nn.Linear(out_h_dim,n_vertex*args_embedding.embedding_dim)
            else:
                self.output2 = nn.Linear(out_h_dim,args_embedding.embedding_dim)        

            self.relu = nn.ReLU()
        if True: 
            self.gru = GRN(x_input_size,out_h_dim,args_embedding.embedding_dim,input_dim,dropout)

    def forward(self,x,time_elt): 
        '''
        x : 4-th order Tensor [B,C,N,L]
        elt : List of One-Hot-Encoded Vector (2-th order Tensor [B,L_calendar_i]) 
        '''
        z = torch.cat([emb(time_elt[k].argmax(dim=1).long()) for k,emb in enumerate(self.embedding)],-1)

        if False:
            z = self.output1(z)
            z = self.output2(self.relu(z))
        if True: 
            # PROBLEME -> Même information calendaire commune à toute les stations. 
            # __ z [B,z] ) -> [B,1,N,z]
            z = z.unsqueeze(1)
            z = z.repeat(1,self.n_vertex,1)
            z = z.unsqueeze(1)
            # __

            #print('calendar_embedding size: ',z.size())
            z = self.gru(x,z)
            #print('x size après prise en compte calendar: ',z.size())
        return(z)
    

class TE_module(nn.Module):
    def __init__(self,args):
        super(TE_module, self).__init__()
        args_embedding =  args.args_embedding
        #self.multi_embedding = args_embedding.multi_embedding
        self.Tembedding = TimeEmbedding(args_embedding,args.n_vertex,args.L,args.dropout) #args.n_vertex
        #self.N_repeat = 1 if self.multi_embedding else args.n_vertex
        self.C = args.C
        self.n_vertex = args.n_vertex
        self.multi_embedding = args_embedding.multi_embedding
        #self.n_vertex = args.n_vertex

    def forward(self,x,time_elt):
        """
        args:
        time_elt: list of One-Hot Encoded torch.Tensor. Each element of the list correspond to a calendar attribute (Weekdays, Hour, Minutes...)
        >>> time_elt[i] is a torch.Tensor of dimension [B,P_i], where B is a batch_size, P_i is the number of possible label of the i-th calendar attribute.
        >>> if i correspond to 'weekdays', then P_i = 7, and 'a sample of Tuesday' will be represented by [0,1,0,0,0,0,0]
        """
        mini_batch_size = time_elt[0].size(0)
        time_elt = self.Tembedding(x,time_elt)   # [B,1] -> [B,N_station,embedding_dim]  or [B,embedding_dim] 
        if not(self.multi_embedding):
            time_elt = time_elt.repeat(1,self.n_vertex*self.C,1)
        time_elt = time_elt.reshape(mini_batch_size,self.C,self.n_vertex,-1)   # [B,N_station*embedding_dim] -> [B,C,embedding_dim,N]
        return(time_elt)

# ======     
# ======
# Debug Calendar : 
def debug(elt,z):
    def f_cal2str(calendar_type,x):
        if calendar_type == 0: #dayofweek
            dayofweek = ['lundi','mardi','mercredi','jeudi','vendredi','samedi','dimanche']
            return(dayofweek[x])
        elif calendar_type == 1: #'hour'
            return(f" {x}:") 
        elif calendar_type == 2: # 'minute'
            return(f"{15*x} ")
        elif calendar_type == 3: #'bank_holidays
            return(f"bh{x}")
        elif calendar_type == 4: #'school_holidays
            return(f"sh{x}")
        elif calendar_type == 5: #'remaining_holidays
            return(f"rh{x}")
        else:
            return(f'calendar type {calendar_type} as not been designed')
    #print('elt: ',elt)
    batch_size = elt[0].size(0)
    time_step_ohe = []
    for b in range(batch_size):
        total_ohe = [calendar_tensor[b].argmax(0) for calendar_tensor in elt]
        time_step_ohe.append(total_ohe)
    
    for b in range(batch_size):
        print('\n',''.join([f_cal2str(calendar_type,total_ohe.item()) for calendar_type,total_ohe in enumerate(time_step_ohe[b])]))
        #print([f_cal2str(calendar_type,elt[calendar_type][b]) for calendar_type in range(len(time_step_ohe))])
        print('embedding: ',z[b,:])
# ======
# ======       

"""
class TimeEmbedding(nn.Module):
    def __init__(self,nb_words,embedding_dim,type_calendar,mapping_tensor,calendar_class,embedding_with_dense_layer = True, n_embedding = 1):
        super(TimeEmbedding, self).__init__()
        self.nb_words = nb_words
        self.embedding_with_dense_layer = embedding_with_dense_layer
        self.type_calendar = type_calendar
        self.mapping_tensor = mapping_tensor
        self.n_embedding = n_embedding
        self.calendar_class = calendar_class

        if self.type_calendar == 'tuple':
            nb_embeddings = mapping_tensor.size(1)  # = size tuple = 3  (weekday, hour, minute)

            # self.dic_sizes = [nb_weekdays rpz, nb_hours rpz, nb_minutes rpz] within mappin tensor (i.e represenntation of the class) 
            self.dic_sizes = [mapping_tensor[:,i].max().item() +1 for i in range(nb_embeddings) if mapping_tensor[:,i].max().item() > 0]
            #self.dic_sizes = [mapping_tensor[:,i].unique().numel() for i in range(nb_embeddings) if mapping_tensor[:,i].max().item() > 0]

            # self.Embedding_dims ~ self.dic_sizes/2
            self.Embedding_dims = [max(int(dic_size/2), 1) for dic_size in self.dic_sizes]


            # [Emb_hour, Emb_jour, Emb_minute]
            # For each embedding: 
            # embedding = nn.Linear(dic_size,emb_dim*n_embedding)

            # Example : 
            # dic_size = 7
            # emb_dim = 3
            # if n_embedding = 40:
            # >>>> return a vector of size [40*3] which is unstack to [3,3, ... ,3] (shape = [40])
            # >>>> each of the 7 words in dic_size has is own representation on a 3D vector.

            self.embedding = nn.ModuleList([nn.Linear(dic_size,emb_dim*n_embedding,bias=False) for dic_size,emb_dim in zip(self.dic_sizes,self.Embedding_dims)])
            #self.output1 = nn.Linear(sum(Embedding_dims),embedding_dim*2)
            ''' A modifier ici, à priori ne peut pas prendre en compte de Multi-Embedding'''
            self.output1 = nn.Linear(sum(self.Embedding_dims),int(sum(self.Embedding_dims)/2)) 
            #self.output2 = nn.Linear(embedding_dim*2,embedding_dim) 
            self.output2 = nn.Linear(int(sum(self.Embedding_dims)/2),embedding_dim) 
            self.relu = nn.ReLU()

        elif self.type_calendar == 'unique_long_embedding' : 
            if embedding_with_dense_layer:
                self.embedding = nn.Linear(self.nb_words[calendar_class],embedding_dim)
            else: 
                self.embedding = nn.Embedding(self.nb_words[calendar_class],embedding_dim)

        else:
            raise NotImplementedError(f"args.type_calendar '{self.type_calendar}' has not been implemented")

    def forward(self,elt): 
        if self.type_calendar == 'tuple':
            #print('elt: ',elt)
            elt = self.mapping_tensor[elt.long()].to(elt)
            #print('mappped elt: ',elt)
            concat_z = torch.Tensor().to(elt)
            for i,emb_layer in enumerate(self.embedding):
                if len(elt.size()) == 1:    # When there is no batch, but just a  single element
                    elt = elt.unsqueeze(0) 
                elt_i = elt[:,i].long().squeeze()
                one_hot_encodding_matrix = nn.functional.one_hot(elt_i,num_classes =self.dic_sizes[i]).to(elt).float()
                emb_vector = emb_layer(one_hot_encodding_matrix)
                emb_vector = emb_vector.reshape(elt.size(0),self.n_embedding,self.Embedding_dims[i])
                concat_z = torch.cat([concat_z,emb_vector],dim=-1) # [B,N_stations,embedding_dim*len(self.dic_sizes)]

            #    print('concat_z size: ',concat_z.size())
            #print('embedded vector before FC layer: ',concat_z.size())
            z = self.relu(self.output1(concat_z))
            #print('embedded vector after FC1: ',z.size())
            z = self.output2(z)  # [B, N_stations, Z]
            #print('embedded vector after FC2: ',z.size())


        if self.type_calendar == 'unique_long_embedding':
            if self.embedding_with_dense_layer:
                one_hot_encodding_matrix = nn.functional.one_hot(elt.long().squeeze(),num_classes =self.nb_words[self.calendar_class]).to(elt).float()
                z = self.embedding(one_hot_encodding_matrix)
            else: 
                z = self.embedding(elt)
        return(z)
    

class TE_module(nn.Module):
    def __init__(self,args):
        super(TE_module, self).__init__()
        args_embedding =  args.args_embedding

        # size of mapping_tensor = number of class * 3.   3 = tuple size (weekday,hour,minute)
        mapping_tensor = torch.tensor([(week[0], time[0][0], time[0][1]) for _, (week, time) in sorted(args_embedding.dic_class2rpz[args_embedding.calendar_class].items())]).to(args.device)

        self.multi_embedding = args_embedding.multi_embedding
        self.Tembedding = TimeEmbedding(args_embedding.nb_words_embedding,args_embedding.embedding_dim,args_embedding.type_calendar,mapping_tensor,calendar_class = args_embedding.calendar_class, n_embedding= args.n_vertex if self.multi_embedding else 1)
        self.Tembedding_position = args_embedding.position
        self.N_repeat = 1 if self.multi_embedding else args.n_vertex
        self.C = args.C
        self.n_vertex = args.n_vertex



    def forward(self,time_elt):
        mini_batch_size = time_elt.size(0)
        if self.Tembedding_position == 'input':
            time_elt = self.Tembedding(time_elt)   # [B,1] -> [B,embedding_dim*N_station]  
            if not(self.multi_embedding):
                time_elt = time_elt.repeat(1,self.N_repeat*self.C,1)
            time_elt = time_elt.reshape(mini_batch_size,self.C,self.n_vertex,-1)   # [B,N_station*embedding_dim] -> [B,C,embedding_dim,N]

        else:
            raise NotImplementedError(f'Position {self.Tembedding_position} has not been implemented')

        return(time_elt)
"""