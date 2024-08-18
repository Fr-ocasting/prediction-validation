import torch
import glob 


# Relative path:
import sys 
import os 
current_file_path = os.path.abspath(os.path.dirname(__file__))
parent_dir = os.path.abspath(os.path.join(current_file_path,'..'))
if parent_dir not in sys.path:
    sys.path.insert(0,parent_dir)
# ...

# Personnal inputs:
from dataset import PersonnalInput


def find_positions(applications, file_list):
    positions = []
    for app in applications:
        for idx, file_path in enumerate(file_list):
            if app in file_path:
                positions.append(idx)
    return positions

def replace_heure_d_ete(tensor,start = 572, end = 576):
    values_before = tensor[start-1:start]
    values_after = tensor[end:end+1]

    mean_values = (values_before + values_after) / 2
    mean_values = mean_values.repeat(4,1,1,1,1)

    tensor[start:end,:,:,:,:] = mean_values
    return tensor

def load_netmob_data(dataset,invalid_dates,args,folder_path,columns,
                     trafic_apps = ['Uber', 'Google_Maps','Waze'],
                     music_apps = ['Spotify','Deezer','Apple_Music','Apple_iTunes','SoundCloud'],
                     direct_messenger_apps = ['Telegram','Apple_iMessage','Facebook_Messenger','Snapchat','WhatsApp'],
                     social_networks_apps = ['Twitter', 'Pinterest','Facebook','Instagram'],
                     normalize = True
                     ):
    '''Load NetMob Data:
    outputs:
    --------
    # NetMob Tensor : [T,N,C,H,W]
    # dims : [0,3,4] #[0,-2,-1]  -> dimension for which we want to retrieve stats 
    '''

    selected_apps = trafic_apps # music_apps  # direct_messenger_apps # social_networks_apps
     
    if torch.cuda.is_available():
        apps=  glob.glob(f'{folder_path}NetMob_tensor/[!station]*.pt')
        trafic_pos = find_positions(selected_apps,apps)
        trafic_pos = [2*k for k in trafic_pos] + [2*k+1 for k in trafic_pos]
        assert len(apps) == 136//2 # Tensor.size(1) =nb_mode_transfer x nb_apps =2*68  = 136


        netmob_T = torch.stack([torch.load(f"{folder_path}NetMob_tensor/station_{station}.pt")[:,trafic_pos,:,:] for station in columns])
        '''
        if args.time_slot_limit is not None:
            netmob_T = torch.stack([torch.load(f"{folder_path}NetMob_tensor/station_{station}.pt")[:args.time_slot_limit,trafic_pos,:,:] for station in columns])
        else:
            netmob_T = torch.stack([torch.load(f"{folder_path}NetMob_tensor/station_{station}.pt")[:,trafic_pos,:,:] for station in columns])
        '''
            
        netmob_T = netmob_T.permute(1,0,*range(2, netmob_T.dim()))

        # Replace problematic time-slots:
        netmob_T = replace_heure_d_ete(netmob_T,start = 572, end = 576)

        # Keep only time-slots associated to the dataset:
        netmob_T = netmob_T[dataset.time_slot_limits]

    else:
        netmob_T = torch.randn(dataset.length,40,2,8,8)  # (7400,40,67,22,22)
        print("Load des données NetMob .pt impossible. Création d'un random Tensor")

    print('Init NetMob Dataset: ', netmob_T.size())
    print('Number of Nan Value: ',torch.isnan(netmob_T).sum())
    print('Total Number of Elements: ', netmob_T.numel(),'\n')

    NetMob_ds = PersonnalInput(invalid_dates,args, tensor = netmob_T, dates = dataset.df_dates,
                           time_step_per_hour = dataset.time_step_per_hour,Weeks = args.W, Days = args.D, historical_len = args.H,step_ahead = args.step_ahead,minmaxnorm = True,dims =[0,3,4])
    NetMob_ds.preprocess(args.train_prop,args.valid_prop,args.test_prop,normalize)

    return(NetMob_ds)



def tackle_netmob(dataset,dataset_names,invalid_dates,args,folder_path,columns,vision_model_name,normalize = True):
    if 'netmob' in dataset_names:
        NetMob_ds = load_netmob_data(dataset,invalid_dates,args,folder_path,columns = columns, normalize = normalize )
        C_netmob = NetMob_ds.U_train.size(2)  # [B,N,C,H,W,L]
        L = NetMob_ds.U_train.size(5)

        # FeatureExtractor_ResNetInspired
        if vision_model_name == 'FeatureExtractor_ResNetInspired':
            args_vision = {'model_name':vision_model_name, 'c_in' : C_netmob, 'h_dim': 128, 'out_dim':256}  

        # MinimalFeatureExtractor  
        elif vision_model_name == 'MinimalFeatureExtractor':
            h_dim = 16
            args_vision = {'model_name':vision_model_name, 'c_in' : C_netmob,'h_dim': h_dim, 'out_dim' : L*h_dim//2} 

        # ImageAvgPooling
        elif vision_model_name == 'ImageAvgPooling':
            args_vision = {'model_name':vision_model_name, 'out_dim' : L}
    else:
        NetMob_ds = None
        args_vision = None
    return args_vision,NetMob_ds