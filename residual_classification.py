# Classification / Regression
import torch.nn as nn
import torch
import pandas as pd 
import numpy as np

from sklearn.experimental import enable_halving_search_cv 
from sklearn.model_selection import HalvingGridSearchCV

def load_dataset_for_classifier_for_ONE_single_station(trainer,split_prop,bank_holidays,school_holidays,station = 0,dataloader_set = 'cal',
                                                       add_bank_holidays = True,
                                                       add_scholar_holidays = True,
                                                        hour = True,
                                                        weekday = True,
                                                        minutes = True):
    # Load DataSets
    (X_cal,Y_cal,T_cal,T_cal_exact_dates,res_lower,res_upper) = load_inputs_preds_and_init_PI(trainer,calibration_calendar_class=3,dataloader_set = dataloader_set)
    df_dummies = load_df_dummies(T_cal_exact_dates,bank_holidays,school_holidays, add_bank_holidays, add_scholar_holidays, hour = hour, weekday = weekday,minutes = minutes)
    df_classification = load_df_classification(df_dummies,res_lower,res_upper,station = station)
    (X_train,X_valid,Y_train_lower,Y_valid_lower,Y_train_upper,Y_valid_upper) = get_train_valid_lower_upper(df_classification,split_prop)

    datasets = [(X_train,Y_train_lower),(X_train,Y_train_upper)]
    datasets_valid = [(X_valid,Y_valid_lower),(X_valid,Y_valid_upper)]
    # ...
    return(datasets,datasets_valid,df_classification,Y_train_lower)


def labelise_residual(Y_train_lower,fraction_maxi = 2,fraction_std_step = 4):
    # Restrain to no 'outlier'
    quantile95 = Y_train_lower.quantile(0.95)
    positif = Y_train_lower > 0
    no_outliers = Y_train_lower[(Y_train_lower < quantile95)& positif]

    # Stats
    std_no_outliers = no_outliers.std()  
    maxi = no_outliers.max()
    global_max = Y_train_lower.max()
    step = std_no_outliers/fraction_std_step  #1, 1.5, 2 ...


    label_range = [-np.inf] + list(np.arange(0,maxi/fraction_maxi,step)) + [np.inf]
    labels = np.arange(len(label_range)-1)
    return(labels,step)



def get_data(trainer,calibration_calendar_class,dataloader_set = 'cal'):
    data = [[x_b,y_b,t_b[trainer.args.calendar_class],t_b[calibration_calendar_class]] for  x_b,y_b,*t_b in trainer.dataloader[dataloader_set]]
    X_cal,Y_cal,T_pred,T_cal = torch.cat([x_b for [x_b,_,_,_] in data]).to(trainer.args.device),torch.cat([y_b for [_,y_b,_,_] in data]).to(trainer.args.device),torch.cat([t_pred for [_,_,t_pred,_] in data]).to(trainer.args.device),torch.cat([t_cal for [_,_,_,t_cal] in data]).to(trainer.args.device)
    return(X_cal,Y_cal,T_pred,T_cal)


def forward_pass(trainer,X_cal,T_pred):
    trainer.model.eval()
    with torch.no_grad():
        # Forward Pass: 
        if trainer.args_embedding : 
            preds = trainer.model(X_cal,T_pred.long())
        else:
            preds = trainer.model(X_cal) 

        if len(preds.size()) == 2:
            preds = preds.unsqueeze(1)
        # ...
        return(preds)

def get_init_PI_bands(trainer,preds,Y_cal):
    if preds.size(-1) == 2:
        lower_q,upper_q = preds[...,0].unsqueeze(-1),preds[...,1].unsqueeze(-1)   # The Model return ^q_l and ^q_u associated to x_b

    elif preds.size(-1) == 1:
        lower_q,upper_q = preds,preds 
    else:
        raise ValueError(f"Shape of model's prediction: {preds.size()}. Last dimension should be 1 or 2.")
    # ...
    lower_q, upper_q = trainer.dataset.unormalize_tensor(lower_q,device = trainer.args.device),trainer.dataset.unormalize_tensor(upper_q,device = trainer.args.device)
    Y_cal = trainer.dataset.unormalize_tensor(Y_cal,device = trainer.args.device)

    return(Y_cal,lower_q,upper_q)

def is_holidays(date_ts,hoilidays):
    return(date_ts in hoilidays)

def load_inputs_preds_and_init_PI(trainer,calibration_calendar_class=3,dataloader_set = 'cal'):
    # Load Data
    X_cal,Y_cal,T_pred,T_cal = get_data(trainer,calibration_calendar_class,dataloader_set)
    if dataloader_set == 'cal':
        T_cal_exact_dates = trainer.dataset.df_verif_train.iloc[trainer.dataset.indices_cal,-1]
    elif dataloader_set == 'valid':
         T_cal_exact_dates = trainer.dataset.df_verif_valid.iloc[:,-1]
    elif dataloader_set == 'train':
         T_cal_exact_dates = trainer.dataset.df_verif_train.iloc[trainer.dataset.indices_train,-1]
    elif dataloader_set == 'test':
         T_cal_exact_dates = trainer.dataset.df_verif_test.iloc[:,-1]
    else:
        raise ValueError(f"dataloader_set {dataloader_set} doesn't exists. Please use 'cal','train','valid' or 'test'. ")

         
    preds = forward_pass(trainer,X_cal,T_pred)

    # Load Initial Lower and upper band 
    Y_cal,lower_q,upper_q = get_init_PI_bands(trainer,preds,Y_cal)

    # Get Residuals 
    res_lower = Y_cal - lower_q
    res_upper = upper_q - Y_cal

    return(X_cal,Y_cal,T_cal,T_cal_exact_dates,res_lower,res_upper)

def load_df_dummies(T_cal_exact_dates,bank_holidays,school_holidays, add_bank_holidays = True, add_scholar_holidays = True,hour =True,weekday = True,minutes = True):
    df_T = pd.DataFrame(T_cal_exact_dates).rename(columns={'t+0':'date'})
    if hour:
        df_T['hour'] = df_T.date.dt.hour.astype(str)
    if weekday:
        df_T['weekday'] = df_T.date.dt.weekday.astype(str)
    if minutes: 
        df_T['minutes'] = (df_T.date.dt.minute/15).astype(int).astype(str)
    if add_bank_holidays:
        df_T['bank_holidays'] = df_T.date.apply(lambda date_ts : is_holidays(date_ts,bank_holidays)).astype(int)
    if add_scholar_holidays:
        df_T['school_holidays'] = df_T.date.apply(lambda date_ts : is_holidays(date_ts,school_holidays)).astype(int)
    df_dummies = pd.get_dummies(df_T)
    return(df_dummies)

def load_df_classification(df_dummies,res_lower,res_upper,station = 0):
    df_classification = df_dummies.drop(columns = ['date'])
    df_classification['res_lower'] = res_lower[:,station,0]
    df_classification['res_upper'] = res_upper[:,station,0]
    return(df_classification)

class MLP(nn.Module):
    def __init__(self,c_in,h_dim1,h_dim2,c_out):
        super().__init__()
        self.hidden1 = nn.Linear(c_in,h_dim1)
        self.hidden2 = nn.Linear(h_dim1,h_dim2)
        self.relu = nn.ReLU()
        self.output = nn.Linear(h_dim2,c_out)
    
    def forward(self,x):
        x = self.hidden1(x)
        x = self.relu(x)
        x = self.relu(self.hidden2(x))
        x = self.output(x)
        return(x)

def train_and_valid(model,epochs,dataloader,loss_function,optimizer,L_train,L_valid,device='cpu'):
    for epoch in range(epochs):
        L_train,L_valid = train_valid_one_epoch(model,dataloader,loss_function,optimizer,L_train,L_valid,device)

    return(L_train,L_valid)

def train_valid_one_epoch(model,dataloader,loss_function,optimizer,L_train,L_valid,device):
    # Train and Valid each epoch 
    model.train()   #Activate Dropout 
    loss_train = loop(model,dataloader,loss_function,optimizer,device,training_mode='train')
    L_train.append(loss_train)
    model.eval()   # Desactivate Dropout 
    loss_valid = loop(model,dataloader,loss_function,optimizer,device,training_mode='valid') 
    L_valid.append(loss_valid)

    return(L_train,L_valid)


def loop(model,dataloader,loss_function,optimizer,device,training_mode):
    loss_epoch,nb_samples = 0,0
    with torch.set_grad_enabled(training_mode=='train'):
        for x_b,y_b in dataloader[training_mode]:
            x_b,y_b = x_b.to(device),y_b.to(device)
            pred = model(x_b)
            loss = loss_function(pred,y_b)

            # Back propagation (after each mini-batch)
            if training_mode == 'train': 
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            # Keep track on metrics 
            nb_samples += x_b.shape[0]
            loss_epoch += loss.item()*x_b.shape[0]
    return(loss_epoch/nb_samples)


def get_train_valid_lower_upper(df_classification,split_prop):
    split_ind = int(len(df_classification)*split_prop)
    X_train,X_valid = df_classification.drop(columns = ['res_lower','res_upper'])[:split_ind],df_classification.drop(columns = ['res_lower','res_upper'])[split_ind:]
    Y_train_lower,Y_valid_lower = df_classification['res_lower'][:split_ind],df_classification['res_lower'][split_ind:]
    Y_train_upper,Y_valid_upper = df_classification['res_upper'][:split_ind],df_classification['res_upper'][split_ind:]
    return(X_train,X_valid,Y_train_lower,Y_valid_lower,Y_train_upper,Y_valid_upper)


def minmax_normalise(mini,maxi,X_tensor):
    return((X_tensor-mini)/(maxi-mini))

def minmax_normalise_train_valid(Y_train_lower,Y_valid_lower):
    mini_l,maxi_l=Y_train_lower.min(),Y_train_lower.max()
    Y_train_lower = minmax_normalise(mini_l,maxi_l,Y_train_lower)
    Y_valid_lower = minmax_normalise(mini_l,maxi_l,Y_valid_lower)
    return(mini_l,maxi_l,Y_train_lower,Y_valid_lower)

def load_hp(df_classification):
    # HyperParameters
    batch_size = 32
    epochs = 100
    c_in = len(df_classification.columns)-2
    h_dim1,h_dim2 = 16,8
    c_out = 1
    lr=1e-4
    weight_decay= 0.98
    return(batch_size,epochs,c_in,h_dim1,h_dim2,c_out,lr,weight_decay)


def load_model(c_in,h_dim1,h_dim2,c_out,lr,weight_decay):
    model =  MLP(c_in,h_dim1,h_dim2,c_out)
    optimizer = torch.optim.AdamW(model.parameters(),lr=lr,weight_decay= weight_decay)
    return(model,optimizer)


def load_dataloader(X_train,X_valid,Y_train,Y_valid,batch_size):
    dataloader = {}
    for training_mode, X,Y in zip(['train','valid'],[torch.tensor(X_train.values.astype(float)),torch.tensor(X_valid.values.astype(float))],[torch.tensor(Y_train.values),torch.tensor(Y_valid.values)]):
        X = X.to(torch.float32)
        Y = Y.to(torch.float32)
        inputs = list(zip(X,Y))
        dataloader[training_mode] = torch.utils.data.DataLoader(inputs, batch_size=batch_size,shuffle = False)
    
    return(dataloader)

def train_regression_MLP(Models,band_name,loss_function,X_train,X_valid,Y_train,Y_valid,batch_size,epochs,c_in,h_dim1,h_dim2,c_out,lr,weight_decay):
    # Load model 
    model,optimizer = load_model(c_in,h_dim1,h_dim2,c_out,lr,weight_decay)
    # Load dataloader
    dataloader = load_dataloader(X_train,X_valid,Y_train,Y_valid,batch_size)
    # Train Model
    L_train,L_valid = train_and_valid(model,epochs,dataloader,loss_function,optimizer,L_train=[],L_valid=[],device='cpu')
    # Plot Results
    pd.DataFrame(dict(train_loss=L_train[10:],train_valid = L_valid[10:])).plot()
    # Update Dictionary 
    Models = update_dictionnary(Models,model,band_name,optimizer,L_train,L_valid) 
    return(Models)

def update_dictionnary(Models,model,band_name,optimizer,L_train,L_valid):
    Models[band_name]['model'] = model
    Models[band_name]['optimizer'] = optimizer
    Models[band_name]['L_train'] = L_train
    Models[band_name]['L_valid'] = L_valid
    return(Models)

def load_and_train_model(df_classification,split_prop = 0.7, task = 'regressio'):
    # Load Data
    (X_train,X_valid,Y_train_lower,Y_valid_lower,Y_train_upper,Y_valid_upper) = get_train_valid_lower_upper(df_classification,split_prop)

    # MinMaxNormalize
    mini_l,maxi_l,Y_train_lower,Y_valid_lower = minmax_normalise_train_valid(Y_train_lower,Y_valid_lower)
    mini_u,maxi_u,Y_train_upper,Y_valid_upper = minmax_normalise_train_valid(Y_train_upper,Y_valid_upper) 
    
    # Init Hyperparameter
    (batch_size,epochs,c_in,h_dim1,h_dim2,c_out,lr,weight_decay) = load_hp(df_classification)

    # Init
    Models = {'lower':{},'upper':{}}
    loss_function = nn.MSELoss()

    for Y_train,Y_valid,band_name in zip([Y_train_lower,Y_train_upper],[Y_valid_lower,Y_valid_upper],['lower','upper']):

        if task == 'regression':
            Models = train_regression_MLP(Models,band_name,loss_function,X_train,X_valid,Y_train,Y_valid,batch_size,epochs,c_in,h_dim1,h_dim2,c_out,lr,weight_decay)

    return(Models,mini_l,maxi_l,mini_u,maxi_u)


# Map Residual to Label 
def label2res(residual,labels,step):
    return labels[min(int(abs(residual)/step),len(labels)-1)]


def Continuous_res2Discret_label(Y_train_lower,labels,step):
    df_lower = pd.DataFrame(Y_train_lower)
    column = df_lower.columns[0]
    df_lower['label'] = df_lower[column].apply(lambda residual : label2res(residual,labels,step))

    # Tackle negativ residual 
    df_lower['label'] = df_lower['label'] + 1   # add 1 to each label
    #df_lower['label'][df_lower[column] < 0] = 0  # set label0 to negativ residual 
    df_lower['label'] = df_lower.apply(lambda row : 0 if row[column] < 0 else row['label'],axis=1)

    return(df_lower['label'])



def get_classifier_inputs(ds,ds_valid,labels,step):
   # preprocess dataset, split into training and test part
    X_train, y_train_continous = ds
    X_test, y_test_continous = ds_valid

    y_train = Continuous_res2Discret_label(y_train_continous,labels,step)
    y_test = Continuous_res2Discret_label(y_test_continous,labels,step)

    X_train = X_train.values
    X_test = X_test.values
    y_train = y_train.values
    y_test = y_test.values

    return(X_train,X_test,y_train,y_test)

def sklearn_grid_search(clf,X_train, y_train,param_grid,cv=5,factor=2):
    sh = HalvingGridSearchCV(clf,   #classifier
                                param_grid,   #search space
                                cv=cv, #nb of fold
                            factor=factor,   # Moitié des candidats sélectionné après chaque iteration 
                            #resource='n_estimators',
                            #max_resources=30
                            ).fit(X_train, y_train)
    print(sh.best_estimator_)
    return(sh)

def train_classifier(clf,X_train,y_train,X_test,y_test):
    clf.fit(X_train, y_train)
    score_test = clf.score(X_test, y_test)
    score_train = clf.score(X_train, y_train)
    #print(f'Score on Train Dataset: {score_train}')
    print(f'Score on Test Dataset: {score_test}')
    return(clf)


