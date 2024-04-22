import pandas as pd
def time_embedding2dict(args):
    dict1 = dict(CalendarClass = args.calendar_class, 
        Position = args.position, 
        Specific_lr = args.specific_lr,
        Type_calendar = args.type_calendar,
        )
    
    return(dict1)

def results2dict(args,epoch,picp,mpiw,valid_loss,train_loss):
    '''Set results in a dictionnary. Will be used as a row for results dataframe'''
    dict_row = {}
    dict1 = time_embedding2dict(args)
    dict2 = dict(epoch = epoch, PICP = picp, MPIW = mpiw, valid_loss = valid_loss, train_loss = train_loss)
    for dic in [dict1,dict2]:
        dict_row.update(dic)
    return(dict_row)

def Multi_results2dict(args,mean_picp,mean_mpiw,dict3,dict4):
    '''Same then resuls2dict, but change the name of PICP and MPIW as it's for several folds '''
    dict_row = {}
    dict1 = time_embedding2dict(args)
    dict2 = dict(PICP_mean = mean_picp.item(),MPIW_mean = mean_mpiw.item())
    for dic in [dict1,dict2,dict3,dict4]:
        dict_row.update(dic)
    return(dict_row)


def update_results_df(results_df, dict_row):
    results_df = pd.concat([results_df,pd.DataFrame.from_records([dict_row])])
    return(results_df)


def build_results_df(results_df,args, mean_picp,mean_mpiw,dict2,dict3):
    dict_row = Multi_results2dict(args,mean_picp,mean_mpiw,dict2,dict3)
    results_df = update_results_df(results_df, dict_row)
    return(results_df)


