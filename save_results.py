
import pandas as pd
def time_embedding_tuning(args,mean_picp,mean_mpiw):
    dict1 = dict(CalendarClass = args.calendar_class, 
        Position = args.position, 
        Specific_lr = args.specific_lr,
        Type_calendar = args.type_calendar,
        PICP_mean = mean_picp.item(),
        MPIW_mean = mean_mpiw.item()
        )
    
    return(dict1)

def set_new_results_row(args,mean_picp,mean_mpiw,dict2,dict3):
    dict_row = {}
    #columns = []
    dict1 = time_embedding_tuning(args,mean_picp,mean_mpiw)
    for dic in [dict1,dict2,dict3]:
        dict_row.update(dic)
        #columns = columns+ list(dic.keys())
    return(dict_row)#,columns)

def update_results_df(results_df,args, mean_picp,mean_mpiw,dict2,dict3,i):
    dict_row = set_new_results_row(args,mean_picp,mean_mpiw,dict2,dict3)
    #results_df = results_df.append(pd.DataFrame(dict_row, columns = columns,index = [i]))
    results_df = pd.concat([results_df,pd.DataFrame.from_records([dict_row])])
    return(results_df)