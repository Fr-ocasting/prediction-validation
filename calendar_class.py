import torch
import pandas as pd

def find_class(elt, week_group, hour_minute_group):
    day,hour,minute = elt
    
    # Convertir l'heure et les minutes en minutes depuis minuit pour faciliter la comparaison
    time_in_minutes = hour * 60 + minute
    
    for week in week_group:
        if week[0] <= day <= week[1]:
            for start, end in hour_minute_group:
                # Convertir les heures de début et de fin en minutes depuis minuit
                start_minutes = start[0] * 60 + start[1]
                end_minutes = end[0] * 60 + end[1]
                
                # Gérer le cas où l'intervalle passe à travers minuit
                if end_minutes < start_minutes:
                    if time_in_minutes >= start_minutes or time_in_minutes < end_minutes:
                        return (week, [start, end])
                else:
                    if start_minutes <= time_in_minutes < end_minutes:
                        return (week, [start, end])
    return None

def get_week_hour_minute_class(type_class):
    '''No Specific class'''
    if type_class == 0:
        week_group = [[0,6]]
        hour_minute_group = [[(0,0),(0,0)]]

    ''' Group of tuple: '''
    if type_class == 1:
        week_group = [[0,2],[3,3],[4,4],[5,5],[6,6]]
        hour_minute_group = [[(2,0),(5,0)],
                            [(5,0),(6,0)],
                            [(6,0),(7,0)],
                            [(7,0),(9,0)],
                            [(9,0),(12,0)],
                            [(12,0),(14,0)],
                            [(14,0),(17,0)],
                            [(17,0),(19,0)],
                            [(19,0),(21,0)],
                            [(21,0),(2,0)],
                            ]
        
    ''' Class where every tuple is independant: '''
    if type_class == 2:
        week_group = [[k,k] for k in range(7)]
        tmps = [[[(h1,0),(h1,15)],[(h1,15),(h1,30)],[(h1,30),(h1,45)],[(h1,45),(h1+1,0)]] for h1 in range(23)]
        hour_minute_group = []
        for l in tmps:
            for elt in l:
                hour_minute_group.append(elt)
        hour_minute_group = hour_minute_group+[[(23,0),(23,15)],[(23,15),(23,30)],[(23,30),(23,45)],[(23,45),(0,0)]]

    ''' Class where every couple (hour,day) is independant '''
    if type_class == 3:
        week_group = [[k,k] for k in range(7)]
        tmps = [[[(h1,0),(h1+1,0)]] for h1 in range(23)]
        hour_minute_group = []
        for l in tmps:
            for elt in l:
                hour_minute_group.append(elt)
        hour_minute_group = hour_minute_group+[[(23,0),(0,0)]]

    return(week_group,hour_minute_group)

def get_time_slots_labels(dataset,calendar_class):

    # Associate Label to a timestamp
    df_time_slots = pd.DataFrame(dataset.df_verif[f"t+{dataset.step_ahead-1}"]).rename(columns = {f"t+{dataset.step_ahead-1}":'datetime'})
    df_time_slots['hour'] = df_time_slots.datetime.dt.hour
    df_time_slots['weekday'] = df_time_slots.datetime.dt.weekday
    df_time_slots['minutes'] = df_time_slots.datetime.dt.minute

    week_group,hour_minute_group = get_week_hour_minute_class(calendar_class)
    dic_class2rpz = {i*len(hour_minute_group)+k:([w1,w2],[(h1,m1),(h2,m2)]) for i,(w1,w2) in enumerate(week_group) for k,([(h1,m1),(h2,m2)]) in enumerate(hour_minute_group)  }
    dic_rpz2class = {f"{'_'.join(list(map(str,[w1,w2])))}-{'_'.join(list(map(str,[(h1,m1),(h2,m2)])))}":i*len(hour_minute_group)+k for i,(w1,w2) in enumerate(week_group) for k,([(h1,m1),(h2,m2)]) in enumerate(hour_minute_group)  }

    # According choosen type_class: 
    if calendar_class == 0:
        dataset.time_slots_labels = torch.Tensor([0.0]*len(dataset.df_verif))
        return(dataset.time_slots_labels,dic_class2rpz,dic_rpz2class,1)
    else:
        df_time_slots['calendar_class_rpz'] = df_time_slots.apply(lambda row : find_class((row.weekday,row.hour,row.minutes),week_group,hour_minute_group),axis=1)
        df_time_slots['calendar_class_rpz_str'] = df_time_slots.calendar_class_rpz.apply(lambda class_rpz : f"{'_'.join(list(map(str,class_rpz[0])))}-{'_'.join(list(map(str,class_rpz[1])))}" )
        df_time_slots['calendar_class'] = df_time_slots.calendar_class_rpz_str.apply(lambda class_rpz : dic_rpz2class[class_rpz]) 
        #time_slots_labels = torch.Tensor(df_time_slots['calendar_class'])
        time_slots_labels = torch.Tensor(df_time_slots['calendar_class'].values)  #.long()
        nb_words_embedding = len(df_time_slots['calendar_class'].unique())
        dataset.time_slots_labels = time_slots_labels
        return(dataset.time_slots_labels,dic_class2rpz,dic_rpz2class,nb_words_embedding)