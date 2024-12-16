import torch
import pandas as pd
from datetime import datetime
import torch

def is_bank_holidays(timestamp):
    # Liste des jours fériés en 2019 en France
    bank_holidays = [
        "2019-01-01",  # Mardi 1er janvier 2019
        "2019-04-22",  # Lundi 22 avril 2019
        "2019-05-01",  # Mercredi 1er mai 2019
        "2019-05-08",  # Mercredi 8 mai 2019
        "2019-05-30",  # Jeudi 30 mai 2019
        "2019-06-10",  # Lundi 10 juin 2019
        "2019-07-14",  # Dimanche 14 juillet 2019
        "2019-08-15",  # Jeudi 15 août 2019
        "2019-11-01",  # Vendredi 1er novembre 2019
        "2019-11-11",  # Lundi 11 novembre 2019
        "2019-12-25",  # Mercredi 25 décembre 2019
    ]
    date = timestamp.strftime("%Y-%m-%d")
    
    return date in bank_holidays

def is_school_holidays(timestamp):
    school_holidays = [
        ("2018-12-22", "2019-01-07"),  # Vacances de Noël
        ("2019-02-16", "2019-03-04"),  # Vacances d'hiver
        ("2019-04-13", "2019-04-29"),  # Vacances de printemps
    ]

    for start, end in school_holidays:
        start_date = datetime.strptime(start, "%Y-%m-%d")
        end_date = datetime.strptime(end, "%Y-%m-%d")
        if start_date <= timestamp < end_date:
            # remaining days before the end of the holidays
            days_to_end = (end_date - timestamp).days
            return True, days_to_end

    return False, -1

def calendar_inputs(df_dates, calendar_type=['dayofweek', 'hour', 'minute', 'bank_holidays', 'school_holidays', 'remaining_holidays'],city = 'Lyon'):
    if city != 'Lyon': 
        if ('bank_holidays' in calendar_type) or ('school_holidays' in calendar_type) or ('remaining_holidays' in calendar_type) :
            raise NotImplementedError(f'The holidays are not designed for another city than Lyon (here city is {city})')
        
    df = pd.DataFrame({'date': df_dates})
    df['date'] = pd.to_datetime(df['date'])

    if 'dayofweek' in calendar_type:
        df['dayofweek'] = df['date'].dt.weekday
    if 'hour' in calendar_type:
        df['hour'] = df['date'].dt.hour
    if 'minute' in calendar_type:
        df['minute'] = df['date'].dt.minute
    if 'bank_holidays' in calendar_type:
        df['bank_holidays'] = df['date'].apply(is_bank_holidays)
    if 'school_holidays' in calendar_type:
        school_holidays,remaining_holidays  = zip(*df['date'].apply(is_school_holidays))
        df['school_holidays'] = school_holidays
        if 'remaining_holidays' in calendar_type:
            df['remaining_holidays'] = remaining_holidays

    return df.drop(columns=['date'])


def one_hot_encode_dataframe(df, columns):
    """
    Encode selected columns in a DataFrame into a dictionary of one-hot encoded PyTorch tensors.

    :param df: The DataFrame to encode.
    :param columns: List of column names to encode.
    :return: Dictionary where keys are column names and values are tensors of one-hot encodings.
    """
    one_hot_dict = {}
    for column in columns:
        unique_labels = df[column].fillna(-1).unique().tolist()  # Treat NaN as -1
        df[column] = df[column].fillna(-1)
        one_hot = pd.get_dummies(df[column], prefix=column)
        one_hot_tensor = torch.tensor(one_hot.values, dtype=torch.float32)
        one_hot_dict[column] = one_hot_tensor
    return one_hot_dict




if __name__ == '__main__':

    calendar_type=['dayofweek', 'hour', 'minute', 'bank_holidays', 'school_holidays', 'remaining_holidays']
    dates = pd.date_range(start='01/01/2019', end='01/01/2020', freq='15min')[:-1]

    # Load calendar-related informaiton : 
    df_calendar = calendar_inputs(dates,city = 'Lyon')

    # Load One-Hot encoded Vector :
    one_hot_dict = one_hot_encode_dataframe(df_calendar, calendar_type)



# ===================================================================================================
# ===================================================================================================
# ===================== A SUPPRIMER A SUPPRIMER A SUPPRIMER =========================================
# ===================================================================================================
# ===================================================================================================
# ====== Calendar Class compliqué. Ce formalisme ne peut pas prendre en compte les vacances, les jours fériées, ou autre 'événement' notable.



# ===================================================================================================
# ====== Calendar Class compliqué. Ce formalisme ne peut pas prendre en compte les vacances, les jours fériées, ou autre 'événement' notable. 
# ===================================================================================================

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

def get_time_slots_labels(dataset,nb_class = [0,1,2,3]):
    dataset.nb_class = nb_class
    tsph = dataset.time_step_per_hour 
    Dic_T_labels,Dic_nb_words_embedding,Dic_class2rpz,Dic_rpz2class = {},{},{},{}
    # Associate Label to a timestamp
    df_time_slots = pd.DataFrame(dataset.df_verif[f"t+{dataset.step_ahead-1}"]).rename(columns = {f"t+{dataset.step_ahead-1}":'datetime'})
    df_time_slots['hour'] = df_time_slots.datetime.dt.hour
    df_time_slots['weekday'] = df_time_slots.datetime.dt.weekday
    df_time_slots['minutes'] = df_time_slots.datetime.dt.minute

    for calendar_class in range(len(nb_class)):
        week_group,hour_minute_group = get_week_hour_minute_class(calendar_class)
        dic_class2rpz = {i*len(hour_minute_group)+k:([w1,w2],[(h1,m1//tsph),(h2,m2//tsph)]) for i,(w1,w2) in enumerate(week_group) for k,([(h1,m1),(h2,m2)]) in enumerate(hour_minute_group)  }
        dic_rpz2class = {f"{'_'.join(list(map(str,[w1,w2])))}-{'_'.join(list(map(str,[(h1,m1),(h2,m2)])))}":i*len(hour_minute_group)+k for i,(w1,w2) in enumerate(week_group) for k,([(h1,m1),(h2,m2)]) in enumerate(hour_minute_group)  }

        # According choosen type_class: 
        if calendar_class == 0:
            T_labels = torch.Tensor([0.0]*len(dataset.df_verif))
            nb_words_embedding = 1
        else:
            df_time_slots['calendar_class_rpz'] = df_time_slots.apply(lambda row : find_class((row.weekday,row.hour,row.minutes),week_group,hour_minute_group),axis=1)
            df_time_slots['calendar_class_rpz_str'] = df_time_slots.calendar_class_rpz.apply(lambda class_rpz : f"{'_'.join(list(map(str,class_rpz[0])))}-{'_'.join(list(map(str,class_rpz[1])))}" )
            df_time_slots['calendar_class'] = df_time_slots.calendar_class_rpz_str.apply(lambda class_rpz : dic_rpz2class[class_rpz]) 
            #time_slots_labels = torch.Tensor(df_time_slots['calendar_class'])
            T_labels = torch.Tensor(df_time_slots['calendar_class'].values)  #.long()
            nb_words_embedding = len(df_time_slots['calendar_class'].unique()) 

        Dic_class2rpz[calendar_class] = dic_class2rpz
        Dic_rpz2class[calendar_class] = dic_rpz2class
        Dic_nb_words_embedding[calendar_class] = nb_words_embedding
        Dic_T_labels[calendar_class] = T_labels

    dataset.time_slots_labels = Dic_T_labels


    return(Dic_T_labels,Dic_class2rpz,Dic_rpz2class,Dic_nb_words_embedding)



# ===================================================================================================
# ====== Calendar Class compliqué. Ce formalisme ne peut pas prendre en compte les vacances, les jours fériées, ou autre 'événement' notable. 
# ===================================================================================================