import torch
import pandas as pd
import datetime 
import torch

def is_morning(predicted_dates_h,start = datetime.time(7, 00), end=datetime.time(11, 30)):
    return (predicted_dates_h.dt.time >= start) & (predicted_dates_h.dt.time <= end)

def is_evening(predicted_dates_h, start = datetime.time(16, 00), end=datetime.time(20, 00)):
    return (predicted_dates_h.dt.time >= start) & (predicted_dates_h.dt.time <= end)

def is_morning_peak(predicted_dates_h, start = datetime.time(7, 30), end=datetime.time(9, 00)):
    return (predicted_dates_h.dt.time >= start) & (predicted_dates_h.dt.time <= end)

def is_evening_peak(predicted_dates_h, start = datetime.time(17, 00), end=datetime.time(19, 00)):
    return (predicted_dates_h.dt.time >= start) & (predicted_dates_h.dt.time <= end)

def is_weekday(predicted_dates_h):
    return (predicted_dates_h.dt.dayofweek < 5)


def get_mask_working_day(s_dates,mask_specific_period,city):
    """
    Create a mask for working days, excluding weekends and bank holidays.
    Args:   
    -----
        s_dates (pd.Series): Series of datetime objects.
        mask_specific_period (pd.Series): Boolean mask for specific time periods.
        city (str): City name to check for bank holidays.

    Returns:
    ------
        pd.Series: Boolean mask indicating whether each date is a working day.
    """
    weekday_mask = is_weekday(s_dates)
    dates_is_bank_holidays = s_dates.apply(lambda x: is_bank_holidays(x,city= city))
    filter_mask = mask_specific_period & weekday_mask & ~dates_is_bank_holidays
    return filter_mask

def get_temporal_mask(s_dates: pd.Series, 
                      start: datetime.time = datetime.time(7, 0), 
                      end: datetime.time = datetime.time(21, 0),
                      temporal_agg: str = None,
                      city = None,
                      start_morning_peak: datetime.time = datetime.time(7, 30),
                      end_morning_peak: datetime.time = datetime.time(9, 0),
                      start_evening_peak: datetime.time = datetime.time(17, 0),
                      end_evening_peak: datetime.time = datetime.time(19, 0)
                      ) -> pd.Series:
    """
    Create a mask for the specified time range within the given dates.
    
    Args:
    -----
        s_dates (pd.Series): Series of datetime objects.
        start (datetime.time): Start time for the mask.
        end (datetime.time): End time for the mask.
        temporal_agg (str): Temporal aggregation type (e.g., 'morning_peak', 'evening_peak', etc.).
        city (str): City name to check for bank holidays.

    Returns:
    -----
        pd.Series: Boolean mask indicating whether each date falls within the specified time range.
    """
    def get_usefull_hours(s_dates, start, end):
        if start < end:
            return (s_dates.dt.time >= start) & (s_dates.dt.time <= end)
        else: 
            return (s_dates.dt.time >= start) | (s_dates.dt.time <= end)

    if temporal_agg == 'morning_peak':
        filter_mask = get_mask_working_day(s_dates,is_morning_peak(s_dates,start_morning_peak,end_morning_peak),city)
    elif temporal_agg == 'evening_peak':
        filter_mask = get_mask_working_day(s_dates,is_evening_peak(s_dates,start_evening_peak,end_evening_peak),city)
    elif temporal_agg == 'off_peak':
        filter_mask = get_mask_working_day(s_dates,~is_morning_peak(s_dates,start_morning_peak,end_morning_peak) & ~is_evening_peak(s_dates,start_evening_peak,end_evening_peak),city)
    elif temporal_agg == 'non_business_day':
        weekday_mask = is_weekday(s_dates)
        dates_is_bank_holidays = s_dates.apply(lambda x: is_bank_holidays(x,city= city))
        usefull_hours =  get_usefull_hours(s_dates, start, end)
        filter_mask = usefull_hours & (~weekday_mask | dates_is_bank_holidays)
    elif temporal_agg == 'business_day':
        weekday_mask = is_weekday(s_dates)
        dates_is_bank_holidays = s_dates.apply(lambda x: is_bank_holidays(x,city= city))
        usefull_hours =  get_usefull_hours(s_dates, start, end)
        filter_mask = usefull_hours & weekday_mask & ~dates_is_bank_holidays
    elif temporal_agg == 'morning':
        filter_mask = get_mask_working_day(s_dates,is_morning(s_dates),city)
    elif temporal_agg == 'evening':
        filter_mask = get_mask_working_day(s_dates,is_evening(s_dates),city)
    elif temporal_agg is None:
        usefull_hours =  get_usefull_hours(s_dates, start, end)
        filter_mask = usefull_hours
    else: 
        raise ValueError(f"Unknown temporal aggregation: {temporal_agg}")

    return filter_mask



BANK_HOLIDAYS = {
    'Lyon': [
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
        ],
    'California': [
        # 2016
        "2016-01-01",  # New Year's Day
        "2016-01-18",  # Martin Luther King Jr. Day (3ᵉ lundi de janvier)
        "2016-02-15",  # Washington's Birthday (Presidents Day, 3ᵉ lundi de février)
        "2016-03-25",  # Good Friday (vendredi avant Pâques)
        "2016-05-30",  # Memorial Day (dernier lundi de mai)
        "2016-07-04",  # Independence Day
        "2016-09-05",  # Labor Day (1ᵉʳ lundi de septembre)
        "2016-10-10",  # Columbus Day (2ᵉ lundi d’octobre)
        "2016-11-11",  # Veterans Day
        "2016-11-24",  # Thanksgiving Day (4ᵉ jeudi de novembre)
        "2016-12-26",  # Christmas Day observé (25 déc. tombe un dimanche)

        # 2017
        "2017-01-02",  # New Year's Day observé (1ᵉʳ janv. tombe un dimanche)
        "2017-01-16",  # MLK Jr. Day
        "2017-02-20",  # Presidents Day
        "2017-04-14",  # Good Friday
        "2017-05-29",  # Memorial Day
        "2017-07-04",  # Independence Day
        "2017-09-04",  # Labor Day
        "2017-10-09",  # Columbus Day
        "2017-11-10",  # Veterans Day observé (11 nov. tombe un samedi)
        "2017-11-23",  # Thanksgiving Day
        "2017-12-25",  # Christmas Day

        # 2018
        "2018-01-01",  # New Year's Day
        "2018-01-15",  # MLK Jr. Day
        "2018-02-19",  # Presidents Day
        "2018-03-30",  # Good Friday
        "2018-05-28",  # Memorial Day
        "2018-07-04",  # Independence Day
        "2018-09-03",  # Labor Day
        "2018-10-08",  # Columbus Day
        "2018-11-12",  # Veterans Day observé (11 nov. tombe un dimanche)
        "2018-11-22",  # Thanksgiving Day
        "2018-12-25",  # Christmas Day

        # 2019
        "2019-01-01",  # New Year's Day
        "2019-01-21",  # MLK Jr. Day
        "2019-02-18",  # Presidents Day
        "2019-04-19",  # Good Friday
        "2019-05-27",  # Memorial Day
        "2019-07-04",  # Independence Day
        "2019-09-02",  # Labor Day
        "2019-10-14",  # Columbus Day
        "2019-11-11",  # Veterans Day
        "2019-11-28",  # Thanksgiving Day
        "2019-12-25",  # Christmas Day
    ],
        'Manhattan': [
            # 2020
            "2020-01-01",  # New Year's Day
            "2020-01-20",  # MLK Jr. Day
            "2020-02-17",  # Presidents Day
            "2020-04-10",  # Good Friday
            "2020-05-25",  # Memorial Day
            "2020-07-03",  # Independence Day (Observed)
            "2020-09-07",  # Labor Day
            "2020-10-12",  # Columbus Day
            "2020-11-11",  # Veterans Day
            "2020-11-26",  # Thanksgiving Day
            "2020-12-25",  # Christmas Day

            # 2021
            "2021-01-01",  # New Year's Day
            "2021-01-18",  # MLK Jr. Day
            "2021-02-15",  # Presidents Day
            "2021-04-02",  # Good Friday
            "2021-05-31",  # Memorial Day
            "2021-06-18",  # Juneteenth (Observed - Fed only)
            "2021-07-05",  # Independence Day (Observed)
            "2021-09-06",  # Labor Day
            "2021-10-11",  # Columbus Day
            "2021-11-11",  # Veterans Day
            "2021-11-25",  # Thanksgiving Day
            "2021-12-24",  # Christmas Day (Observed)

            # 2022
            "2022-01-17",  # MLK Jr. Day
            "2022-02-21",  # Presidents Day
            "2022-04-15",  # Good Friday
            "2022-05-30",  # Memorial Day
            "2022-06-20",  # Juneteenth (Observed)
            "2022-07-04",  # Independence Day
            "2022-09-05",  # Labor Day
            "2022-10-10",  # Columbus Day
            "2022-11-11",  # Veterans Day
            "2022-11-24",  # Thanksgiving Day
            "2022-12-26",  # Christmas Day (Observed)
        ]
}




def is_bank_holidays(timestamp,city):
    city_i = 'California' if  'California' in city else city
    bank_holidays = BANK_HOLIDAYS[city_i]
    date = timestamp.strftime("%Y-%m-%d")
    
    return date in bank_holidays

def is_school_holidays(timestamp,city):
    if city == 'Lyon':
        school_holidays = [
            ("2018-12-22", "2019-01-07"),  # Vacances de Noël
            ("2019-02-16", "2019-03-04"),  # Vacances d'hiver
            ("2019-04-13", "2019-04-29"),  # Vacances de printemps
        ]
    elif 'California' in city:
        school_holidays = []
    
    elif 'Manhattan' in city:
        school_holidays = [
            # Année scolaire 2018-2019 (Partie 2019)
            ("2019-02-18", "2019-02-22"),  # Mid-winter Recess
            ("2019-04-19", "2019-04-26"),  # Spring Recess
            
            # Année scolaire 2019-2020
            ("2019-12-23", "2020-01-01"),  # Winter Recess
            ("2020-02-17", "2020-02-21"),  # Mid-winter Recess
            ("2020-03-16", "2020-06-26"),  # FERMETURE COVID (Bâtiments fermés / Remote)
            ("2020-04-09", "2020-04-17"),  # Spring Recess (Inclus dans fermeture COVID)
            
            # Année scolaire 2020-2021
            ("2020-12-24", "2021-01-01"),  # Winter Recess
            ("2021-02-15", "2021-02-19"),  # Mid-winter Recess
            ("2021-03-29", "2021-04-02"),  # Spring Recess
            
            # Année scolaire 2021-2022
            ("2021-12-24", "2022-01-02"),  # Winter Recess
            ("2022-02-21", "2022-02-25"),  # Mid-winter Recess
            ("2022-04-15", "2022-04-22"),  # Spring Recess
            
            # Année scolaire 2022-2023
            ("2022-12-26", "2023-01-02"),  # Winter Recess
            ("2023-02-20", "2023-02-24"),  # Mid-winter Recess
            ("2023-04-10", "2023-04-14"),  # Spring Recess
            
            # Année scolaire 2023-2024
            ("2023-12-25", "2024-01-01"),  # Winter Recess
            ("2024-02-19", "2024-02-23"),  # Mid-winter Recess
            ("2024-04-22", "2024-04-30"),  # Spring Recess (Pessah)
        ]

    for start, end in school_holidays:
        start_date = datetime.datetime.strptime(start, "%Y-%m-%d")
        end_date = datetime.datetime.strptime(end, "%Y-%m-%d")
        if start_date <= timestamp < end_date:
            # remaining days before the end of the holidays
            days_to_end = (end_date - timestamp).days
            return True, days_to_end

    return False, -1

def calendar_inputs(df_dates,
                    calendar_type=['timeofday','dayofweek', 'hour', 'minute', 'bank_holidays', 'school_holidays', 'remaining_holidays'],
                    city = 'Lyon'):
        
    df = pd.DataFrame({'date': df_dates})
    df['date'] = pd.to_datetime(df['date'])

    if 'dayofweek' in calendar_type:
        df['dayofweek'] = df['date'].dt.weekday
    if 'hour' in calendar_type:
        df['hour'] = df['date'].dt.hour
    if 'minute' in calendar_type:
        df['minute'] = df['date'].dt.minute
    if 'bank_holidays' in calendar_type:
        df['bank_holidays'] = df['date'].apply(lambda x: is_bank_holidays(x,city=city))
    if 'school_holidays' in calendar_type:
        school_holidays,remaining_holidays  = zip(*df['date'].apply(lambda x : is_school_holidays(x,city=city)))
        df['school_holidays'] = school_holidays
        if 'remaining_holidays' in calendar_type:
            df['remaining_holidays'] = remaining_holidays
    if 'timeofday' in calendar_type:
        df['timeofday'] = df['date'].dt.hour * 60 + df['date'].dt.minute
        df['timeofday'] = df['timeofday'] / (24 * 60)

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


if False:
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