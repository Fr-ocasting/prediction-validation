from datetime import datetime, timedelta
import pandas as pd

import sys 
import os 
current_file_path = os.path.abspath(os.path.dirname(__file__))
parent_dir = os.path.abspath(os.path.join(current_file_path,'..'))
if parent_dir not in sys.path:
    sys.path.insert(0,parent_dir)


def peak_hour(coverage,morning_peak= [7,8,9], evening_peak = [16,17,18,19]):
    """Identifies peak hours from a list of timestamps 'coverage'.

    args:
    ------
    coverage : list of Timestamp, a list containing timestamps to evaluate.

    outputs:
    ------
    peak_hours : list of Timestamp, a list of timestamps that correspond to peak hours."""
    peak_hour = [x for x in coverage if x.hour in morning_peak+evening_peak]
    return peak_hour

def build_window(kick_off_time,coverage,range):

    match_times= [ts for kick_time in kick_off_time
                    for ts in coverage
                    if kick_time - timedelta(minutes=range) <= ts <= kick_time + timedelta(minutes=range)
                    ]
    kick_off_time = [ts for ts in kick_off_time if ts in coverage]

    return kick_off_time,match_times

def rugby_matches(coverage, range = 3*60):
   """Retrieves the rugby match times of the 'Lyon Lou Rugby Team' at 'Gerland Stadium' from a list of of timestamps 'coverage'.

    args:
    ------
    coverage : list of Timestamp, a list containing timestamps to evaluate.
    range : int, the range in minutes to look for timestamps around each kick-off time  

    outputs:
    ------
    match_times : list of Timestamp, a list of timestamps within +/- 3 hours of the match time ."""
   kick_off_time= [
    datetime(2019,1,26,20,45),  # 26 Jan 2019  20h45 
    datetime(2019,2,24,16,50),  # 24 Fev 2019  16h50 
    datetime(2019,3,2,20,45),  # 2 Mar 2019  20h45
    datetime(2019,3,23,14,45),   # 23 Mar 2019 14h45
    datetime(2019, 4, 13, 18, 0),  # 13 Avr 2019 18h 
    datetime(2019, 5, 5, 12, 30),   # 5 Mai 2019 12h30 
    datetime(2019, 5, 18, 21, 0),   # 18 Mai 2019 21h 
    datetime(2019, 6, 1, 17, 0),    # 1 Juin 2019 17h 
    datetime(2019, 8, 24, 18, 0),    # 24 Aout 2019 18h 
    datetime(2019, 9, 1, 17, 5),    # 1 Sept 2019 17h05 
    datetime(2019, 9, 14, 18, 0),    # 14 Sept 2019 18h 
    datetime(2019, 10, 5, 20, 45),# 5 Oct 2019 20h45 
    datetime(2019, 10, 12, 20, 45), # 12 Oct 2019 20h45 
    datetime(2019, 11, 10, 12, 30),  # 10 Nov 2019 12h30 
    datetime(2019, 12, 28, 14, 0),   # 28 Dec 2019 14h 
    datetime(2020, 1, 25, 15, 30),   # 25 Jan 2020 15h30 
    datetime(2020, 2, 23, 12, 30),   # 23 Fev 2020 12h30 
    datetime(2020, 9, 5, 18, 0),     # 5 Sept 2020 18h 
    datetime(2020, 10, 5, 20, 45),    # 5 Oct 2020 20h45 
    datetime(2020, 10, 18, 17, 0)    # 18 Oct 2020 17h
   ]

   return build_window(kick_off_time,coverage,range)


if False:
    def basket_matches(coverage, range = 3*60):
        """Retrieves the rugby match times of the 'Lyon Lou Rugby Team' at 'Gerland Stadium' from a list of of timestamps 'coverage'.

            args:
            ------
            coverage : list of Timestamp, a list containing timestamps to evaluate.
            range : int, the range in minutes to look for timestamps around each kick-off time  

            outputs:
            ------
            match_times : list of Timestamp, a list of timestamps within +/- 3 hours of the match time ."""
        kick_off_time= [
            datetime(2019,3,2,20,45),      
            datetime(2019,3,11,20,45),  
            datetime(2019,3,24,20,45),  
            datetime(2019,4,1,20,45),   
            datetime(2019,4,6,20,45),   
            datetime(2019,4,11,20,45), 
            datetime(2019,4,14,20,45), 
            datetime(2019,4,22,20,45), 
            datetime(2019,4,25,20,45),
            datetime(2019,4,29,20,45), 
            datetime(2019,5,6,20,45), 
            datetime(2019,5,8,20,45), 
            datetime(2019,5,16,20,45), 
            datetime(2019,5,19,20,45), 
            datetime(2019,5,26,20,45), 
            datetime(2019,5,28,20,45), 
            datetime(2019,6,3,20,45), 
            datetime(2019,6,5,20,45), 
            datetime(2019,6,8,20,45), 
            datetime(2019,6,16,20,45), 
            datetime(2019,6,18,20,45), 
            datetime(2019,6,21,20,45), 
            datetime(2019,6,23,20,45), 
            datetime(2019,6,26,20,45), 
        ]

        
        return build_window(kick_off_time,coverage,range)


