from datetime import datetime
from time import time
import numpy as np
try:
    from pynvml.smi import nvidia_smi
    nvidia_smi_available = True
except:
    print("'pynvml' is not available on this environment.")
    nvidia_smi_available = False
import json

###############################
#Author : Bertrand CABOT from IDRIS(CNRS)
#Modified by Romain ROCHAS from LICIT-ECO7 
########################

class Chronometer:
    """
    A light profiler to time a pytorch training loop

    Methods
    -------
    power_measurement(self)
        get the power measurement at time point from CUDA nvsmi
    tac_time(self, clear=False)
        like a stopwatch, get time difference between each call
    clear(self)
        clear all timer
    display(self)
        print all the traces (in out log)
    ...
    
    Example
    -------
    chrono = Chronometer()
    
    chrono.start()
    ...
    for epoch in range(args.epochs):    
        
        chrono.next_iter()
        
        for i, (samples, labels) in enumerate(train_loader):    
            
            chrono.forward()
            
            optimizer.zero_grad()
            outputs = model(samples)
            loss = criterion(samples, labels)
            
            chrono.backward()       

            loss.backward()
            optimizer.step()

            chrono.update()
            ...
            #### VALIDATION ############
            chrono.validation()
            for iv, (val_images, val_labels) in enumerate(val_loader):
                ....
            chrono.validation()
            #### END OF VALIDATION ############
            
            chrono.next_iter()
            
    chrono.stop()
    """
    def __init__(self):
        self.time_perf_train = []
        self.time_perf_load = []
        self.time_perf_forward = []
        self.time_perf_backward = []
        self.power = []
        self.time_scheduler = []
        self.time_tracking_pi = []
        self.time_saving_model = []
        self.time_plotting = []
        self.time_prefetch = []

        self.start_proc = None
        self.stop_proc = None
        self.start_training = None
        self.start_dataload = None
        self.start_backward = None
        self.start_forward = None
        self.start_valid = None
        self.val_time = None
        self.start_calib = None
        self.cal_time = None
        self.time_point = None
        self.start_plotting = None
        self.start_save_model = None
        self.start_track_pi = None
        self.start_scheduler = None
        self.start_prefetch = None

        if nvidia_smi_available:
            self.nvsmi = nvidia_smi.getInstance()
        
    def power_measurement(self):
        powerquery = self.nvsmi.DeviceQuery('power.draw')['gpu']
        for g in range(len(powerquery)):
            self.power.append(powerquery[g]['power_readings']['power_draw'])
    
    def tac_time(self, clear=False):
        if self.time_point == None or clear:
            self.time_point = time()
            return
        else:
            new_time = time() - self.time_point
            self.time_point = time()
            return new_time
    
    def clear(self):
        self.time_perf_train = []
        self.time_perf_load = []
        self.time_perf_forward = []
        self.time_perf_backward = []
        
    def start(self):
        self.start_proc = datetime.now()
    
    def stop(self):
        self.stop_proc = datetime.now()
            
    def _dataload(self):
        if self.start_dataload==None: self.start_dataload = time()
        else:
            self.time_perf_load.append(time() - self.start_dataload)
            self.start_dataload = None
                
    def _training(self):
        if self.start_training==None: self.start_training = time()
        else:
            self.time_perf_train.append(time() - self.start_training)
            self.start_training = None
                
    def _forward(self):
        if self.start_forward==None: self.start_forward = time()
        else:
            self.time_perf_forward.append(time() - self.start_forward)
            self.start_forward = None
                
    def _backward(self):
        if self.start_backward==None: self.start_backward = time()
        else:
            self.time_perf_backward.append(time() - self.start_backward)
            self.start_backward = None

    def _torch_scheduler(self):
        if self.start_scheduler == None: self.start_scheduler = time()
        else:
            self.time_scheduler.append(time()-self.start_scheduler)
            self.start_scheduler = None

    def _track_pi(self):
        if self.start_track_pi == None: self.start_track_pi = time()
        else:
            self.time_tracking_pi.append(time()-self.start_track_pi)
            self.start_track_pi = None

    def _save_model(self):
        if self.start_save_model == None: self.start_save_model = time()
        else:
            self.time_saving_model.append(time()-self.start_save_model)
            self.start_save_model = None

    def _plotting(self):
        if self.start_plotting == None: self.start_plotting = time()
        else:
            self.time_plotting.append(time()-self.start_plotting)
            self.start_plotting = None   

    def _prefetch_all_data(self):
        if self.start_prefetch == None: self.start_prefetch = time()
        else:
            self.time_prefetch.append(time()-self.start_prefetch)
            self.start_prefetch = None       
            
    def next_iter(self):
        self._dataload()
        
    def forward(self):
        self._dataload()
        self._training()
        self._forward()
        
    def backward(self):
        self._forward()
        self._backward()
    
    def update(self):
        self._backward()
        if nvidia_smi_available:
            self.power_measurement()
        self._training()

    def torch_scheduler(self):
        self._torch_scheduler()
    
    def track_pi(self):
        self._track_pi()

    def save_model(self):
        self._save_model()

    def plotting(self):
        self._plotting()

    def prefetch_all_data(self):
        self._prefetch_all_data()
 
    def validation(self):
        if self.start_valid==None: self.start_valid = datetime.now()
        else: 
            self.val_time = datetime.now() - self.start_valid
            self.start_valid = None

    '''
    def calibration(self):
        if self.start_calib==None: self.start_calib = datetime.now()
        else: 
            self.cal_time = datetime.now() - self.start_calib
            self.start_calib = None
    '''

    def display(self):
        total_times = [np.sum(self.time_perf_load) if len(self.time_perf_load) >0 else 0,
                       np.sum(self.time_perf_forward) if len(self.time_perf_forward) >0 else 0,
                       np.sum(self.time_perf_backward) if len(self.time_perf_backward) >0 else 0,
                       np.sum(self.time_plotting) if len(self.time_plotting) >0 else 0,
                       np.sum(self.time_saving_model) if len(self.time_saving_model) >0 else 0,
                       np.sum(self.time_tracking_pi) if len(self.time_tracking_pi) >0 else 0,
                       np.sum(self.time_scheduler) if len(self.time_scheduler) >0 else 0,
                       np.sum(self.time_prefetch) if len(self.time_prefetch) >0 else 0,
                       ]
        sum_total_times = np.sum(total_times)

        total_prop = {'time_perf_train':np.sum(self.time_perf_train)/sum_total_times if len(self.time_perf_train) >0 else 0,
                       'time_perf_load':np.sum(self.time_perf_load)/sum_total_times if len(self.time_perf_load) >0 else 0,
                       'time_perf_forward':np.sum(self.time_perf_forward)/sum_total_times if len(self.time_perf_forward) >0 else 0,
                       'time_perf_backward':np.sum(self.time_perf_backward)/sum_total_times if len(self.time_perf_backward) >0 else 0,
                       'time_plotting':np.sum(self.time_plotting)/sum_total_times if len(self.time_plotting) >0 else 0,
                       'time_saving_model':np.sum(self.time_saving_model)/sum_total_times if len(self.time_saving_model) >0 else 0,
                       'time_tracking_pi':np.sum(self.time_tracking_pi)/sum_total_times if len(self.time_tracking_pi) >0 else 0,
                       'time_scheduler':np.sum(self.time_scheduler)/sum_total_times if len(self.time_scheduler) >0 else 0,
                       'time_prefetch':np.sum(self.time_prefetch)/sum_total_times if len(self.time_prefetch) >0 else 0,
                       }

        """
        total_names = ['Loading','Forward','Backward','Plotting','CheckPoint Saving','Tracking PI','Update Scheduler','Read all data on GPU']
        for time,name in zip(total_times,total_names):
            prop = time/np.sum(total_times)
            print(f"Proportion of time consumed for {name}: {'{:.1%}'.format(prop)}")
        """


        if self.stop_proc and self.start_proc: print(">>> Training complete in: " + str(self.stop_proc - self.start_proc))
        if len(self.time_perf_train) > 1: print(">>> Training ({:.2%}) performance time: min {:.2f} avg {:.2e} seconds (+/- {:.2e})".format(total_prop['time_perf_train'],np.min(self.time_perf_train[1:]), np.median(self.time_perf_train[1:]), np.std(self.time_perf_train[1:])))
        if len(self.time_perf_train) == 1: print(">>> Training ({:.2%}) performance time: min {:.2f}".format(total_prop['time_perf_train'],np.min(self.time_perf_train[:])))
        if len(self.time_perf_load) > 1: print(">>> Loading ({:.2%}) performance time: min {:.2f} avg {:.2e} seconds (+/- {:.2e})".format(total_prop['time_perf_load'],np.min(self.time_perf_load[1:]), np.mean(self.time_perf_load[1:]), np.std(self.time_perf_load[1:])))
        if len(self.time_perf_load) == 1: print(">>> Loading ({:.2%}) performance time: min {:.2f}".format(total_prop['time_perf_load'],np.min(self.time_perf_load[:])))

        dic_correspondance = {'time_perf_forward': 'Forward',
                              'time_perf_backward': 'Backward',
                              'time_plotting': 'Plotting',
                              'time_saving_model': 'Saving',
                              'time_tracking_pi': 'PI-tracking',
                              'time_scheduler': 'Scheduler update'}
        
        for attr, name in dic_correspondance.items():
            if (len(getattr(self, attr)) > 1): 
                print(f">>> {name}  ({'{:.2%}'.format(total_prop[attr])}) performance time: {'{:.2e}'.format(np.mean(getattr(self, attr)[1:]))} seconds (+/- {'{:.2e}'.format(np.std(getattr(self, attr)[1:]))})")
            if (len(getattr(self, attr)) == 1):
                print(f">>> {name}  ({'{:.2%}'.format(total_prop[attr])}) performance time: {'{:.2e}'.format(getattr(self, attr)[0])} seconds")


        """
        if len(self.time_perf_forward) > 0: print(">>> Forward performance time: {} seconds (+/- {})".format(np.mean(self.time_perf_forward[1:]), np.std(self.time_perf_forward[1:])))
        if len(self.time_perf_backward) > 0: print(">>> Backward performance time: {} seconds (+/- {})".format(np.mean(self.time_perf_backward[1:]), np.std(self.time_perf_backward[1:])))
        if len(self.time_plotting) > 0: print(">>> Plotting performance time: {} seconds (+/- {})".format(np.mean(self.time_plotting[1:]), np.std(self.time_plotting[1:])))
        if len(self.time_saving_model) > 0: print(">>> Saving performance time: {} seconds (+/- {})".format(np.mean(self.time_saving_model[1:]), np.std(self.time_saving_model[1:])))
        if len(self.time_tracking_pi) > 0: print(">>> PI-tracking performance time: {} seconds (+/- {})".format(np.mean(self.time_tracking_pi[1:]), np.std(self.time_tracking_pi[1:])))
        if len(self.time_scheduler) > 0: print(">>> Scheduler-update performance time: {} seconds (+/- {})".format(np.mean(self.time_scheduler[1:]), np.std(self.time_scheduler[1:])))
        """

        #if len(self.power) > 0: print(">>> Peak Power during training: {} W)".format(np.max(self.power)))
        if self.val_time: print(">>> Validation time: {}".format(self.val_time))

        #if len(self.time_perf_train) > 0 and len(self.time_perf_load) > 0: 
        #    print(">>> Sortie trace #####################################" )
        #    print(">>>JSON", json.dumps({'GPU process - Forward/Backward':self.time_perf_train, 'CPU process - Dataloader':self.time_perf_load}))
                