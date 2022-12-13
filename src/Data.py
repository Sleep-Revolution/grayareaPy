from scipy import signal
from scipy import io as scio
import time
import os
import pandas as pd
import seaborn as sn
import numpy as np
import mne
import re
import pickle
from statsmodels.stats.inter_rater import fleiss_kappa,aggregate_raters
from datetime import datetime, timedelta
from matplotlib import pyplot as plt
from ipywidgets import IntProgress
from IPython.display import display
import sklearn.metrics as skM


def fmt(ticks):
    if all(isinstance(h, str) for h in ticks):
        return "%s"
    return ("%.d" if all(float(h).is_integer() for h in ticks) else
            "%.2f")

class DataSet:
    def __init__(self,path_data,times_stamps = 0.005,epoch_time = 30):
        self.SCORE_DICT = {
            'Wake': 0.,
            'N1': 1.,
            'N2': 2.,
            'N3': 3.,
            'REM': 4.}
        self.path_data=path_data
        if os.path.exists(os.path.join(self.path_data,'psg')):
            pathhypnoInfo =  os.path.join(self.path_data,'psg/padded_hypnograms/hypnogram_key.csv')
            if os.path.exists(pathhypnoInfo):
                self.hypnoInfo = pd.read_csv(pathhypnoInfo,sep=';',header=None)
        readme = os.path.join(self.path_data,'psg/readme.txt')
        if os.path.exists(readme):
            with open(readme, 'r') as text:
                self.readme = text.read()
        if os.path.exists(os.path.join(self.path_data,'psg')):
            self.pathedf = os.path.join(self.path_data,'psg/edf_recordings/')
            
        if os.path.exists(os.path.join(self.path_data,'psg')):
            self.pathhypno = os.path.join(self.path_data,'psg/padded_hypnograms/')
            
        self.times_stamps = times_stamps
        self.epoch_time = epoch_time
        
        if os.path.exists(os.path.join(self.path_data,'psg')):
            self.allEdf = os.listdir(self.pathedf)
            self.allEdf.sort()
            self.allHypno = os.listdir(self.pathhypno)
            self.allHypno.sort(key=self.FindIntInStr)
            self.allIndHypno = [i for i in range(len(self.allHypno)) 
                                if len(list(map(int, re.findall(r'\d+',self.allHypno[i]))))==0]
            if len(self.allIndHypno) > 0:
                for i in self.allIndHypno:
                    del self.allHypno[i]
            self.edf_partsort = [int(s) for file in self.allEdf for s in re.findall(r'\d+', file)]
            self.allPart = list(range(1,len(self.allHypno)))
            self.vecNumToLabel = np.vectorize(self.NumToLabel)
        
    @property
    def DataEdf(self):
        return self.DataEdf_
    
    @DataEdf.setter
    def DataEdf(self,newdata):
        self.DataEdf_ = newdata
    
    @property
    def DataHypno(self):
        return self.DataHypno_
    
    @DataHypno.setter
    def DataHypno(self,newdata):
        self.DataHypno_ = newdata
    
    def GetDataPart(self,participant,**kwargs):
        self.prev_hz = kwargs.get("prev_hz", 200)
        self.next_hz = kwargs.get("next_hz", 100)
        self.lowcut = kwargs.get("lowcut", 0.003)
        self.highcut = kwargs.get("highcut", 0.4)
        self.order = kwargs.get("order", 100)
        self.convuV = kwargs.get("VTouV", True)
        self.verbose = kwargs.get("verbose", 0)
        self.filtering = kwargs.get("filtering", False)
        if self.convuV:
            TouV = 1e6
        else:
            TouV = 1
        dictdata = {"DataHypno":[],"DataEdf":{"TimeHHMMSS":[],"TimeSS":[],"Edf":[]}}
        
        i = participant
        fileEDF = os.path.join(self.pathedf,self.allEdf[i])
        fileHYPNO = os.path.join(self.pathhypno,self.allHypno[i])
        if self.verbose>0:
            print("Load Part : %s, file EDF: %s, file HYPNO: %s" % (i,self.allEdf[i],self.allHypno[i]))
        data_scorer = pd.read_csv(fileHYPNO,sep=';')
        edf = mne.io.read_raw_edf(os.path.join(fileEDF,os.listdir(fileEDF)[0]),verbose=self.verbose)

        data, times = edf[:]
        indcha = [i for i in range(len(edf.ch_names)) if edf.ch_names[i] == "C4-M1"]
        eegf4m1 = data[indcha,:]

        today = datetime.today()
        s = edf.info['meas_date'].strftime("%H:%M:%S")
        edfstart = datetime.combine(today, datetime.strptime(s, '%H:%M:%S').time())
        edfend= edfstart+timedelta(seconds=eegf4m1.shape[1]*self.times_stamps)

        scostart = data_scorer.iloc[0,1]
        scoend= data_scorer.iloc[data_scorer.shape[0]-1,1]

        TimeFromStart = np.arange(edfstart, edfend, timedelta(seconds=self.times_stamps)).astype(datetime)
        TimeFromStart = np.array([i.strftime("%H:%M:%S") for i in TimeFromStart])

        datetime_obj = datetime.strptime(scostart, "%H:%M:%S")
        edfstart = datetime_obj.replace(year=edfend.year, month=edfend.month, day=edfend.day)

        datetime_obj = datetime.strptime(scoend, "%H:%M:%S")
        scoend = datetime_obj.replace(year=edfend.year, month=edfend.month, day=edfend.day)
        datetime_obj = scoend+timedelta(seconds=self.epoch_time)
        
        if (datetime_obj).strftime("%H:%M:%S") !=  edfend.strftime("%H:%M:%S"):
            scoend = scoend-timedelta(seconds=self.epoch_time)
            edfend = scoend+timedelta(seconds=self.epoch_time)
        while edfend.strftime("%H:%M:%S") not in TimeFromStart:
            scoend = scoend-timedelta(seconds=self.epoch_time)
            edfend = scoend+timedelta(seconds=self.epoch_time)
        
        indToStart = np.where(TimeFromStart==edfstart.strftime("%H:%M:%S"))[0][0]
        indToEnd = np.where(TimeFromStart==edfend.strftime("%H:%M:%S"))[0][0]

        TimeFromStart = TimeFromStart[indToStart:indToEnd]
        eegf4m1 = eegf4m1[:,indToStart:indToEnd]

        indToStart = data_scorer[data_scorer["Epoch starttime"]==edfstart.strftime("%H:%M:%S")].index[0]
        indToEnd = data_scorer[data_scorer["Epoch starttime"]==scoend.strftime("%H:%M:%S")].index[0]

        data_scorer = data_scorer.iloc[indToStart:(indToEnd+1),:]
        len43 = data_scorer.shape[0]
        Timearray = np.arange(0,(len43)*self.epoch_time,self.times_stamps)

        TimeFromStart_re = TimeFromStart.reshape(len43,int(self.epoch_time/self.times_stamps))
        Timearray_re = Timearray.reshape(len43,int(self.epoch_time/self.times_stamps))
        eegf4m1_re = eegf4m1.reshape(len43,int(self.epoch_time/self.times_stamps))
        
        assert data_scorer["Epoch starttime"].iloc[0] == TimeFromStart_re[0,0]
        assert data_scorer["Epoch starttime"].iloc[len43-1] == TimeFromStart_re[len43-1,0]
        assert data_scorer["Epoch starttime"].shape[0] == TimeFromStart_re.shape[0]
        
        if self.filtering:
            eegf4m1_re = np.array(list(map(lambda x: self.resample_signal(x,prev_hz=self.prev_hz,next_hz=self.next_hz), 
                                  eegf4m1_re)))
            eegf4m1_re = np.array(list(map(lambda x: self.bandpass_filter(x,self.lowcut ,self.highcut,self.order), 
                                  eegf4m1_re)))
            Timearray_re = np.array(list(map(lambda x: self.resample_signal(x,prev_hz=self.prev_hz,next_hz=self.next_hz), 
                      Timearray_re)))
        
        dictdata["DataHypno"] = data_scorer     
        dictdata["DataEdf"]["TimeHHMMSS"] = TimeFromStart_re
        dictdata["DataEdf"]["TimeSS"] = Timearray_re
        dictdata["DataEdf"]["Edf"] = eegf4m1_re
        return dictdata
        
    
    def LoadData(self,**kwargs):
        f = IntProgress(min=0, max=len(self.allEdf),description="File: ")
        display(f)
        alldata = {"participant":[],"Data":[]}
        
        for i in range(len(self.allHypno)):
            f.value += 1
            f.description="File: "+self.allHypno[i]
            alldata["participant"].append(i)
            alldata["Data"].append(self.GetDataPart(i,**kwargs))
            
        self.data = alldata
    
    def FindIntInStr(self,my_list):
        return list(map(int, re.findall(r'\d+', my_list)))

    def NumToLabel(self,a):
        return(self.hypnoInfo[self.hypnoInfo[1]==a][0].iloc[0])
    
    

    def check_signals(signal_names):
        # TODO: find eeg, eog and emg signals
        eeg = 0
        emg = 0
        eog = 0
        possible_eeg_names = set(['C4-A1', 'C4-M1', 'AF3-E3E4'])
        possible_emg_names = set(['CHIN1-CHIN2', 'E1-M2', 'EMG.Frontalis-Ri'])
        possible_eog_names = set(['ROC-LOC', '1-2', 'E2-AFZ'])
        for i, s in enumerate([t['label'] for t in signal_names]):
            if s in possible_eeg_names:
                eeg = i
            if s in possible_emg_names:
                emg = i
            if s in possible_eog_names:
                eog = i
        return eeg, emg, eog

    def get_second_offset(self, df, startdate):
        # use information from event grid to cut away raw data before and after scored events
        first_measured = df.iloc[2][2]
        print((first_measured-startdate).total_seconds())
        return int((first_measured-startdate).total_seconds())

    def resample_signal(self, data, prev_hz, next_hz):
        new_data = signal.resample(data, int(len(data)/prev_hz*next_hz))
        return new_data

    def bandpass_filter(self, data, lowcut, highcut, order):
        # sos = signal.butter(order, [lowcut, highcut], btype='band', output='sos')
        b = signal.firwin(order, [lowcut, highcut], window='hamming', pass_zero=False)
        y = signal.filtfilt(b, 1, data)
        return y

    def MNEbandpass_filter(self, data, lowcut, highcut,verbose=False):
        y = mne.filter.filter_data(data, data.shape[1]/self.epoch_time,lowcut,highcut,verbose=verbose)
        return y

    
    def get_spectrograms(self, epochized):
        # get spectrograms for each segment, this is "X"
        # TODO tweak until identical(?) to MATLAB version
        megaZxx = np.empty((len(epochized), 29, 129))
        for i, e in enumerate(epochized):
            f, t, Zxx = signal.stft(e, fs=100, 
                                    window=signal.windows.general_hamming(200, 0.54), 
                                    nperseg=200, 
                                    noverlap=100, 
                                    nfft=256)
            # mlab specgram
            # f, t, Zxx = mlab.specgram(e, NFFT=256, Fs=100, noverlap=100, window=np.hamming(200), mode='complex')
            # remove first and last column of spectrogram
            t = t[1:-1]
            Zxx = np.delete(Zxx, 0, 1)
            Zxx = np.delete(Zxx, -1, 1)
            Zxx = np.log10(np.abs(Zxx))
            megaZxx[i] = np.transpose(Zxx)
        # transposed_megaZxx = np.moveaxis(megaZxx, 0, 2)
        return megaZxx

    def numerize_labels(self,df):
        # turn event grid into both numerized list and 1 hot encoded matrix, the list is "label"
        scorings = []
        one_hot_scorings = []
        for r in df.itertuples():
            if r.Event in SCORE_DICT:
                # dirty method to skip header
                phase = r.Event
                dur = int(r.Duration)
                epochs = dur/30
                scorings += [SCORE_DICT[phase]]*int(epochs)

        # create 1 hot encoded matrix, this is "y"
        one_hot_scorings = []
        for i in range(len(scorings)):
            to_append = np.zeros(5)
            to_append[int(scorings[i])] = 1
            one_hot_scorings.append(to_append)
        # following actions perform label = label' and y = cell2mat(y') MATLAB functions
        # making one_hot_scoring the shape of size x 1 instead of 1 x size
        one_hot_scorings = np.concatenate(np.array(one_hot_scorings)[np.newaxis])
        # same for scorings
        scorings = np.transpose(np.array(scorings)[np.newaxis])

        print("shape of scorings, should be num x 1:", np.shape(scorings))
        print("shape of one_hot_scorings, should be num x 5:", np.shape(one_hot_scorings))
        return scorings, one_hot_scorings

    def write_to_mat(self,name, X, y, label):
        # write to mat file
        hdf5storage.write({'X': X, 'y': y, 'label': label}, '.', f'{name}.mat', matlab_compatible=True)
        pass

    def saveData(self,name):
        if os.path.isdir(os.path.join('save_data')):
            path_save = os.path.join('save_data')
        else:
            path_save = os.mkdir('save_data')
            path_save = os.path.join('save_data')
        
        path_name = os.path.join(path_save,name+'.pickle')
        i=0
        while os.path.exists(path_name):
            path_name = os.path.join(path_save,name+str(i)+'.pickle')
            i+=1
        with open(path_name, 'wb') as handle:
            pickle.dump(self.data, handle, protocol=pickle.HIGHEST_PROTOCOL)
            
    def loadDataPickle(self,path_name):
        with open(path_name, 'rb') as file:
            self.data = pickle.load(file)


if __name__ == "main":
    data10 = DataSet('/datasets/10x100/')
    data10.LoadData()
    data10.saveData("Data50HypnoC4M1")