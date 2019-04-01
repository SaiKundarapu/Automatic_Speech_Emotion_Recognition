import joblib
import librosa
import os,sys
import numpy as np

class featureExtraction:

    def __init__(self, data_path):
        self.data=list()
        self.data_path=data_path
        
    def __labelEncoder(self,filename):
        if(filename[-11:-4]=='neutral' or filename[7:8]=='1'):
            return 0;
        if(filename[-8:-4]=='calm' or filename[7:8]=='2'):
            return 1;
        if(filename[-9:-4]=='happy' or filename[7:8]=='3'):
            return 2;
        if(filename[-7:-4]=='sad' or filename[7:8]=='4'):
            return 3;
        if(filename[-9:-4]=='angry' or filename[7:8]=='5'):
            return 4;
        if(filename[-8:-4]=='fear' or filename[7:8]=='6'):
            return 5;
        if(filename[-11:-4]=='disgust' or filename[7:8]=='7'):
            return 6;
        if(filename[-6:-4]=='ps' or filename[7:8]=='8'):
            return 7;

    def extract(self):
        print(".....Extracting....")
        try:
            for subdir, dirs, files in os.walk(self.data_path):
              for file in files:
                  try:
                   # The instruction below converts the labels (from 0to 7)
                    label = self.__labelEncoder(file)
                    #Load librosa array, obtain filter bank energies from 26 filters
                    X, sample_rate = librosa.load(os.path.join(subdir,file), res_type='kaiser_fast')
                    fbanks = np.mean(librosa.feature.melspectrogram(y=X, sr=sample_rate, n_mels=26).T,axis=0) 
                    #store the file information(features, label) in a new array
                    file_info = fbanks, label
                    self.data.append(file_info)
                  # If the file is not valid, skip it
                  except ValueError:
                    continue
        except:
            sys.stderr.write("-----------Invalid dataset path provided--------------------\n\n")
            input("Press any key to exit\n")
            sys.exit(-1)
        print(".....Successfully Extracted......")
        return self.data

# Saving joblib files to not run them again
    def save(self,path):
        try:
            print(".....Saving......")
            joblib.dump(self.data, os.path.join(path))
            print(".....Successfully Saved......")
        except:
            sys.stderr.write("-----------Invalid save file path provided--------------------\n\n")
            input("Press any key to exit\n")
            sys.exit(-1)