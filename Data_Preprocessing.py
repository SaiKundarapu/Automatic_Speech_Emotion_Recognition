import numpy as np
import joblib
import sys
class dataPreprocess:

    def __init__(self,datapath,timestep,n_labels=8):

        self.datapath = datapath
        self.n_labels = n_labels
        self.timestep = timestep
        self.__label_dataset = list()
        self.__X = list()
        self.__y = list()
        
    def __load_data(self,data_path):
        try:
            return joblib.load(data_path)
        except:
            sys.stderr.write("-----------Invalid saved file path provided--------------------\n\n")
            input("Press any key to exit\n")
            sys.exit(-1)

    def __dataSeparation(self,data, label):
        label_data = list()
        for file_info in data:
            if(file_info[1] == label):
                label_data.append(file_info[0])
        return label_data

    def __dataReshape(self,data, label):
        X_data = list()
        y_data = list()
        for i in range(self.timestep, data.shape[0]):
            X_data.append(data[i-self.timestep:i, :])
            y_data.append(label)
        return X_data,y_data

    def __dataScaling(self,data):
        # Feature Scaling
        from sklearn.preprocessing import MinMaxScaler
        sc_data=MinMaxScaler()
        data=sc_data.fit_transform(data)
        return data

    def process(self):
        # Creating a data structure with separate label dataset
        data=self.__load_data(self.datapath)
        for i in range(0,self.n_labels):
            self.__label_dataset.append(self.__dataScaling(np.asarray(self.__dataSeparation(data,i))))
        
        # Creating a data structure with 20 timesteps and 1 output    
        for i in range(0,self.n_labels):
            temp=self.__dataReshape(self.__label_dataset[i],i)
            self.__X.extend(temp[0])
            self.__y.extend(temp[1])
            
        # Splitting the dataset into the Training set and Test set
        from sklearn.model_selection import train_test_split
        X_train, X_test, y_train, y_test = map(np.array, train_test_split(self.__X, self.__y, test_size=0.25, random_state=0))
        
        return X_train, X_test, y_train, y_test