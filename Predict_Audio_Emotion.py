import keras
import numpy as np
import librosa
import sys

class livePredictions:

    def __init__(self, path, file):

        self.path = path
        self.file = file

    def load_model(self):
        '''
        I am here to load you model.
        :param path: path to your h5 model.
        :return: summary of the model with the .summary() function.
        '''
        try:
            self.loaded_model = keras.models.load_model(self.path)
            return self.loaded_model
        except:
            sys.stderr.write("-----------Invalid saved file path provided--------------------\n\n")
            input("Press any key to exit\n")
            sys.exit(-1)
    def makepredictions(self):
        '''
        I am here to process the files and create your features.
        '''
        X,sample_rate = librosa.load(self.file, res_type='kaiser_fast')    
        features = np.mean(librosa.feature.melspectrogram(y=X, sr=sample_rate, n_mels=26).T,axis=0)
        x=list()
        for i in range(0,20):
            x.append(features)
        x=np.array(x)
        x=x.reshape(1,20,26)
        predictions = self.loaded_model.predict_classes(x)
        return "!!!!!!! Predicted Emotion is "+self.convertclasstoemotion(predictions)+"!!!!!!!!!!!!"

    def convertclasstoemotion(self, pred):
        '''
        I am here to convert the predictions (int) into human readable strings.
        '''
        self.pred  = pred

        if pred == 0:
            pred = "neutral"
            return pred
        elif pred == 1:
            pred = "calm"
            return pred
        elif pred == 2:
            pred = "happy"
            return pred
        elif pred == 3:
            pred = "sad"
            return pred
        elif pred == 4:
            pred = "angry"
            return pred
        elif pred == 5:
            pred = "fearful"
            return pred
        elif pred == 6:
            pred = "disgust"
            return pred
        elif pred == 7:
            pred = "surprised"
            return pred