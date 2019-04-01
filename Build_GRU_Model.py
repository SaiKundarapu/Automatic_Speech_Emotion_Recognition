import joblib, sys
from Data_Preprocessing import dataPreprocess
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import GRU
from keras.layers import Dropout
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV

class GRU_Classifier:

    def __init__(self):
        self.__X_train=None
        self.__y_train=None
        self.__X_test=None
        self.__y_test=None
    
    def __load_data(self,joblib_path):
        try:
            return joblib.load(joblib_path)
        except:
            sys.stderr.write("-----------Invalid saved file provided--------------------\n\n")
            input("Press any key to exit\n")
            sys.exit(-1)
            
    def buildClassifier(self,optimizer):    
        classifier = Sequential()
    
        classifier.add(GRU(units = 100, return_sequences = True, input_shape = (self.__X_train.shape[1],26)))
        classifier.add(Dropout(0.2))
        
        classifier.add(GRU(units = 100, return_sequences = True))
        classifier.add(Dropout(0.2))
        
        classifier.add(GRU(units = 100))
        classifier.add(Dropout(0.2))
        
        classifier.add(Dense(units=8,activation="softmax"))
        
        classifier.compile(optimizer = optimizer, loss = 'sparse_categorical_crossentropy',metrics=['accuracy'])    
        return classifier

    def __build_classifier(self):
    
        classifier = Sequential()
    
        classifier.add(GRU(units = 100, return_sequences = True, input_shape = (self.__X_train.shape[1],26)))
        classifier.add(Dropout(0.2))
        
        classifier.add(GRU(units = 100, return_sequences = True))
        classifier.add(Dropout(0.2))
        
        classifier.add(GRU(units = 100))
        classifier.add(Dropout(0.2))
        
        classifier.add(Dense(units=8,activation="softmax"))
        
        classifier.compile(optimizer = 'rmsprop', loss = 'sparse_categorical_crossentropy',metrics=['accuracy'])    
        return classifier

    def train_classifier(self,path,timestep,n_epochs,n_batches):
        
        preprocess=dataPreprocess(path,timestep)       
        self.__X_train, self.__X_test, self.__y_train, self.__y_test = preprocess.process()
        classifier = self.__build_classifier()
        gruhistory=classifier.fit(self.__X_train, self.__y_train, epochs = n_epochs, batch_size = n_batches, validation_data=(self.__X_test,self.__y_test))
        
        plt.plot(gruhistory.history['loss'])
        plt.plot(gruhistory.history['val_loss'])
        plt.title('model loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(['train', 'test'], loc='upper left')
        plt.show()
        
        plt.plot(gruhistory.history['acc'])
        plt.plot(gruhistory.history['val_acc'])
        plt.title('model accuracy')
        plt.ylabel('acc')
        plt.xlabel('epoch')
        plt.legend(['train', 'test'], loc='upper left')
        plt.show()
        
        return classifier
        
    def test_classifier(self,classifier):
        y_pred = classifier.predict_classes(self.__X_test)
        
        report = classification_report(self.__y_test, y_pred)
        
        matrix = confusion_matrix(self.__y_test, y_pred)
        return report,matrix

    # Saving the Model
    def saveModel(self,classifier,model_path):
        classifier.save(model_path)
        print("\n-------------------Model Saved Successfully------------------\n\n")
        
    def evaluateModel(self,path,timestep,n_epochs,n_batches,n_folds):
        # Evaluating the GRU Model
        preprocess=dataPreprocess(path,timestep)       
        self.__X_train, self.__X_test, self.__y_train, self.__y_test = preprocess.process()
        classifier = KerasClassifier(build_fn = self.__build_classifier, batch_size = n_batches, epochs = n_epochs)
        accuracies = cross_val_score(estimator = classifier, X = self.__X_train, y = self.__y_train, cv = n_folds)
        mean = accuracies.mean()
        variance = accuracies.std()
        print("Mean Accuracy: ",mean)
        print("Accuracy Variance: ",variance)
    
    def tuningModel(self,path,timestep,n_epochs,n_batches,n_folds,parameters={'batch_size': [16, 32,1000], 'epochs': [50, 100], 'optimizer': ['adam', 'rmsprop']}):        
        # Tuning the GRU Model
        preprocess=dataPreprocess(path,timestep)       
        self.__X_train, self.__X_test, self.__y_train, self.__y_test = preprocess.process()
        classifier = KerasClassifier(build_fn = self.buildClassifier)
        
        grid_search = GridSearchCV(estimator = classifier,
                                   param_grid = parameters,
                                   scoring = 'accuracy',
                                   cv = n_folds)
        grid_search = grid_search.fit(self.__X_train, self.__y_train)
        best_parameters = grid_search.best_params_
        best_accuracy = grid_search.best_score_
        print("Best Accuracy: ",best_accuracy)
        print("Best Parameter Setting: ",best_parameters)
        
        y_pred = grid_search.predict(self.__X_test)
        
        report = classification_report(self.__y_test, y_pred)
        
        matrix = confusion_matrix(self.__y_test, y_pred)
        
        return report,matrix