import sys,os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import tkinter
from tkinter import filedialog
from tkinter import messagebox
from Feature_Extraction import featureExtraction
from Data_Preprocessing import dataPreprocess
from Build_GRU_Model import GRU_Classifier 
from Audio_Recorder import record
from Predict_Audio_Emotion import livePredictions

def openFile(ext):
    root=tkinter.Tk()
    root.withdraw() # Close the root window
    if(ext==".joblib"):
        ftype=(("JOBLIB File","*.joblib"),)
    elif(ext==".h5"):
        ftype=(("HDF5 File","*.h5"),)
    else:
        ftype=(("WAV File","*.wav"),)
    in_path = filedialog.askopenfilename(initialdir = os.getcwd(),defaultextension=ext,title = "Select the "+ext+" file",parent=root,filetypes = ftype)
    return in_path

def openDirectory():
    root=tkinter.Tk()
    root.withdraw() # Close the root window
    in_path = filedialog.askdirectory(mustexist=True,initialdir=os.getcwd(),title='Please Select The Dataset Folder',parent=root)
    return in_path

def saveFile(ext):
    root=tkinter.Tk()
    root.withdraw() # Close the root window
    if(ext==".joblib"):
        ftype=(("JOBLIB File","*.joblib"),)
    elif(ext==".h5"):
        ftype=(("HDF5 File","*.h5"),)
    else:
        ftype=(("WAV File","*.wav"),)
    in_path = filedialog.asksaveasfilename(confirmoverwrite=True,defaultextension=ext,initialdir=os.getcwd(),title = "Save as file "+ext,parent=root,filetypes = ftype)
    return in_path

def menuExtract():
    path=openDirectory()
    features = featureExtraction(path)
    data=features.extract()
    print(data[:10])
    menuSave(features)
    
def menuSave(features):
    loop=True
    while(loop):
        choice=input("Do you want save the features data extracted?[Y/N]\n")
        if(choice in ['Y','y']):
            path=saveFile(".joblib")
            features.save(path)
            loop=False
        elif(choice in ['N','n']):
            loop=False
        else:
            print("\nPlease a valid input\n")

def menuPreprocess():
    datapath=openFile(".joblib")
    timestep=int(input("Enter TimeStep period\n"))
    print("\n--------------PreProcessing Initiated-----------------")
    preprocess=dataPreprocess(datapath,timestep)
    print("\n--------------PreProcessing Sucessfully Completed-----------------")
    X_train, X_test, y_train, y_test=preprocess.process()
    print("X_train:",X_train.shape)
    print("X_test:",X_test.shape)
    print("y_train:",y_train.shape)
    print("y_test:",X_test.shape)
    
def menuTrainTest():
    datapath=openFile(".joblib")
    timestep=int(input("Enter TimeStep period\n"))
    n_epochs=int(input("Enter No of epochs\n"))
    n_batches=int(input("Enter batch size\n"))
    model=GRU_Classifier()
    classifier=model.train_classifier(datapath,timestep,n_epochs,n_batches)
    report,matrix=model.test_classifier(classifier)
    labels = ['netural', 'calm', 'happy','sad','angry','fear','disgust','suprise']
    cm1 = pd.DataFrame(matrix, index = labels, columns = labels)
    plt.figure(figsize = (6, 4))
    sns.heatmap(cm1, annot = True, cbar = False, fmt = 'g')
    plt.ylabel('Actual values')
    plt.xlabel('Predicted values')
    plt.show()
    print(report)
    menuSaveModel(model,classifier)

def menuSaveModel(model,classifier):
    loop=True
    while(loop):
        choice=input("Do you want save the Model?[Y/N]\n")
        if(choice in ['Y','y']):
            path=saveFile(".h5")
            model.saveModel(classifier,path)
            loop=False
        elif(choice in ['N','n']):
            loop=False
        else:
            print("\nPlease a valid input\n")
        
def menuEvaluate():
    datapath=openFile(".joblib")
    timestep=int(input("Enter TimeStep period\n"))
    n_epochs=int(input("Enter No of epochs\n"))
    n_batches=int(input("Enter batch size\n"))
    n_folds=int(input("Enter the No of folds in cross fold validation\n"))
    model=GRU_Classifier()
    model.evaluateModel(datapath,timestep,n_epochs,n_batches,n_folds)

def menuTune():
    datapath=openFile(".joblib")
    timestep=int(input("Enter TimeStep period\n"))
    n_epochs=int(input("Enter No of epochs\n"))
    n_batches=int(input("Enter batch size\n"))
    n_folds=int(input("Enter the No of folds in cross fold validation\n"))
    model=GRU_Classifier()
    report,matrix=model.tuningModel(datapath,timestep,n_epochs,n_batches,n_folds)
    labels = ['netural', 'calm', 'happy','sad','angry','fear','disgust','suprise']
    cm1 = pd.DataFrame(matrix, index = labels, columns = labels)
    plt.figure(figsize = (6, 4))
    sns.heatmap(cm1, annot = True, cbar = False, fmt = 'g')
    plt.ylabel('Actual values')
    plt.xlabel('Predicted values')
    plt.show()
    print(report)
    
def menuRecord():
    savepath=saveFile(".wav")
    no_of_sec=int(input("Enter No of seconds to record\n"))
    input("Press any key to start recording\n")
    record(no_of_sec,savepath)

def menuPredict():
    modelpath=openFile(".h5")
    audiopath=openFile(".wav")
    predict=livePredictions(modelpath,audiopath)
    predict.load_model()
    messagebox.showinfo("Information",predict.makepredictions())
    
def main_menu():    
    loop=True
    while loop:
        subloop=True
        print(30 * "-" , "AUDIO DATA ANALYSIS" , 30 * "-")
        print("1. FeatureExtraction")
        print("2. DataPreprocessing")
        print("3. Build GRU Model")
        print("4. Record a live sample")
        print("5. Live Prediction")
        print("6. EXIT")
        print(80 * "-")
        
        choice=int(input("Welcome,\nPlease Enter your choice [1-6]:\n"))
         
        if choice == 1:
            menuExtract()
        elif choice == 2:
            menuPreprocess()            
        elif choice==3:
            while subloop:
                print(30 * "-" , "Build GRU Model" , 30 * "-")
                print("1. Train the Classifier")
                print("2. Evaluate the Classifier")
                print("3. Tune the Classifier")
                print("4. Back to MainMenu")
                print(80 * "-")
                subchoice=int(input("Welcome back,\nPlease Enter your choice [1-5]:\n"))
                if subchoice==1:
                    menuTrainTest()
                    subloop=False
                elif subchoice==2:
                    menuEvaluate()
                    subloop=False
                elif subchoice==3:
                    menuTune()
                    subloop=False
                elif subchoice==4:
                    main_menu()
                else:
                    print("Please Enter a valid choice [1-5]:")
        elif choice==4:
            menuRecord()
        elif choice==5:
            menuPredict()
        elif choice==6:
            sys.exit(0)
        else:
            print("Please Enter a valid choice [1-6]:")
            
# Main Program
if __name__ == "__main__":
    # Launch main menu
    main_menu()