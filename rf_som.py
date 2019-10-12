import pandas, numpy


import csv

from os import listdir
from os.path import isfile, join


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import io
import fnmatch
import os, inspect
import re


from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Dense, Embedding, LSTM, SpatialDropout1D, Dropout
from keras.utils.np_utils import to_categorical
from keras.preprocessing import text, sequence
from keras import layers, models, optimizers

from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.mixture import GaussianMixture
from sklearn.decomposition import PCA
from sklearn.mixture import BayesianGaussianMixture
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn import model_selection, preprocessing, linear_model, naive_bayes, metrics, svm, decomposition, ensemble

from sklearn.feature_selection import SelectKBest, mutual_info_classif, chi2, f_classif

from neupy import algorithms, utils

#mappa sempre sugli stessi neuroni un certo tipo di input
utils.reproducible()


def combina_colonne(lista):
    vett=np.asarray(lista)
    #print("vett:",vett)
    
    comb=[]
    for i in range(len(vett[0])):
        temp=[]
        for a in range(len(vett)):
            temp.append(vett[a][i])
        comb.append(temp)
    data=np.asarray(comb)
    #print("data:",data)
    #print("len_data:",len(data))
    
    
    
    #som con 20 neuroni
    GRID_HEIGHT = 1
    GRID_WIDTH = 20

    sofm = algorithms.SOFM(
        n_inputs=len(data[0]),
        features_grid=(GRID_HEIGHT, GRID_WIDTH),
        
        # Learning radius defines area within which we find
        # winning neuron neighbours. The higher the value
        # the more values we will be updated after each iteration.
        learning_radius=5,
        # Every 20 epochs learning radius will be reduced by 1.
        reduce_radius_after=50,

        step=0.4,
        std=1,
        #show_epoch=5
        shuffle_data=True,
        verbose=True,
        )

    sofm.train(data, epochs=500)
    clusters = sofm.predict(data).argmax(axis=1)
    #print("cluster:",clusters)
    #print("len_clusters:",len(clusters))

    
    
    '''
    colonna=[]
    for i in range(0,len(lista[0])):
        
        #sum=0
        for a in range(0,len(lista)):
            sum+=(2**a)*int(lista[a][i])
            
    
        colonna.append(str(sum))
    print("colonna:",colonna)
    exit(1)
    '''
    return clusters,sofm;

def calc_accuracy(image, image2, image_train,image_test,classe_train,classe_test):
    
    accuracy=0
    acv=0

	#ORIGINAL DATA
	
    X_image_train=np.asarray(image)
    #print("X_train",X_train)
    X_image_test=np.asarray(image2)

    # split X and y into training and test sets
    #X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=4)

    X_train=np.asarray(image_train)
    #print("X_train",X_train)
    X_test=np.asarray(image_test)
    #print("X_test",X_test)
    y_train=np.asarray(classe_train)
    #print("y_train",y_train)
    y_test=np.asarray(classe_test)
    #print("y_test",y_test)
	
    	
    my_xtick=str(X_train.shape[1])
	
	#PCA
    pca = PCA(n_components=X_train.shape[1])
    pca.fit(X_image_train)
    X_image_train_pca=pca.transform(X_image_train)
    X_image_test_pca=pca.transform(X_image_test)
	
	
	#INFOGAIN	
    ig = SelectKBest(mutual_info_classif, k=X_train.shape[1])
    ig.fit(X_image_train, y_train)
    X_image_train_ig = ig.transform(X_image_train)
    X_image_test_ig = ig.transform(X_image_test)
	
	#CHI2
    chi2_sel = SelectKBest(chi2, k=X_train.shape[1])
    chi2_sel.fit(X_image_train, y_train)
    X_image_train_chi2 = chi2_sel.transform(X_image_train)
    X_image_test_chi2 = chi2_sel.transform(X_image_test)
	
	#F_CLASSIF	
    fcl = SelectKBest(f_classif, k=X_train.shape[1])
    fcl.fit(X_image_train, y_train)
    X_image_train_fcl = fcl.transform(X_image_train)
    X_image_test_fcl = fcl.transform(X_image_test)


	# CLASSIFICATION
	
	
    # RI
    clf = RandomForestClassifier(n_estimators=100, max_depth=None,random_state=0)
    # fit the model with data
    clf.fit(X_train, y_train)
    # make prediction on the test set
    y_pred = clf.predict(X_test)
	
	#PCA
    clf_pca = RandomForestClassifier(n_estimators=100, max_depth=None,random_state=0)
    # fit the model with data
    clf_pca.fit(X_image_train_pca, y_train)
    # make prediction on the test set
    y_image_pred_pca = clf_pca.predict(X_image_test_pca)

	#INFOGAIN
    clf_ig = RandomForestClassifier(n_estimators=100, max_depth=None,random_state=0)
    # fit the model with data
    clf_ig.fit(X_image_train_ig, y_train)
    # make prediction on the test set
    y_image_pred_ig = clf_ig.predict(X_image_test_ig)
	

	#CHI2
    clf_chi2 = RandomForestClassifier(n_estimators=100, max_depth=None,random_state=0)
    # fit the model with data
    clf_chi2.fit(X_image_train_chi2, y_train)
    # make prediction on the test set
    y_image_pred_chi2 = clf_chi2.predict(X_image_test_chi2)
	
	#F_CLASSIF
    clf_fcl = RandomForestClassifier(n_estimators=100, max_depth=None,random_state=0)
    # fit the model with data
    clf_fcl.fit(X_image_train_fcl, y_train)
    # make prediction on the test set
    y_image_pred_fcl = clf_fcl.predict(X_image_test_fcl)


    # compute classification accuracy
	
	#RI
	
    #print("\nAccuracy (splitting training/test sets)")
    #print(metrics.accuracy_score(y_test, y_pred))
    accuracy=metrics.accuracy_score(y_test, y_pred)
    # Compute confusion matrix
    #cnf_matrix = confusion_matrix(y_test, y_image_pred)
    np.set_printoptions(precision=2)
    #print("\nConfusion matrix:")
    #print(cnf_matrix)
	
	#PCA
    accuracy_pca=metrics.accuracy_score(y_test, y_image_pred_pca)

	
	#INFOGAIN
    accuracy_ig=metrics.accuracy_score(y_test, y_image_pred_ig)
	
	#CHI2
    accuracy_chi2=metrics.accuracy_score(y_test, y_image_pred_chi2)
	
	#INFOGAIN
    accuracy_fcl=metrics.accuracy_score(y_test, y_image_pred_fcl)

    # 10-fold cross validation
    #scores = cross_val_score(clf, X_image_train, y_train, cv=2, scoring='accuracy')
    #print("\nAccuracy (10-fold cross validation):")
    #print(scores.mean())
    #acv=scores.mean()

    return accuracy,accuracy_pca,accuracy_ig,accuracy_chi2,accuracy_fcl,my_xtick;

'''
def calc_accuracy(image_train,image_test,classe_train,classe_test):
    
    accuracy=0
    acv=0
    # store feature matrix in "X"
    #X = np.asarray(list_bigtest[i])

    # store class vector in "y"
    #y = np.asarray(classi)

    #print("X:",X)

    #print("y:",y)

    
    # print the shape of iris data
    #print("bigtest data dimensions (X):")
    #print(X.shape)



    # split X and y into training and test sets
    #X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=4)

    X_train=np.asarray(image_train)
    #print("X_train",X_train)
    X_test=np.asarray(image_test)
    #print("X_test",X_test)
    y_train=np.asarray(classe_train)
    #print("y_train",y_train)
    y_test=np.asarray(classe_test)
    #print("y_test",y_test)
	
    #print the shapes of the new X objects
    #print("\nTraining set dimensions (X_train):")
    #print(X_train.shape)
    #print("\nTest set dimensions (X_test):")
    #print(X_test.shape)

    #print the shapes of the new y objects
    #print("\nTraining set dimensions (y_train):")
    #print(y_train.shape)
    #print("\nTest set dimensions (y_test):")
    #print(y_test.shape)

    # instantiate k-Nearest Neighbors, k=3 (n_neighbors=3)
    clf = MultinomialNB()
    # fit the model with data
    clf.fit(X_train, y_train)

    # make prediction on the test set
    y_pred = clf.predict(X_test)

    # compute classification accuracy
    #print("\nAccuracy (splitting training/test sets)")
    #print(metrics.accuracy_score(y_test, y_pred))
    accuracy=metrics.accuracy_score(y_test, y_pred)
    # Compute confusion matrix
    cnf_matrix = confusion_matrix(y_test, y_pred)
    np.set_printoptions(precision=2)
    #print("\nConfusion matrix:")
    #print(cnf_matrix)

    # 10-fold cross validation
    scores = cross_val_score(clf, X_train, y_train, cv=2, scoring='accuracy')
    #print("\nAccuracy (10-fold cross validation):")
    #print(scores.mean())
    acv=scores.mean()

    return accuracy,acv;
'''


def test_txt_to_list(file):
    with open("./test/%s" % file) as f:
        b=f.readlines()
        for i in range(len(b)):
            b[i]=(b[i].rstrip('\n').split(' '))
            for a in range(len(b[i])):
                b[i][a]=int(b[i][a])
    f.close()
    return b
    
def train_txt_to_list(file):
    with open("./training/%s" % file) as f:
        b=f.readlines()
        for i in range(len(b)):
            b[i]=(b[i].rstrip('\n').split(' '))
            for a in range(len(b[i])):
                b[i][a]=int(b[i][a])
    f.close()
    return b

'''

reader  = csv.reader(f,delimiter=";")
    image2,classi2=[],[]
    for row in reader:
        #print(row)
        #print(len(row))
        app=[]
        for c in range(0, len(row)):
            #print(row[c])
            if c == len(row)-1:
                classi2.append(int(row[c]))
            else:
                app.append(int(row[c]))
        image2.append(app)
f.close()
'''

# ------------------- Step 1 -----------------------
#leggo limmagine di partena Bigtest1_trasl_data.txt
#Define 2 list, for the classi and for the image (__label__2 and__label__1)


classi, image ,app= [], [],[]

#Read the file with utf-8 encoding
f =open('Bigtest1_trasl_data.txt', encoding="utf-8")
data=f.read()
f.close()
#print(data)
app=data.split("\n")
for i in range(0,len(app)):
    app2=[]
    for a in range(0,len(app[i])):
        app2.append(int(app[i][a]))
    image.append(app2)



#print("image:",image)
#print('len image: ',len(image))



f =open('classi.txt', encoding="utf-8")
data=f.read()

f.close()
classi = data.split("\n")
for i in range(len(classi)):
    classi[i]=int(classi[i])

#print("classi:",classi)
#print('len classi:',len(classi))




#scompongo image in colonne
column=[]
#print("pixel image:",len(image[0]))
#print("len image:",len(image))
for i in range(0,len(image[0])):
    #print("index:",i)
    #print("-------")
    tmp=[]
    for a in range(0,len(image)):
        #print(image[a][i])
        tmp.append(image[a][i])
    #print(tmp)
    column.append(tmp)
#print("colonne Bigtest1_trasl_data.txt:\n",column)

lista_files_variabili = [f for f in listdir("./variabili/") if isfile(join("variabili/", f))]

#print("lista_files_variabili:",lista_files_variabili)
#print("file nella cartella variabili:",len(lista_files_variabili))



#leggo un file e scompongo i dati nella lista di appoggio l
#che inserirò in una lista ll dove saranno salvati tutti i dati letti
l,ll=[],[]


for i in range(len(lista_files_variabili)):
    #print("#######",i,"######")
    with open("./variabili/variabili_"+str(i+2)+".txt") as f:
        data=f.readline()
        #print(data)
        l= list(data.rstrip('\n').split(' '))
        #print(l)
    f.close()
    ll.append(l)

#in ll ho una lista di liste nella quale in ogni cella ho tutti i file variabili_x.txt
#print("lista ll:",ll)


#trovo il mio path
my_path=os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))


#print("\n--------------------step 3--------------------------\n")

print("\napro file.csv\n")

with open("./Bigtest2.csv") as f:
    reader  = csv.reader(f,delimiter=";")
    image2,classi2=[],[]
    for row in reader:
        
        app=[]
        for c in range(0, len(row)):
            #print(row[c])
            if c == len(row)-1:
                classi2.append(int(row[c]))
            else:
                app.append(int(row[c]))
        image2.append(app)
f.close()

#print("image2:",image2)
#print("classi2:",classi2)
#print('len image2:',len(image2))
#print('len classi2:',len(classi2))

#print("ll:",ll)
#print("len_ll:",len(ll))

#scompongo image2 in colonne
column2=[]
#print("pixel image:",len(image[0]))
#print("len image:",len(image))
for i in range(0,len(image2[0])):
    #print("index:",i)
    #print("-------")
    tmp=[]
    for a in range(len(image2)):
        #print(image[a][i])
        tmp.append(image2[a][i])
    #print(tmp)
    column2.append(tmp)
#print("colonne Bigtest2.csv:\n",column2)

print("\nimmagini per il test calcolate")



#print("image:",image)
#print('len image:',len(image))
#print("classi:",classi)
#print('len classi:',len(classi))

#print("image2:",image2)
#print('len image2:',len(image2))
#print("classi2:",classi2)
#print('len classi2:',len(classi2))

'''
iterazioni=list(range(0,len(lista_files_variabili)+1))

accuracy=[]
acv=[]


tmp_acc, tmp_acv =calc_accuracy(image,image2,classi,classi2)
accuracy.append(tmp_acc)
acv.append(tmp_acv)
'''

iterazioni=list(range(0,len(lista_files_variabili)+1))

accuracy=[]
#acv=[]
accuracy_pca=[]
accuracy_ig=[]
accuracy_chi2=[]
accuracy_fcl=[]
my_xticks=[]


tmp_acc, tmp_acc_pca, tmp_acc_ig, tmp_acc_chi2, tmp_acc_fcl, tmp_xtick =calc_accuracy(image,image2,image,image2,classi,classi2)
accuracy.append(tmp_acc)
accuracy_pca.append(tmp_acc_pca)
accuracy_ig.append(tmp_acc_ig)
accuracy_chi2.append(tmp_acc_chi2)
accuracy_fcl.append(tmp_acc_fcl)
#acv.append(tmp_acv)
my_xticks.append(tmp_xtick)

#print("accuracy:",accuracy)
#print("acv:",acv)


lista_files_training = [f for f in listdir("./training/") if isfile(join("training/", f))]
lista_files_test = [f for f in listdir("./test/") if isfile(join("test/", f))]

#print("lista_files_training:",lista_files_training)
#print("file nella cartella train:",len(lista_files_training))

#print("lista_files_test:",lista_files_test)
#print("file nella cartella test:",len(lista_files_test))

#print("classi:",classi)
#print("len_classi:",len(classi))

#print("classi2:",classi2)
#print("len_classi2:",len(classi2))


image_train=[]
image_test=[]
'''
for i in range(len(lista_files_training)):
    print("iterazione:",i+2)
    image_train=train_txt_to_list("train"+str(i+2)+".txt")
    image_test=test_txt_to_list("test"+str(i+2)+".txt")
    #print("image_train:",image_train)
    #print("image_test:",image_test)
    #print("len_image_train:",len(image_train))
    #print("len_image_test:",len(image_test))
    
    tmp_acc, tmp_acv =calc_accuracy(image_train,image_test,classi,classi2)
    accuracy.append(tmp_acc)
    acv.append(tmp_acv)

print("accuracy:",accuracy)
print("acv:",acv)

print("len accuracy:",len(accuracy))
print("len acv:",len(acv))

# plot the relationship between K and testing accuracy
# plt.plot(x_axis, y_axis)
plt.plot(iterazioni, accuracy)
plt.xlabel('ITERAZIONI')
plt.ylabel('Testing Accuracy')
plt.show()


plt.plot(iterazioni, acv)
plt.xlabel('ITERAZIONI')
plt.ylabel('Testing Acv')
plt.show()
'''

for i in range(len(lista_files_training)):
    print("iterazione:",i+2)
    image_train=train_txt_to_list("train"+str(i+2)+".txt")
    image_test=test_txt_to_list("test"+str(i+2)+".txt")
    #print("image_train:",image_train)
    #print("image_test:",image_test)
    #print("len_image_train:",len(image_train))
    #print("len_image_test:",len(image_test))
    
    tmp_acc, tmp_acc_pca, tmp_acc_ig, tmp_acc_chi2, tmp_acc_fcl, tmp_xtick =calc_accuracy(image,image2,image_train,image_test,classi,classi2)
    accuracy.append(tmp_acc)
    accuracy_pca.append(tmp_acc_pca)
    accuracy_ig.append(tmp_acc_ig)
    accuracy_chi2.append(tmp_acc_chi2)
    accuracy_fcl.append(tmp_acc_fcl)
    #acv.append(tmp_acv)
    my_xticks.append(tmp_xtick)

print("accuracy:",accuracy)
#print("acv:",acv)
print("accuracy PCA:",accuracy_pca)
print("accuracy IG:",accuracy_ig)
print("accuracy CHI2:",accuracy_chi2)
print("accuracy FCL:",accuracy_fcl)

print("len accuracy:",len(accuracy))
#print("len acv:",len(acv))

# plot the relationship between K and testing accuracy
# plt.plot(x_axis, y_axis)
plt.xticks(iterazioni, my_xticks, rotation=90, ha="right")
plt.plot(iterazioni, accuracy)
plt.plot(iterazioni, accuracy_pca)
plt.plot(iterazioni, accuracy_ig)
plt.plot(iterazioni, accuracy_chi2)
plt.plot(iterazioni, accuracy_fcl)
plt.xlabel('Iterations')
plt.ylabel('Testing Accuracy')
#plt.legend(['zI', 'PCA', 'IG'], loc='upper left')
plt.legend(['zI', 'PCA', 'IG','CHI2','F_VAL'], loc='lower left')
#plt.show()
plt.savefig('RF_Bigtest1.pdf')