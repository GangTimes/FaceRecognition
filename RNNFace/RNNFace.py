# -*- coding: utf-8 -*-
"""
Created on Fri Dec  1 21:58:09 2017

@author: GangTimes
"""

import os,glob
from skimage import io
import dlib as db
import numpy as np
import pandas as pd
from sklearn import neighbors
predictor_path='predictor.dat'
model_path='model.dat'
detector=db.get_frontal_face_detector()
face_model=db.face_recognition_model_v1(model_path)
sp=db.shape_predictor(predictor_path)
#win = db.image_window()

def extract_feature(file):
    img=io.imread(file)   
    faces=detector(img,1) 
    if len(faces)==0:
        raise Exception
    for index,face in enumerate(faces):        
        shape=sp(img,face)
        descriptor=np.array(face_model.compute_face_descriptor(img,shape))

    data=pd.Series(data=descriptor)      
    return data

def batch_extract(file):
    datas=pd.DataFrame()
    with open(file,'r') as fdata:
        lines=fdata.readlines()
        for line in lines:
            strs=line.strip('\n').split(',')
            path=strs[0]
            if len(strs)==2:
                  label=strs[1]
            else:
                label=path
            try:
                data=extract_feature(path)
            except Exception:
                continue
            else:
                datas=datas.append(data,ignore_index=True)
                clabel= data.index.shape[0]
                if clabel not in list(datas.columns):
                    datas[clabel]=''
                datas.iloc[-1,-1]=label
    fdata.close()
    return datas

      
def main():
    n_neighbors = 15
    Train='Train.txt'
    Test='Test.txt'
    train_datas=batch_extract(Train)
    Ty=train_datas[128]
    TX=train_datas.drop([128],axis=1)
    test_datas=batch_extract(Test)
    labels=test_datas[128]
    X=test_datas.drop([128],axis=1)
    model=neighbors.KNeighborsClassifier(n_neighbors,weights='uniform')
    model.fit(TX,Ty)
    y=model.predict(X)
    count=0
    with open('result.txt','w') as rdata:
        for label,iy in zip(labels,y):
            rdata.write(label+','+iy+'\n')
            count+=1
    print(count)
            
        
if __name__=='__main__':
    main()
    