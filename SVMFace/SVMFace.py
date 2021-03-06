# -*- coding: utf-8 -*-
"""
Created on Fri Dec  1 17:29:25 2017

@author: GangTimes
"""
import os,glob
from skimage import io
import dlib as db
import numpy as np
import pandas as pd
from sklearn import svm
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
            
            try:
                data=extract_feature(path)
            except Exception:
                continue
            else:
                datas=datas.append(data,ignore_index=True)
                if len(strs)==2:
                    clabel= data.index.shape[0]
                    if clabel not in list(datas.columns):
                        datas[clabel]=''
                    datas.iloc[-1,-1]=label
    fdata.close()
    return datas

      
def main():
    Train='Train.txt'
    Test='Test.txt'
    train_datas=batch_extract(Train)
    Ty=train_datas[128]
    TX=train_datas.drop([128],axis=1)
    X=batch_extract(Test)
    model=svm.SVC(kernel='linear',C=1)
    model.fit(TX,Ty)
    y=model.predict(X)
    print(len(y))
    print(X.shape[0])
    count=0
    with open('result.txt','w') as rdata:
        with open(Test,'r') as fdata:
            lines=fdata.readlines()
            for line,label in zip(lines,y):
                rdata.write(line.strip('\n')+','+label+'\n')
                count+=1
    print(count)
            
        
if __name__=='__main__':
    main()
    