# -*- coding: utf-8 -*-
"""
Created on Fri Dec  1 09:04:11 2017

@author: GangTimes
"""

import os,glob
from skimage import io
import dlib as db
import numpy as np
import re
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
          
    return descriptor

def batch_extract(file):
    with open(file,'r') as data:
        lines=data.readlines()
        descriptors=dict()
        for line in lines:
            strs=line.strip('\n').split(',')
            path=strs[0]
            label=strs[1]
            
            try:
                descriptor=extract_feature(path)
            except Exception:
                continue
            else:
                if label not in descriptors.keys():    
                    descriptors[label]=[]
                descriptors[label].append(descriptor)
    data.close()
    return descriptors

      
def predict(file,descriptors):
    with open('result.txt','w') as result:
        with open(file,'r') as data:
            lines=data.readlines()
            right_count=0
            unknow_count=0
            total_count=0
            for line in lines:
                total_count+=1
                strs=line.strip('\n').split(',')
                path=strs[0]
                try:
                    descriptor=extract_feature(path)
                except Exception:
                    result.write(path+','+'Unknow'+'\n')
                    unknow_count+=1
                    continue
                else:
                    dist=[]
                    keys=[]
                    for label,values in descriptors.items():
                        dist_=[]
                        for value in values:
                            dist_.append(np.linalg.norm(descriptor-value))
                        dist.append(np.mean(dist_))
                        keys.append(label)
                    index=dist.index(min(dist))
                    flag=re.search(keys[index],path)
                    if flag!=None:
                        right_count+=1
                    result.write(path+','+keys[index]+'\n')
        data.close()
    result.close()
    print('未能判决的图片比例:'+str(unknow_count/total_count))
    print('判决正确的比例为:'+str(right_count/total_count))
    return descriptors        
        
        
if __name__=='__main__':
    Train='Train.txt'
    Test='Test.txt'
    descriptors=batch_extract(Train)
    predict(Test,descriptors)