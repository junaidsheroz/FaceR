import cv2
import face_recognition
import numpy as np
import os
from datetime import datetime
from playsound import playsound
from numpy import nan
import pandas as pd
import numpy
# from PIL import ImageGrab
from datetime import timedelta
import pymysql
import streamlit as st

##########################################Function to add encodings to csv file############################################


def DbEncodings(img_dir,Name):
    '''img_dir = image file name with extension
        Name = Name of the person in the image'''
    encList = []
    images = cv2.imread(img_dir)
    image = cv2.cvtColor(images, cv2.COLOR_BGR2RGB)
    try:
        enc = face_recognition.face_encodings(image)[0]
    except IndexError as e:
        print(e)
    #encList.append(enc)
    print(enc)
    encod = pd.read_csv('encodingsKnown.csv')
    #enc = np.array(enc)
    print(enc)
    Name = Name.upper()
    print(encod.head())
    encod.loc[len(encod.index)]=[Name,enc]
    encod.to_csv('encodingsKnown.csv',index=False)
    return Name
current_in = {}


def face_recog():
    ##############################NOT DataBase###################
    def timecomp(s):
        number = int(s.split(':')[0]+s.split(':')[1]+s.split(':')[2])
        return number
    count=0
    vcount = 0
    enc_csv = pd.read_csv('encodingsKnown.csv')
    Names = [n for n in enc_csv['Names']]
    encodeKnown = []
    for enc in range(len(enc_csv)):
        
        encodeKnown.append(np.array([float(i) for i in enc_csv['Encodings'][enc].replace('[','').replace(']','').replace(',','').split()]))

    cap = cv2.VideoCapture(0)
    attend = pd.read_csv('attendance.csv')
    in_out = pd.read_csv('in_out.csv')


    current_in={}     #Employees currently in office
    entered = 0       #Variable used to acknowledge the updated file when 1
    current=0

    while True:
        if current==0:
            with open("file.txt", "r") as file: 
                data = file.readlines() 
                for line in data: 
                    word = line.split() 
                    #print (word) 
            current_in = {}
            for i in data:
                current_in[i.split()[0]]=i.split()[1] 
            current=1
        if entered==1:        #Save file when var=1
            in_out = pd.read_csv('in_out.csv') 
            entered=0
            
        _, img = cap.read()
        image = cv2.resize(img,(0,0),None,0.25,0.25)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        facesInFrame = face_recognition.face_locations(image)
        encodesInFrame = face_recognition.face_encodings(image,facesInFrame)

        for encFace,faceLoc in zip(encodesInFrame,facesInFrame):
            #print(encFace,faceLoc)
            matchList = face_recognition.compare_faces(encodeKnown,encFace)
        # print(matchList)
            faceDis = face_recognition.face_distance(encodeKnown,encFace)
            print(faceDis)
            match1 = min(faceDis)          #min Distance 
            #match = faceDis.index(match1)
            match = np.argmin(faceDis)     #index of min distance 

            
            
            if match1<0.5:
                
                if match1<=0.40:
                    
                    count+=1
                    vcount=0
                    if count>=3:
                        name = Names[match].upper()
    #                     try:
    #                         playsound('D:/Audio/%s.mp3'%(name))
    #                     except:
    #                         print(name)
    #                     playsound('D:/59089586.mp3')
                        count = 0
                        
                        y,w,z,x = faceLoc
                        y, w, z, x1 = y*4,w*4,z*4,x*4
                        cv2.rectangle(img,(x,y),(w,z),(0,255,0),2)
                        cv2.rectangle(img,(x,z-35),(w,z),(0,255,0),cv2.FILLED)
                        cv2.putText(img,name,(x+6,z-6),cv2.FONT_HERSHEY_COMPLEX,1,(255,255,255),2)
                        print(name)

                        curr = datetime.now()
                        dt = curr.strftime('%d-%m-%Y')
                        time = curr.strftime('%H:%M:%S')
                        itime = timecomp(time)
                        if name not in current_in:
                            if name not in [q for q in in_out[(in_out['Name']==name) & (in_out['Date']==dt)]['Name']]:
                                current_in[name]=itime
                                in_out.loc[len(in_out.index)]=[name,dt,time,nan,nan]
                                entered = 1
                                f1 = open('file.txt','a')
                                f1.write(name+' '+time+'\n')
                                f1.close()
                                in_out.to_csv('in_out.csv',index=False)
                                current = 0
                            else:
                                try:
                                    last_out = timecomp([j for j in in_out[(in_out['Name']==name) & (in_out['Date']==dt) & (in_out['logout_time'].isna()==False)]['logout_time']][-1])
                                    if last_out+150<itime:
                                        f1 = open('file.txt','a')
                                        f1.write(name+' '+itime+'\n')
                                        f1.close()
                                        current_in[name]=itime
                                        in_out.loc[len(in_out.index)]=[name,dt,time,nan,nan]
                                        entered = 1
                                        in_out.to_csv('in_out.csv',index=False)
                                        current=0
                                except:
                                    print('in_pass')
                                    pass
                        else:
                            try:
                                if name in [j for j in in_out[(in_out['Name']==name) & (in_out['Date']==dt) & (in_out['logout_time'].isna()==True)]['Name']]:
                                    last_in = [k for k in in_out[(in_out['Name']==name) & (in_out['Date']==dt) & (in_out['logout_time'].isna()==True)]['login_time']][-1]
                                    if timecomp(last_in)+500<itime:
                                        worktime = str(int(time[:2])-int(str(last_in)[:2]))+':'+str(int(time[3:5])-int(str(last_in)[3:5]))+':'+str(int(time[6:7])-int(str(last_in)[6:7]))
                                        indx = in_out[(in_out['Name']==name) & (in_out['Date']==dt) & (in_out['logout_time'].isna()==True)].index[0]
                                        in_out.drop(indx,axis=0,inplace=True)
                                        in_out.loc[len(in_out.index)]=[name,dt,timecomp(last_in),time,worktime]
                                
                                #write current_in file 
                                        try:
                                            for i in range(len(data)):
                                                if name in data[i]:
                                                    data.pop(i)
                                        except:
                                            print('Current_in', data,'::::::::::::::::::::::::::::::::::::::::::')
                                        f1 = open('file.txt','a')
                                        for i in data:
                                            f1.write(i)
                                        f1.close()
                                        
                                        current=0
                                        current_in.pop(name)
                                        entered = 1
                                        in_out.to_csv('in_out.csv',index=False)
                            except:
                                print('OUT_Error')
                                pass
            else:
                vcount+=1
                count=0
                if vcount>=6:
                    y,w,z,x = faceLoc
                    y, w, z, x1 = y*4,w*4,z*4,x*4
                    cv2.rectangle(img,(x,y),(w,z),(0,255,0),2)
                    cv2.rectangle(img,(x,z-35),(w,z),(0,255,0),cv2.FILLED)
                    cv2.putText(img,'visitor',(x+6,z-6),cv2.FONT_HERSHEY_COMPLEX,1,(255,255,255),2)
                # Attendance('Visitor')
                    print('Visitor')
                    vcount=0
                    
        cv2.imshow('Webcam',img)
        k = cv2.waitKey(1)
        if k == 27:         # wait for ESC key to exit
            break
        cv2.waitKey(1)
    cv2.destroyAllWindows()


st.title("Face Recognition Abrar")

st.camera_input()


