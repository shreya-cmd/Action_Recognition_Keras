import csv
import glob
import os
import os.path
import cv2
videonames=[]
data_file = [] 
 
folders = ['bend', 'run', 'skip', 'walk']
videonames=[]
for item in folders:   
    files = glob.glob('data' + '/' + item+ '/*.avi' )
    for name in files:
        cap = cv2.VideoCapture(os.path.join(name))
        nb_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        data_file.append([item, name, nb_frames])       
    videonames.extend(files)
 
with open('data_file.csv', 'w') as fout:
    writer = csv.writer(fout)
    writer.writerows(data_file)
