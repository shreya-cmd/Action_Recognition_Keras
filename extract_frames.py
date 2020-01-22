import os
import cv2

listing = os.listdir("data")
count = 0
for folder in listing:
    os.makedirs("data_frames/"+ folder+"/")
    for videoname in os.listdir("data/"+ folder):
        video = cv2.VideoCapture("data/" + folder + "/" + videoname)
        count = 1
        success=1
        os.makedirs("data_frames/" + folder+"/"+videoname.replace(".avi",""))
        while success:        
            success,image = video.read()
            if not success: 
                break
            filename = "data_frames/" + folder+"/"+videoname.replace(".avi","") + "/" + folder + "_" + videoname.replace(".avi","")+'_' + str(count) + '.jpg'
            print(filename)
            cv2.imwrite(filename,image)
            count+=1