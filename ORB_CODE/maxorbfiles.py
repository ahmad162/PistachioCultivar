#see also:  https://www.ncbi.nlm.nih.gov/pmc/articles/PMC4081273/
import numpy as np
import cv2 as cv
import os
from matplotlib import pyplot as plt
from imutils import paths    
import time
class MaxORBFiles:
    @staticmethod
    def extract(SRCFOLDER='J:/Temporary/ACHENY_LAST_1024',DSTFOLDER='J:/Temporary/ACHENY_MaxORB',percent=10):
        files = list(paths.list_images(SRCFOLDER)) 
        if not os.path.isdir(DSTFOLDER):
             os.mkdir(DSTFOLDER)
        folders=os.listdir(SRCFOLDER)
        for fold in folders:
            if not os.path.isdir(DSTFOLDER+'\\'+fold):
                 os.mkdir(DSTFOLDER+'\\'+fold)
        #imgo = cv.resize(imgo,(256,256))
        # Initiate ORB detector
        orb = cv.ORB_create(nfeatures = 500,
                    scaleFactor = 1.2, #1.2
                    nlevels = 8, 
                    edgeThreshold = 0, #31
                    firstLevel = 0,
                    WTA_K = 3, #2
                    scoreType = cv.ORB_FAST_SCORE, #cv.ORB_HARRIS_SCORE,
                    patchSize = 2,fastThreshold = 50)#31 20 )
        f=0;rejected=0;xf=0
        t=time.time()
        for file in files:
            f=f+1
            dest=file.replace(SRCFOLDER,DSTFOLDER)
            if not os.path.exists(dest):
                imgcolor = cv.imread(file)
                img = cv.cvtColor(imgcolor, cv.COLOR_BGR2GRAY)
                # find the keypoints with ORB
                kp = orb.detect(img,None)
                print("===== ",len(kp)," =====")
                if (len(kp)==0):
                    rejected+=1; print(rejected,' rejects')
                    continue
         
                #destfolder=dest[:dest.rfind('\\')] 
                #destfile=dest[:dest.rfind('.')]+'.jpg' 
        
                #if not os.path.exists(destfolder):
                    #os.mkdir(destfolder)

                #image= cv.imwrite(destfile,imgcolor[Y:Y+H,X:X+W,:]) #img)
        ###########################
                print(f,')',file,' processed ',xf)
        #############################################################
                #cv.putText(imgcolor,'%d/%dkeys in %dpcs'%(findedkeypoints,num_of_keypoints,piece),org,cv.FONT_HERSHEY_SIMPLEX, 
                   #1.4, (255,255, 0),2, lineType=cv.LINE_AA)  
                #fig = plt.figure(num=None, figsize=(10, 10), dpi=150, facecolor='w', edgecolor='k')
                #plt.imshow(imgcolor),plt.show();#cv.UMat.get(
        
        print('Done in', (time.time()-t))
        