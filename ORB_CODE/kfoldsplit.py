from imutils import paths
import numpy as np
import os, shutil
from numpy import array

# scikit-learn k-fold cross-validation
from sklearn.model_selection import KFold

class KFoldSplit:
    @staticmethod
    def perform(base_dir="J:/Temporary/ACHENY_LAST_1024",k=5, fold=1, onlytrain=True):
        validpath = os.path.join(base_dir,'validation')
        trainpath = os.path.join(base_dir,'train')
        folders=os.listdir(trainpath)        
        if not os.path.exists(validpath):
            os.mkdir(validpath)
            for folder in folders:
                os.mkdir(os.path.join(base_dir,'validation',folder))
        validfiles=list(paths.list_images(validpath))
        trainfiles=list(paths.list_images(trainpath))
        print("Move validation files to train ...")

        for image in validfiles:
            os.rename(image,image.replace('validation','train'))

        #move fold=fold from train to validation
        if not onlytrain:
            for folder in folders:
                files=array(os.listdir(base_dir+'/train/'+folder))
                files.sort()
                # prepare cross validation
                kfold = KFold(k, True, 1)
                # enumerate splits
                f=1
                for train, validation in kfold.split(files):
                    if f==fold : 
                        break
                    f=f+1
                i=0
                for file in files[validation]:
                    try:
                        os.rename(base_dir+'/train/'+folder+'/'+file, base_dir+'/validation/'+folder+'/'+file)
                        i=i+1
                    except:
                        i=i+1
                print("from {} {} files moved".format(folder, i))
        else:
             print("validation is Empty")
