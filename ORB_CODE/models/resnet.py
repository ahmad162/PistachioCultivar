import numpy as np
import keras
from keras import backend as K
from keras.models import Model
from keras.applications import imagenet_utils
from keras.layers import GlobalAveragePooling2D, Dropout, BatchNormalization, MaxPooling2D
from keras.layers import Input, Dense, Conv2D, DepthwiseConv2D, SeparableConv2D, Activation
from keras.layers import Flatten, MaxPool2D, AvgPool2D, GlobalAvgPool2D, UpSampling2D
from keras.layers import concatenate, add, Dropout,  Lambda, Activation #, LeakyReLU,ReLU
from keras.layers.convolutional import AveragePooling2D
from keras import optimizers, regularizers
from keras.callbacks import LearningRateScheduler
from keras.regularizers import l2
from keras.optimizers import Adam
from keras.metrics import categorical_crossentropy
from keras.preprocessing.image import ImageDataGenerator
from IPython.display import Image
from keras.callbacks import LearningRateScheduler
from keras.callbacks import  ModelCheckpoint, EarlyStopping
import os, json
import matplotlib.pyplot as plt

class ResNet:
    def __init__(self):
        print("Initializing...")
        #self.__residual_module(data, K, stride, chanDim, red=False,reg=0.0001, bnEps=2e-5, bnMom=0.9)
        #self.__buildmodel(width, height, depth, classes, stages=(9, 9, 9), filters=(64, 64, 128, 256),
                #reg=0.0001, bnEps=2e-5, bnMom=0.9, dataset="cifar")
        #self.build(base_dir = "I:/GHOLAMREZA_Z/ACHENY_LAST_ORB64", EPOCHS=200, patient=30,fold=1)
        
    def __residual_module(data, K, stride, chanDim, red=False,
            reg=0.0001, bnEps=2e-5, bnMom=0.9):
        # the shortcut branch of the ResNet module should be
        # initialize as the input (identity) data
        # chanDim: defines the axis which will perform batch normalization
        shortcut = data

         # the first block of the ResNet module are the 1x1 CONVs
        bn1 = BatchNormalization(axis=chanDim, epsilon=bnEps,
                                     momentum=bnMom)(data)
        act1 = Activation("relu")(bn1)
        conv1 = Conv2D(int(K * 0.25), (1, 1), use_bias=False,
        kernel_regularizer=l2(reg))(act1)
        # the second block of the ResNet module are the 3x3 CONVs
        bn2 = BatchNormalization(axis=chanDim, epsilon=bnEps,
                                    momentum=bnMom)(conv1)
        act2 = Activation("relu")(bn2)
        conv2 = Conv2D(int(K * 0.25), (3, 3), strides=stride,
        padding="same", use_bias=False,
        kernel_regularizer=l2(reg))(act2)
        # the third block of the ResNet module is another set of 1x1
         # CONVs
        bn3 = BatchNormalization(axis=chanDim, epsilon=bnEps,
                                    momentum=bnMom)(conv2)
        act3 = Activation("relu")(bn3)
        conv3 = Conv2D(K, (1, 1), use_bias=False,
        kernel_regularizer=l2(reg))(act3)
        # if we are to reduce the spatial size, apply a CONV layer to
        # the shortcut
        if red:
            shortcut = Conv2D(K, (1, 1), strides=stride,
                                use_bias=False, kernel_regularizer=l2(reg))(act1)
            # add together the shortcut and the final CONV
        x = add([conv3, shortcut])
        
        # return the addition as the output of the ResNet module
        return x

    def __buildmodel(width, height, depth, classes, stages=(9, 9, 9), filters=(64, 64, 128, 256),
                reg=0.0001, bnEps=2e-5, bnMom=0.9, dataset="cifar"):
            # initialize the input shape to be "channels last" and the
            # channels dimension itself
            inputShape = (height, width, depth)
            chanDim = -1

            # if we are using "channels first", update the input shape
            # and channels dimension
            if K.image_data_format() == "channels_first":
                inputShape = (depth, height, width)
                chanDim = 1
            # set the input and apply BN
            inputs = Input(shape=inputShape)
            x = BatchNormalization(axis=chanDim, epsilon=bnEps,
                                    momentum=bnMom)(inputs)

            # check if we are utilizing the CIFAR dataset
            if dataset == "cifar":
            # apply a single CONV layer
                x = Conv2D(filters[0], (3, 3), use_bias=False,
            padding="same", kernel_regularizer=l2(reg))(x)
            # loop over the number of stages
            for i in range(0, len(stages)):
            # initialize the stride, then apply a residual module
            # used to reduce the spatial size of the input volume
                stride = (1, 1) if i == 0 else (2, 2)
                x = ResNet.__residual_module(x, filters[i + 1], stride,
                                            chanDim, red=True, bnEps=bnEps, bnMom=bnMom)

            # loop over the number of layers in the stage
            for j in range(0, stages[i] - 1):
            # apply a ResNet module
                x = ResNet.__residual_module(x, filters[i + 1],
                                            (1, 1), chanDim, bnEps=bnEps, bnMom=bnMom)
            # apply BN => ACT => POOL
            x = BatchNormalization(axis=chanDim, epsilon=bnEps,
                                    momentum=bnMom)(x)
            x = Activation("relu")(x)
            x = AveragePooling2D((8, 8))(x)
            # softmax classifier
            x = Flatten()(x)
            x = Dense(classes, kernel_regularizer=l2(reg))(x)
            x = Activation("softmax")(x)

            # create the model
            model = Model(inputs, x, name="resnet")

            # return the constructed network architecture
            return model
    @staticmethod
    def build(base_dir = "I:/GHOLAMREZA_Z/ACHENY_LAST_ORB64", EPOCHS=200, patient=30,fold=1):
        input_shape = (64,64)
        train_dir = os.path.join(base_dir, 'train')
        valid_dir = os.path.join(base_dir, 'validation')
        test_dir = os.path.join(base_dir, 'test')
        CLASS_NO = len(os.listdir(train_dir))
        BATCH_SZ = 32 #128
        weight_decay = 1e-4
        model = ResNet.__buildmodel(width=64, height=64, depth=3, classes=30, reg=0.0002)
        
        opt_rms = keras.optimizers.adam(lr=1e-3,decay=1e-6)  
        model.compile(loss='categorical_crossentropy', optimizer=opt_rms,metrics=['accuracy','top_k_categorical_accuracy']) 
        
        train_datagen=ImageDataGenerator(rescale=1./255, samplewise_center=True, samplewise_std_normalization=True)#,
                                  #rotation_range=40,width_shift_range=0.2, height_shift_range=0.2,
                                  #shear_range=0.2,zoom_range=0.2,horizontal_flip=True,fill_mode='nearest')                              

        test_datagen = ImageDataGenerator(rescale=1./255, samplewise_center=True, samplewise_std_normalization=True)
                               
        train_generator=train_datagen.flow_from_directory(train_dir, target_size=input_shape,batch_size=BATCH_SZ,
                                                  class_mode='categorical',shuffle=True)

        validation_generator=test_datagen.flow_from_directory(valid_dir,target_size=input_shape, 
                                                      batch_size=BATCH_SZ, class_mode='categorical')  

        weights_dir = base_dir+'/resnet_weight_'+str(fold)+'.h5'

        if os.path.exists(weights_dir):
                model.load_weights(weights_dir)
                print(weights_dir,' loaded')
        else:
            print(weights_dir,' will be create')
        checkpointer = ModelCheckpoint(weights_dir, save_best_only=True, save_weights_only=True)
        early_stopping = EarlyStopping(patience=patient) 
        
        step_size_train=int(np.ceil(train_generator.n/train_generator.batch_size))
        step_size_validation=int(np.ceil(validation_generator.n/validation_generator.batch_size))
        def lr_schedule(epoch):
            lrate = 0.001
            if epoch > 50:
                lrate = 0.0005
            if epoch > 75:
                lrate = 0.0003
            return lrate
        history=model.fit_generator(generator=train_generator,
                            steps_per_epoch=step_size_train,
                            epochs=EPOCHS,
                            validation_data=validation_generator,
                            validation_steps=step_size_validation,
                            workers=8,             
                            max_queue_size=32,             
                            verbose=1,
                            callbacks=[checkpointer, early_stopping,LearningRateScheduler(lr_schedule)])
        acc=history.history['acc']
        val_acc = history.history['val_acc']
        loss = history.history['loss']
        val_loss = history.history['val_loss']
        top5_acc=history.history['top_k_categorical_accuracy']
        val_top5_acc=history.history['val_top_k_categorical_accuracy']
        with open(base_dir+'/resnet_'+str(fold)+'.txt', 'w') as filehandle: 
            json.dump((acc,val_acc,loss,val_loss,top5_acc,val_top5_acc), filehandle)
        model.save(base_dir+'/resnet_'+str(fold)+'.h5')
        epochs = range(len(acc))

        plt.plot(epochs, acc, 'b.-', label='Training acc')
        plt.plot(epochs, val_acc, 'g', label='Validation acc')
        plt.title('Training and validation accuracy')
        plt.ylabel('accuracy')
        plt.xlabel('epoch')
        plt.legend(['train_accuracy', 'valid_accuracy'], loc='best')
        plt.figure()
        plt.plot(epochs, loss, 'b.-', label='Training loss')
        plt.plot(epochs, val_loss, 'g', label='Validation loss')
        plt.title('Training and validation loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(['train_loss', 'valid_loss'], loc='best')
        plt.figure()
        plt.plot(epochs, top5_acc, 'b.-', label='Training top5-acc')
        plt.plot(epochs, val_top5_acc, 'g', label='Validation top5-acc')
        plt.title('Training and validation Top5 accuracy')
        plt.ylabel('top5 accuracy')
        plt.xlabel('epoch')
        plt.legend(['top5_acc', 'valid_top5_acc'], loc='best')
        plt.show()
        
        batch_sz=32

        for tested_dir in [test_dir, train_dir, valid_dir] :
            test_generator = test_datagen.flow_from_directory(
                tested_dir, 
                target_size=input_shape, 
                batch_size=BATCH_SZ,
                class_mode='categorical',shuffle=False)

            test_loss, test_acc, top5_acc = model.evaluate_generator(test_generator)#, steps=batch_sz)
            print('%s acc:%.3f top5_acc:%.3f loss:%.3f\n'%(tested_dir[tested_dir.rfind('/')+1:],test_acc,top5_acc, test_loss))
