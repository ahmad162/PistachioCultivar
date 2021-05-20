#https://github.com/digital-thinking/ann-jupyter-notebooks/blob/master/TransferLearning-EfficientNet/TransferLearning-EfficientNet.ipynb
import tensorflow as tf
import matplotlib.pyplot as plt
#import pandas as pd
import numpy as np
import scipy.io as sio
import os, json
from keras.layers import GlobalAveragePooling2D, Dense, BatchNormalization
from keras import Model
import efficientnet.keras as efn
from keras.callbacks import  ModelCheckpoint, EarlyStopping

class EfficientB1:
    def build(base_dir = "I:/GHOLAMREZA_Z/ACHENY_LAST_224", EPOCHS=200, patient=3,fold=1):
        input_shape = (224,224)
        train_dir = os.path.join(base_dir, 'train')
        valid_dir = os.path.join(base_dir, 'validation')
        test_dir = os.path.join(base_dir, 'test')
        batch_sz = 32 
        
        train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
                rescale=1. / 255, zoom_range=0.2,  rotation_range = 5,
                horizontal_flip=True)

        test_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1. / 255)

        train_generator=train_datagen.flow_from_directory(train_dir,
                      class_mode="categorical",target_size=input_shape, batch_size=batch_sz)


        validation_generator=test_datagen.flow_from_directory(valid_dir,
                      class_mode="categorical", target_size=input_shape, batch_size=batch_sz)

        base_model = efn.EfficientNetB1(weights='imagenet', include_top=False)
        x = base_model.output
        x = GlobalAveragePooling2D()(x)
        predictions = Dense(len(train_generator.class_indices), activation='softmax')(x)
        model = Model(inputs=base_model.input, outputs=predictions)

        # fix the feature extraction part of the model
        for layer in base_model.layers:
            if isinstance(layer, BatchNormalization):
                layer.trainable = True
            else:
                layer.trainable = False
        
        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['acc','top_k_categorical_accuracy'])

        weights_dir=base_dir+'/EffB1_weight_'+str(fold)+'.h5'
        if os.path.exists(weights_dir):
            model.load_weights(weights_dir)
            print("XXX %s loaded XXX"%(weights_dir)) 
        else : 
            print("XXX %s will be create XXX"%(weights_dir)) 
        
        checkpointer = ModelCheckpoint(weights_dir, save_best_only=True, save_weights_only=True)
        early_stopping = EarlyStopping(patience=patient)
        history = model.fit_generator(generator=train_generator,
                    steps_per_epoch=np.ceil(train_generator.samples / batch_sz),
                    validation_data=validation_generator,
                    validation_steps=np.ceil(validation_generator.samples / batch_sz),
                    epochs=10,workers=8,max_queue_size=32,verbose=1,
                    callbacks=[checkpointer, early_stopping])
          
            
        acc = history.history['acc']
        val_acc = history.history['val_acc']
        loss = history.history['loss']
        val_loss = history.history['val_loss']
        top5_acc = history.history['top_k_categorical_accuracy']
        val_top5_acc = history.history['val_top_k_categorical_accuracy']

        model.save(base_dir+'/EffB1_'+str(fold)+'.h5')

        train_generator=train_datagen.flow_from_directory(train_dir,
                        class_mode="categorical", target_size=input_shape, batch_size=batch_sz)

        validation_generator=test_datagen.flow_from_directory(valid_dir,
                        class_mode="categorical", target_size=input_shape, batch_size=batch_sz)

        for layer in model.layers:
            layer.trainable = True
    
        model.compile(optimizer='sgd', loss='categorical_crossentropy', metrics=['acc','top_k_categorical_accuracy'])

        if os.path.exists(weights_dir):
            model.load_weights(weights_dir)
            print("XXX %s loaded XXX"%(weights_dir)) 
        else : 
            print("XXX %s will be create XXX"%(weights_dir)) 
        
        checkpointer = ModelCheckpoint(weights_dir, save_best_only=True, save_weights_only=True)
        early_stopping = EarlyStopping(patience=5)
        history = model.fit_generator(generator=train_generator,
                    steps_per_epoch=np.ceil(train_generator.samples / batch_sz) ,
                    validation_data=validation_generator,
                    validation_steps=np.ceil(validation_generator.samples / batch_sz),
                    epochs=20, workers=8, max_queue_size=32, verbose=1,
                    callbacks=[checkpointer, early_stopping])
        acc = acc + history.history['acc']
        val_acc = val_acc + history.history['val_acc']
        loss = loss + history.history['loss']
        val_loss = val_loss + history.history['val_loss']
        top5_acc = top5_acc + history.history['top_k_categorical_accuracy']
        val_top5_acc = val_top5_acc + history.history['val_top_k_categorical_accuracy']
        
        epochs = range(len(acc))

        plt.plot(epochs, acc, 'b.-', label='Training acc')
        plt.plot(epochs, val_acc, 'g', label='Validation acc')
        plt.title('Training and validation accuracy')
        plt.legend(['acc', 'val_acc'], loc='upper left')
        plt.ylabel('accuracy')
        plt.xlabel('epoch')

        plt.figure()

        plt.plot(epochs, loss, 'b.-', label='Training loss')
        plt.plot(epochs, val_loss, 'g', label='Validation loss')
        plt.title('Training and validation loss')
        plt.legend()

        plt.figure()

        plt.plot(epochs, top5_acc, 'b-', label='top5 accuracy')
        plt.plot(epochs, val_top5_acc, 'k-', label='val top5 accuracy')
        plt.title('Training and validation top5 accuracy')
        plt.legend()
        plt.show()
                   
        with open(base_dir+'/EffB1_'+str(fold)+'.txt', 'w') as filehandle:  
                json.dump((acc,val_acc,loss,val_loss,top5_acc,val_top5_acc), filehandle)
  
        print('EffB1_'+str(fold)+'.txt & ','EffB1_'+str(fold)+'.h5  saved ...')

        model.save(base_dir+"/EffB1_"+str(fold)+".h5")
                   