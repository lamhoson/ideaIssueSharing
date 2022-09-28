# -*- coding: utf-8 -*-

import tensorflow as tf
from tensorflow.keras import layers, metrics
from tensorflow.keras.layers import Dense, Dropout, Input, Conv2D, MaxPooling2D, Flatten #, concatenate, Activation
from tensorflow.keras.models import Model
from tensorflow.keras.utils import plot_model, to_categorical # plot_model must installed pydot, X.2.
import numpy as np; import time; import sklearn.model_selection as skm; import hgrampy as hg  # import self-made Hologram library
from subprocess import check_output; from multiprocessing import cpu_count # cpu & speed info
import sys; import pandas as pd; import os
from sklearn.metrics import confusion_matrix; import seaborn as sns; import matplotlib.pyplot as plt

""" Common tuning network hyper-parameters """
NB_FILTER =3
DELTA_E= 0.0001
PATIENCE_NB=5 
NB_EPOCHS= 1000 # use earlyStopping to stop
SPILT_RATIO_4T=0.2 
VALID_SPLIT=0.2

BATCH=32 
CONV_LAYERS=2 
DROPOUT_RATE1=0.5 
DROPOUT_RATE2=0.5 
STRIDES_1ST=(1,1) 
STRIDES_2ND=(2,2) 

""" Non-common tuning network hyper-parameters """
X_SILENCE=1 # 0=silence, 1=echo console
KERNAL_SIZE =(3,3) # convolution filter kernel size 3x3.
POOL_SIZE =(2,2)
CHANNELS = 1 # image channel is one for gray image
OPTI_MODE= 'val_auc' #'val_auc' 'val_precision', 'auc'

METRICS_DEF = [metrics.CategoricalAccuracy(name='acc'), 
               metrics.AUC(name='auc') ] #AUC=0 => model's prediction is 100% wrong, AUC=1 => prediction 100% right.

def genConvDrpPooLayer(inputs, noOfLayers, noOfBaseFilter, kernalSize, strides:list, dropRates:list, poolSize ): # Conv2D-Dropout-MaxPooling2D, n times defined by CONV_LAYERS
    for i in range(noOfLayers): #Conv2D-Dropout-MaxPooling2D
        inputs  = Conv2D(filters= noOfBaseFilter*(i+1), #no-of-filters doubles after each layer (e.g. 32-64...)
				   kernel_size=kernalSize, strides=strides[i], #[0] is 1st stride size, [1] is 2nd etc.
				   padding='same', activation='relu')(inputs)
        inputs  = MaxPooling2D(pool_size=poolSize)(inputs) #default is (2, 2). debug output->input
        inputs  = Dropout(dropRates[i])(inputs) #[0] is 1st dropout rate, [1] is 2nd etc.
        
    return inputs #debug output->input

def concateNsoftMax(leftInp, rightInp, dropRate): # concatenateEnsemble-softmax
    """
    concate => flatten => Dropout => Dense(NB_CLASSES) => Softmax

    Parameters
    ----------
    leftInp : x input
    rightInp : y input

    Returns output the TF model
    -------
    """
    output = layers.concatenate([leftInp, rightInp]) # default concat at last axis e.g.(2,5)+(2,5)=>(2,10), A1
    output = Flatten()(output) # flat feature maps 
# CAN't add Dense layer here, all will be 5 !!. Can't explain YET. 
    # output = Dense(NB_NEURON, activation='relu')(output)
    # output = Dropout(dropRate)(output) # output = layers.BatchNormalization()(output) #A4, worster performance and faster training
    output = Dense(NB_CLASSES,)(output)
    output = Dropout(dropRate)(output) # better performance just before Softmax, 
    output = layers.Activation('softmax')(output)           
    
    return output

def createYcnn(noOfLayers, noOfBaseFilter, kernalSize, strides:list, dropRates:list, poolSize, rows, columns, channels): # 2x(Convolute and Pooling layers) + 2x Dense Layers model
# Y network, 2-input and 1-output, A2
	# left branch of Y network. 
    leftType = Input(shape=(rows, columns, channels)) #declare the Class object
    leftInp=leftType # creat the instance
    leftInp=genConvDrpPooLayer(leftInp, noOfLayers, noOfBaseFilter, kernalSize, strides, dropRates, poolSize) # Conv2D-Dropout-MaxPooling2D, n times defined by CONV_LAYERS

	# right branch of Y network
    rightType = Input(shape=(rows, columns, channels))  #declare the Class object
    rightInp=rightType # must difference to leftType debug28aug2021, Error: i/ps to model redundant, all i/p should only appear once
    rightInp=genConvDrpPooLayer(rightInp, noOfLayers, noOfBaseFilter, kernalSize, strides, dropRates, poolSize) # Conv2D-Dropout-MaxPooling2D, n times defined by CONV_LAYERS

    output = concateNsoftMax(leftInp, rightInp, DROPOUT_RATE2) #concate=> flatten=> Dropout=> Dense=> Softmax
    
	# build TF model
    model = Model([leftType, rightType], output) #leftType, rightType MUST differece debug28aug2021, Error: i/ps to model redundant, all i/p should only appear once
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=METRICS_DEF) #'sgd'. adam shall better&faster than sgd
    return model

def creatNprintEnModel(string, rows, columns): #e.g. string='modelSample.png'
    model = createYcnn(CONV_LAYERS, NB_FILTER, KERNAL_SIZE, (STRIDES_1ST, STRIDES_2ND),
                         (DROPOUT_RATE1, DROPOUT_RATE2), POOL_SIZE, rows, columns, CHANNELS) # MUST refresh after each loop
    model.summary(); plot_model(model, to_file=string, show_shapes=True, show_layer_names=True) #plot the model to *.png
    return model

def extractMagOrPhrase(dataset, magPhrase=hg.MAGNITUDE_M, phraseSmoothingMethod=RAW_MODE):
    nbImages,rows,columns = dataset.shape #e.g nbImages,rows,columns=(25000, 64, 64). For later reshape
    if magPhrase==hg.MAGNITUDE_M: dataset=np.absolute(dataset); print("Exacted Hologram Pixels Magnitudes") 

    elif magPhrase==hg.PHRASE_M:
        dataset=np.angle(dataset); print("\nExtracted Hologram Pixels Phrase-Angles.", end =' Applying..')
        if phraseSmoothingMethod==RAW_MODE:
            print("No phrase-smoothing")
        elif phraseSmoothingMethod==COS_MODE:
            print("Cos phrase-smoothing"); dataset=np.cos(dataset) #cos a bit better than sin
            
    else: print('Invalid Extraction Mode !. Quit the code now.'); quit()
            
    dataset=(dataset-dataset.min())/(dataset.max()-dataset.min()) #12oct2021, max => min-max Normalize
    dataset=dataset.reshape((nbImages,rows,columns,CHANNELS)); dataset=dataset.astype("float32") #!! astypes will discard complex part if place before absolute(). Try single precision for speed first.
    return dataset

# =============================================================================
# ###         Main Program start Here         ###
# =============================================================================

#########################################################################################################################
#                   Ensemble magnitude & phrases deep Learning Recognition Start Here                                    #
#########################################################################################################################
mString='Phr_'+modeDict.get(smoothPhr) # get Text description of smoothing methods
nnModel=creatNprintEnModel("modelBioSample"+"EnMode-"+str(CONCATE_M)+".png", rows, columns )

"""
Reference: A)
1) https://keras.io/api/layers/merging_layers/
2) https://github.com/PacktPublishing/Advanced-Deep-Learning-with-Keras/blob/master/chapter2-deep-networks/cnn-y-network-2.1.2.py
   https://www.pyimagesearch.com/2019/02/04/keras-multiple-inputs-and-mixed-data/
3) https://stackoverflow.com/questions/38971293/get-class-labels-from-keras-functional-model
4) Batch regulisation: https://www.kdnuggets.com/2018/09/dropout-convolutional-networks.html
"""    
    