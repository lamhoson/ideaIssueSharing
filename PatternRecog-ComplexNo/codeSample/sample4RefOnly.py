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


def genConvDrpPooLayer(inputs, noOfLayers, noOfBaseFilter, kernalSize, strides:list, dropRates:list, poolSize ): # Conv2D-Dropout-MaxPooling2D, n times defined by CONV_LAYERS
    for i in range(noOfLayers): #Conv2D-Dropout-MaxPooling2D
        inputs  = Conv2D(filters= noOfBaseFilter*(i+1), #no-of-filters doubles after each layer (e.g. 32-64...)
				   kernel_size=kernalSize, strides=strides[i], #[0] is 1st stride size, [1] is 2nd etc.
				   padding='same', activation='relu')(inputs)
        inputs  = MaxPooling2D(pool_size=poolSize)(inputs) #default is (2, 2). debug output->input, 26aug2021
        inputs  = Dropout(dropRates[i])(inputs) #[0] is 1st dropout rate, [1] is 2nd etc.
        
    return inputs #debug output->input, 26aug2021

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
# CAN't add Dense layer here, all will be 5 !!. Can't explain YET. 6jun2021
    # output = Dense(NB_NEURON, activation='relu')(output)
    # output = Dropout(dropRate)(output) # output = layers.BatchNormalization()(output) #A4, worster performance and faster training
    output = Dense(NB_CLASSES,)(output)
    output = Dropout(dropRate)(output) # better performance just before Softmax, 28aug2021
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
    