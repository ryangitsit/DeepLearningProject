import numpy as np
import matplotlib.pyplot as plt
from scipy import linalg 
# numpy.linalg is also an option for even fewer dependencies

def echo_state_network(input_data, regu):
    # load the data
    trainLen = 2000
    testLen = 2000
    initLen = 100
    data = input_data

    # generate the ESN reservoir
    inSize = outSize = 1
    resSize = 10000
    a = 0.3 # leaking rate
    np.random.seed(42)
    Win = (np.random.rand(resSize,1+inSize) - 0.5) * 1
    W = np.random.rand(resSize,resSize) - 0.5 
    # normalizing and setting spectral radius (correct, slow):
    print('Computing spectral radius...')
    rhoW = max(abs(linalg.eig(W)[0]))
    print('done.')
    W *= 1.25 / rhoW

    # allocated memory for the design (collected states) matrix
    X = np.zeros((1+inSize+resSize,trainLen-initLen))
    # set the corresponding target matrix directly
    Yt = data[None,initLen+1:trainLen+1] 

    # run the reservoir with the data and collect X
    x = np.zeros((resSize,1))
    for t in range(trainLen):
        u = data[t]
        x = (1-a)*x + a*np.tanh( np.dot( Win, np.vstack((1,u)) ) + np.dot( W, x ) )
        if t >= initLen:
            X[:,t-initLen] = np.vstack((1,u,x))[:,0]
        
    # train the output by ridge regression
    reg = regu #1e-8  # regularization coefficient
    # direct equations from texts:
    #X_T = X.T
    #Wout = np.dot( np.dot(Yt,X_T), linalg.inv( np.dot(X,X_T) + \
    #    reg*np.eye(1+inSize+resSize) ) )
    # using scipy.linalg.solve:
    Wout = linalg.solve( np.dot(X,X.T) + reg*np.eye(1+inSize+resSize), 
        np.dot(X,Yt.T) ).T

    # run the trained ESN in a generative mode. no need to initialize here, 
    # because x is initialized with training data and we continue from there.
    Y = np.zeros((outSize,testLen))
    u = data[trainLen]
    for t in range(testLen):
        x = (1-a)*x + a*np.tanh( np.dot( Win, np.vstack((1,u)) ) + np.dot( W, x ) )
        y = np.dot( Wout, np.vstack((1,u,x)) )
        Y[:,t] = y
        # generative mode:
        u = y
        ## this would be a predictive mode:
        #u = data[trainLen+t+1] 

    # compute MSE for the first errorLen time steps
    errorLen = 500
    mse = sum( np.square( data[trainLen+1:trainLen+errorLen+1] - 
        Y[0,0:errorLen] ) ) / errorLen
    print('MSE = ' + str( mse ))

