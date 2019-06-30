from __future__ import print_function

import argparse
import os
import pandas as pd

from sklearn.externals import joblib

## TODO: Import any additional libraries you need to define a model
from sklearn.linear_model import SGDClassifier
from sklearn.neural_network import MLPClassifier
from sklearn import svm

# Provided model load function
def model_fn(model_dir):
    """Load model from the model_dir. This is the same model that is saved
    in the main if statement.
    """
    print("Loading model.")
    
    # load using joblib
    model = joblib.load(os.path.join(model_dir, "model.joblib"))
    print("Done loading model.")
    
    return model


## TODO: Complete the main code
if __name__ == '__main__':
    
    # All of the model parameters and training parameters are sent as arguments
    # when this script is executed, during a training job
    
    # Here we set up an argument parser to easily access the parameters
    parser = argparse.ArgumentParser()

    # SageMaker parameters, like the directories for training data and saving models; set automatically
    # Do not need to change
    parser.add_argument('--output-data-dir', type=str, default=os.environ['SM_OUTPUT_DATA_DIR'])
    parser.add_argument('--model-dir', type=str, default=os.environ['SM_MODEL_DIR'])
    parser.add_argument('--data-dir', type=str, default=os.environ['SM_CHANNEL_TRAIN'])
    
    ## TODO: Add any additional arguments that you will need to pass into your model
    parser.add_argument('--iterations', type=int, default=2000)
    parser.add_argument('--hidden_dim', type=int, default=100)
    parser.add_argument('--alg_flag', type=str)
    
    
    # args holds all passed-in arguments
    args = parser.parse_args()
    
    model_type = args.alg_flag

    # Read in csv training file
    training_dir = args.data_dir
    train_data = pd.read_csv(os.path.join(training_dir, "train.csv"), header=None, names=None)

    # Labels are in the first column
    train_y = train_data.iloc[:,0]
    train_x = train_data.iloc[:,1:]
    
    
    ## --- Your code here --- ##
    

    ## TODO: Define a model
    #I will test three different models to identify the model that gives the best learning outcome
    # Alg_flag: 1:SGDClassifier, 2: MLPClassifier, 3:SVM
    if model_type == 'MLPClassifier':
        model = MLPClassifier(hidden_layer_sizes = (args.hidden_dim,), max_iter = args.iterations, solver = 'adam', alpha =.5)
    if model_type == 'SGDClassifier':
        model =SGDClassifier(loss="hinge", penalty="l2", max_iter=args.iterations)
    if model_type == 'SVM':
        model =svm.SVC(gamma='scale', max_iter=args.iterations, )
    
    ## TODO: Train the model
    model.fit(train_x, train_y)
    
    
    ## --- End of your code  --- ##
    

    # Save the trained model
    joblib.dump(model, os.path.join(args.model_dir, "model.joblib"))