'''DO NOT DELETE ANY PART OF CODE
We will run only the evaluation function.

Do not put anything outside of the functions, it will take time in evaluation.
You will have to create another code file to run the necessary code.
'''

# import statements
from sklearn.cluster import KMeans
import pickle
import pandas as pd
import numpy as np

# other functions

def predict(test_set) :
    # find and load your best model
    # Do all preprocessings inside this function only.
    # predict on the test set provided
    '''
    'test_set' is a csv path "test.csv", You need to read the csv and predict using your model.
    '''
    #Load test data
    df = pd.read_csv(test_set)

    #One hot encoding implementation
    features=df.columns
    #features=features[:-1]
    #target=df.columns[-1]
    one_hot_features=pd.get_dummies(df[features])

    with open("model.pkl", "rb") as f:
      model_dict = pickle.load(f)

    model = model_dict['model']
    cluster_labels = model_dict['cluster_labels']

    test_predicted_clusters = model.predict(one_hot_features)
    prediction= np.array([cluster_labels[i] for i in test_predicted_clusters])

    '''
    prediction is a 1D 'list' of output labels. just a single python list.
    '''
    return prediction


