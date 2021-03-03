import numpy as np
import pandas as pd

"""
- The create_dataset function formats data for the echo state network.
- returns input_data
"""

def get_csv_data(dataset):

    df = pd.read_csv("./datasets/" + dataset + ".csv", sep=',')
    print (df.columns)
    return df

def create_dataset(dataset):

    if dataset == 'mackey':
        dataset = "./datasets/" + dataset + ".txt"
        input_data = np.loadtxt(dataset)


    elif dataset == "wind":
        input_data = get_csv_data(dataset)['ActivePower']
        input_data = np.array(input_data.dropna())


    elif dataset == "weather":
        
        input_data = get_csv_data(dataset)[' _tempm']
        input_data = np.array(input_data.dropna())
        
    return input_data