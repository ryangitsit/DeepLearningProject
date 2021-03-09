import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

"""
- The create_dataset function formats data for the echo state network.
- returns input_data
"""

def get_csv_data(dataset):

    df = pd.read_csv("./datasets/" + dataset + ".csv", sep=',')
    #print (df.columns)
    return df

def create_dataset(dataset):

    if dataset == 'mackey':
        dataset = "./datasets/" + dataset + ".txt"
        input_data = np.loadtxt(dataset)


    elif dataset == "wind":
        input_data = get_csv_data(dataset)['ActivePower']
        input_data = np.array(input_data.dropna())/1700


    elif dataset == "weather":
        
        input_data = get_csv_data(dataset)[' _tempm']
        input_data = np.array(input_data.dropna())[0:10000]/70

    elif dataset == "chest":
        input_data = get_csv_data('subject2_chest_1st100k')['Chest_ECG'][:10000]
        input_data = np.array(input_data)


    elif dataset == 'sine':
        x = .1*np.arange(0, 10000, 1)
        input_data = np.array(np.sin(x))
        
    print (f"Input data shape for {dataset} = {input_data.shape} \n of type: {type(input_data)}")

    def close_event():
        plt.close()
    fig = plt.figure()
    timer = fig.canvas.new_timer(interval = 2000) #creating a timer object and setting an interval of 3000 milliseconds
    timer.add_callback(close_event)
    timer.start()
    plt.plot(input_data[:1000])
    plt.title(f'A sample of {dataset} time series')
    plt.show()
    timer.stop()

    return input_data



