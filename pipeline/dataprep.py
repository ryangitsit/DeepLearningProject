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
        
        input_data = get_csv_data(dataset) #[' _tempm']
        input_data = np.array(input_data.dropna())
        
    return input_data



# def data_split(input_data):
#     pass
    
# class WindowGenerator():
#   def __init__(self, input_width, label_width, shift,
#                train_df=train_df, val_df=val_df, test_df=test_df,
#                label_columns=None):
#     # Store the raw data.
#     self.train_df = train_df
#     self.val_df = val_df
#     self.test_df = test_df

#     # Work out the label column indices.
#     self.label_columns = label_columns
#     if label_columns is not None:
#       self.label_columns_indices = {name: i for i, name in
#                                     enumerate(label_columns)}
#     self.column_indices = {name: i for i, name in
#                            enumerate(train_df.columns)}

#     # Work out the window parameters.
#     self.input_width = input_width
#     self.label_width = label_width
#     self.shift = shift

#     self.total_window_size = input_width + shift

#     self.input_slice = slice(0, input_width)
#     self.input_indices = np.arange(self.total_window_size)[self.input_slice]

#     self.label_start = self.total_window_size - self.label_width
#     self.labels_slice = slice(self.label_start, None)
#     self.label_indices = np.arange(self.total_window_size)[self.labels_slice]

#   def __repr__(self):
#     return '\n'.join([
#         f'Total window size: {self.total_window_size}',
#         f'Input indices: {self.input_indices}',
#         f'Label indices: {self.label_indices}',
#         f'Label column name(s): {self.label_columns}'])



# def make_window():
#     w1 = WindowGenerator(input_width=24, label_width=1, shift=24,
#                      label_columns=[' _tempm'])


