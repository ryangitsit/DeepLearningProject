import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def get_data():

    wind_dat = pd.read_csv("Turbine_Data.csv", sep=',')

    wind_dat.rename(columns = {"Unnamed: 0": "Time"}, inplace=True)

    # print(wind_dat.shape)
    # print(wind_dat.columns)

    return wind_dat

def plot_power(wind_dat):

    plt.plot(np.arange(1,1000,1), wind_dat['ActivePower'][1:1000])
    plt.title("Wind Power Output over Time")
    plt.xlabel("Time (10-minute Intervals)")
    plt.ylabel("Active Power")
    plt.show()


def main():
    wind_dat = get_data()
    plot_power(wind_dat)

main()
