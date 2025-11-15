import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt 

# Read the dataset
df = pd.read_csv("Data\\creditcard.csv")  # Source: kagglehub - credit card fraud

# Just a test funtion to view the data and formatting
def view():  
    print(df.head())
    print(df["Class"].unique())  # Out: [0, 1], Data is already in factors

# Plotting the features of the dataset to see the differences between fraud and legitimate transactions
def saveDiffPlot():
    fig, ax = plt.subplots(5, 6, figsize = (10, 9), layout = 'constrained')  # 5x6 Subplots

    # A loop for plotting histograms with the 30 features. Using i // 6 for rows and i mod 6 for colunms
    for (i, cols) in enumerate(df.columns[:-1]):
        ax[i//6, i%6].hist(df[df["Class"] == 1][cols],  color = 'red', label='fraud', alpha=0.65, density=True)  # Fraudulent
        ax[i//6, i%6].hist(df[df["Class"] == 0][cols],  color = 'blue', label='legitimate', alpha=0.65, density=True)
        ax[i//6, i%6].set_title(cols, fontsize=8)
        ax[i//6, i%6].legend(fontsize=8)

    # Displaying the plot
    fig.suptitle('Visual difference: Fraudulent and Legitimate transactions')
    plt.savefig('Data\\Visual_difference-Transactions.png')

view()
saveDiffPlot()
