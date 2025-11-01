import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt 

# Read and diplay the dataset
df = pd.read_csv("Data\\creditcard.csv")
# print(df.head())
# print(df["Class"].unique())  # Out: [0, 1], Data is already in factors

# Plotting the features of the dataset to see the differences betweem fraud and legitimate transactions
fig, ax = plt.subplots(5, 6, figsize = (10, 9), layout = 'constrained')  # 5x6 Subplots

for (i, cols) in enumerate(df.columns[:-1]):
    ax[i//6, i%6].hist(df[df["Class"] == 1][cols],  color = 'red', label='fraud', alpha=0.65, density=True)  # Fraudulent
    ax[i//6, i%6].hist(df[df["Class"] == 0][cols],  color = 'blue', label='legitamte', alpha=0.65, density=True)
    ax[i//6, i%6].set_title(cols, fontsize=8)
    ax[i//6, i%6].legend(fontsize=8)

plt.show()
plt.close()

y = df[df.columns[-1]].values
print(y)

Y = np.reshape(y, (-1, 1))
print(Y)
