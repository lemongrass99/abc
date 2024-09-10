
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

data = pd.read_csv("C:\\Users\\vagis\\OneDrive\\Desktop\\vagish\\Datasets\\homeprices.csv")
print(data.info())
print(data.info())
print(len(data))
print(len(data.columns))
print(data.shape)


inputs = data.drop('price', axis=1)
outputs = data['price']

plt.xlabel('Area')
plt.ylabel('Price')
plt.scatter(inputs, outputs)
plt.show()


