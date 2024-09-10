import pandas as pd


data = pd.read_csv("C:\\Users\\vagis\\OneDrive\\Desktop\\vagish\\Datasets\\homeprices_Mul.csv")
print(data.info())


data['bedrooms'].fillna(data['bedrooms'].median(), inplace=True)

inputs = data.drop('price', axis=1)
output = data['price']

print(inputs,output)



