# import pandas as pd
# import numpy as np
# from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import StandardScaler
# from sklearn.linear_model import LinearRegression
# from sklearn.metrics import mean_squared_error
# df=pd.read_csv("C:\\Users\\vagis\\OneDrive\\Desktop\\vagish\\Datasets\\StudentStudyHour.csv.xls")
# df
# y=df['Scores']
# x=df.drop('Scores',axis=1)

# x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3,random_state=10)
# print(x_train)
# print(x_test)
# scalar=StandardScaler()
# scalar.fit(x_test)
# x_train=scalar.transform(x_train)
# x_test=scalar.transform(x_test)
# print(x_test)
# print(x_test)

# lr=LinearRegression
# model=lr.fit(x_train,y_train)
# y_pred=model.predict(x_test)
# df=pd.DataFrame([y_test:y_test, y_pred : y_pred])
# df
# print(mean_squared_error(y_test,y_pred))




import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

df = pd.read_csv("C:\\Users\\vagis\\OneDrive\\Desktop\\vagish\\Datasets\\StudentStudyHour.csv.xls")

y = df['Scores']
x = df.drop('Scores', axis=1)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=10)
scaler = StandardScaler()

x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

lr = LinearRegression()
model = lr.fit(x_train, y_train)
y_pred = model.predict(x_test)
df_pred = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})
print(df_pred)

mse = mean_squared_error(y_test, y_pred)
print("Mean Squared Error:\n", mse)
