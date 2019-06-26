                         #stock-price-prediction#
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

df = pd.read_csv('infy.csv')
print(df.columns.values)
df['Open']=df['Open Price']
df['High']=df['High Price']
df['Low']=df['Low Price']
df['Close']=df['Close Price']
print(df)
df = df[['Close', 'Open', 'High', 'Low']]
print(df)
forecast_col = 'Close'
forecast_out = 3
df['label'] =df[forecast_col].shift(-forecast_out)
print(df)
x = np.array(df.drop(['Close','label'],1))
#print(x)
x_lately=x[-forecast_out:]
print(x_lately)
x =  x[:-forecast_out]
print(x)

y = np.array(df['label'])
y = y[:-forecast_out]
print(y)

x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=0.2)
print(x_train.shape, x_test.shape, y_train.shape, y_test.shape)

clf = LinearRegression() #clf object (part of that class)
clf.fit(x_train, y_train)

confidence = clf.score(x_test, y_test) #coefficient of determination
print('confidence:' , confidence)
forecast_set = clf.predict(x_lately)
print(clf)
#print(df)
print(forecast_set)




