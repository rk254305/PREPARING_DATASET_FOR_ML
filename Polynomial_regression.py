import pandas as pd
import matplotlib.pyplot as plt 

#load file
df=pd.read_csv("filepath")
df.head()
#check for null value
df.isnull().sum()

#plot a graph for checking line of data 
plt.scatter(df["independent column"],df["dependent column"])
plt.show()

#split data
x=df['independent_column']
y=df['dependent_column']

#import feature for testing
from sklearn.preprocessing import PolynomialFeatures
pf=PolynomialFeatures(degree=2)
pf.fit(x)
x=pf.transform(x)

#divide dataset for training and testing
from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)

#import linear regression feature
from sklearn.linear_model import LinearRegression
lr = LinearRegression()
lr.fit(x_train,y_train)

#check accuracy
lr.score(x_test,y_test)


prd=lr.predict(x)
#create a graph for predicted line
plt.scatter(df["independent column"],df["dependent column"])
plt.plot(df["independent column"],prd,c="red")
plt.show()
