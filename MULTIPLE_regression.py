#import required libraries
import pandas as pd
import matplotlib.pyplot as plt 
import seaborn as sns 

#load dataset
dataset = pd.read_csv(r"C:\Users\HOME\jupe\Flipkart_Mobiles.csv")
print(dataset.head)

#check for null values
dataset.isnull()


#drop categorical column
columns_to_drop = ['Brand', 'Model','Color','Rating',]
dataset = dataset.drop(columns=columns_to_drop)

#converting datatypes
dataset['Memory'] = dataset['Memory'].str.replace('GB', '', regex=True)
dataset['Memory'] = dataset['Memory'].str.replace('MB', '', regex=True)
dataset['Memory'] = dataset['Memory'].str.replace('1.5', '2', regex=True)
dataset['Storage'] = dataset['Storage'].str.replace('GB', '', regex=True)
dataset['Storage'] = dataset['Storage'].str.replace('MB', '', regex=True)

dataset['Memory'].astype("int64")
dataset['Storage'].astype("int64")
dataset.head()

# check either graph is linear in nature or not
sns.pairplot(data=dataset)
plt.show()

#diving dependent and independent columns
x=dataset.iloc[:,:-1]
y=dataset["Selling Price"]

#split data for training and testing
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)

#create model
from sklearn.linear_model import LinearRegression
lr=LinearRegression()
lr.fit(x_train,y_train)

#check accuracy of model
print(lr.score(x_test,y_test)*100)

print(lr.predict(x_test))