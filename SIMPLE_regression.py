#import required libraries
import pandas as pd
import matplotlib.pyplot as plt 
import seaborn as sns 

#load dataset
dataset = pd.read_csv(r"C:\Users\HOME\jupe\Flipkart_Mobiles.csv")
print(dataset.head)

#check for null values
dataset.isnull()

# check either graph is linear in nature or not

plt.figure(figsize=(5,4))
sns.scatterplot(x="Original Price",y="Selling Price",data=dataset)
plt.show()

#diving dependent and independent 
x=dataset[["Original Price"]]
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


y_prd=lr.predict(x)

#check the predicted line
plt.figure(figsize=(5,4))
sns.scatterplot(x="Original Price",y="Selling Price", data=dataset)
plt.plot(dataset["Original Price"],y_prd,c="red")
plt.legend(["Original Price","predicted line"])
plt.show()

lr.predict([["ENTER VALUE FOR PREDICTION"]])  
