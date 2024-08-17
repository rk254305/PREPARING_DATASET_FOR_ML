# import required library first 
import pandas as pd
import numpy as np
import seaborn as sns 
import matplotlib.pyplot as plt 

#for importing data

#path_of_data_including_format
dataset=pd.read_csv(r"C://Users//HOME//jupe/Flipkart_Mobiles.csv")

#for viewing first five column of data
print(dataset.head())

#check for null value if present
print(dataset.isnull().sum())

#filling null value
dataset["Rating"]=dataset["Rating"].fillna(dataset["Rating"].mean())

print(dataset.isnull().sum())

#droping some missing value
dataset.dropna(inplace=True)

#check for null value
print(dataset.isnull().sum())

#handling outliers we take 2 columns for outliers selling price and original price
sns.boxplot(x=["Selling Price"],data=dataset)
plt.title("SELLING PRICE")
plt.show()

