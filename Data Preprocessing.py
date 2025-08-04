#Import necessary libraries
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import warnings

#Read the dataset
df=pd.read_csv('Life Expectancy Data.csv')
df.columns = df.columns.str.strip()  # remove leading and trailing spaces from column names
df.head() #head of the dataset
df.tail() #tail of the dataset

#Sanity Check of the dataset
df.shape #shape of the dataset
df.info() #info of the dataset
df.isnull().sum() #checking for null values 
df.isnull().sum()/df.shape[0]*100 #percentage of null values in each column
df.duplicated().sum() #checking for duplicate rows
for i in df.select_dtypes(include=['object']).columns:
    print(df[i].value_counts()) #checking garbage values
    print("***" * 10)

#Exploratory Data Analysis (EDA)
df.describe().T #statistical summary of the dataset
df.describe(include='object')
warnings.filterwarnings("ignore") #ignore warnings
for i in df.select_dtypes(include="number").columns:
    sns.histplot(data=df,x=i)
    plt.show()

warnings.filterwarnings("ignore") #ignore warnings
for i in df.select_dtypes(include="number").columns:
    sns.boxplot(data=df,x=i)
    plt.show()
df.select_dtypes(include="number").columns
 
# scatter plot to understand the relationship between two numerical variables
for i in ['Adult Mortality', 'infant deaths', 'Alcohol', 'percentage expenditure', 
          'Hepatitis B', 'Measles', 'BMI', 'under-five deaths', 'Polio', 
          'Total expenditure', 'Diphtheria', 'HIV/AIDS', 'GDP', 
          'Population', 'thinness  1-19 years', 'thinness 5-9 years', 
          'Income composition of resources', 'Schooling']:
    sns.scatterplot(data=df, x=i, y='Life expectancy')
    plt.show()
df.select_dtypes(include="number").corr() #correlation matrix
sns.heatmap(df.select_dtypes(include="number").corr(), annot=True, cmap='coolwarm')

#Missing Value Treatment
df.isnull().sum() #checking for null values again
for i in ["BMI","Polio","Income composition of resources","Schooling"]:
    df[i].fillna(df[i].median(), inplace=True) #filling null values with mean for numerical columns
from sklearn.impute import KNNImputer
imputer = KNNImputer()
num_cols= df.select_dtypes(include=["number"]).columns
df[num_cols] = imputer.fit_transform(df[num_cols])  # KNN imputation for numerical columns

#Outlier Treatment
def wisker(col):
    q1,q3 = np.percentile(col.dropna(), [25, 75])
    iqr = q3 - q1
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr
    return lower_bound, upper_bound

for i in ['GDP','Total expenditure','Hepatitis B','Diphtheria','thinness  1-19 years','thinness 5-9 years']:
    lower_bound, upper_bound = wisker(df[i])
    df[i] = np.where(df[i] < lower_bound, lower_bound, df[i])
    df[i] = np.where(df[i] > upper_bound, upper_bound, df[i])

for i in ['GDP','Total expenditure','Hepatitis B','Diphtheria','thinness  1-19 years','thinness 5-9 years']:
   sns.boxplot(x=df[i]) 
   plt.show()
df.columns 

#Duplicate and Garbage Value Treatment
df.drop_duplicates(inplace=True)  # dropping duplicate rows

#Encoding of Data
dummy=pd.get_dummies(data=df, columns=["Country", "Status"], drop_first=True)  # one-hot encoding for categorical variables
print(dummy)