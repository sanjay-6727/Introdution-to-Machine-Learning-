import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris

iris = load_iris(as_frame=True)
df = iris.frame

def eda_analysis(df):
    #shpaes and duplicated values with the info 
    print("SHAPE:",df.shape)

    print("Duplicates Check",df.duplicated().sum())
    df=df.drop_duplicates()
    df.info()
    print("\nSummary Statistics")
    print(df.describe(include="all"))

    print("\nColumn Names")
    print(df.columns)
    #missing vals
    print("missing value check")
    print(df.isnull().sum())

    #null values removal
    
    sns.heatmap(df.isnull(),cbar=False)

    plt.title("Null values removal")
    plt.show()

    numeric_cols=df.select_dtypes(include=np.number).columns.tolist()
    print("IQR to find Outliers in every column");
    for c in numeric_cols:
        q1,q3=df[c].quantile(0.25),df[c].quantile(0.75)
        iqr=q3-q1
        d=iqr*1.5
        low=q1-d
        upper=q3+d
        outliers=df[(df[c]<low)|df[c]>upper]
        print(f"{c}: {len(outliers)} outliers")
        
    #boxplot to show the outliers
    plt.figure(figsize=(10, 6))
    sns.boxplot(data=df)
    plt.show()

    df.hist(bins=30)
    plt.tight_layout()
    plt.show()

    #skewness measures how asymetric the data is 
    print("\nSKEWNESS")
    print(df[numeric_cols].skew())

    sns.pairplot(df)
    plt.show()

eda_analysis(df)        





