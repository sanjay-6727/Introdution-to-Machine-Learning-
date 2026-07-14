import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.preprocessing import StandardScaler, MinMaxScaler


def eda_analysis(df):

    print("Shape :",df.shape)

    print("\nDuplicates :",df.duplicated().sum())
    df=df.drop_duplicates()

    print()
    df.info()

    print("\nColumns")
    print(df.columns.tolist())

    print("\nSummary")
    print(df.describe(include="all"))

    print("\nMissing Values")
    print(df.isnull().sum())

    plt.figure(figsize=(8,5))
    sns.heatmap(df.isnull(),cbar=False)
    plt.title("Missing Values")
    plt.show()

    numeric=df.select_dtypes(include=np.number).columns.tolist()
    categorical=df.select_dtypes(exclude=np.number).columns.tolist()

    print("\nSkewness")
    print(df[numeric].skew())

    print("\nIQR Outliers")

    for col in numeric:

        q1=df[col].quantile(0.25)
        q3=df[col].quantile(0.75)

        iqr=q3-q1

        low=q1-1.5*iqr
        high=q3+1.5*iqr

        out=df[(df[col]<low) | (df[col]>high)]

        print(col,":",len(out))

    if len(numeric)>0:

        plt.figure(figsize=(12,6))
        sns.boxplot(data=df[numeric])
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()

        df[numeric].hist(figsize=(12,8),bins=20)
        plt.tight_layout()
        plt.show()

        plt.figure(figsize=(8,6))
        sns.heatmap(df[numeric].corr(),annot=True,cmap="coolwarm")
        plt.tight_layout()
        plt.show()

        sns.pairplot(df[numeric])
        plt.show()

        for col in numeric:

            plt.figure(figsize=(6,4))
            sns.kdeplot(df[col],fill=True)
            plt.title(col)
            plt.show()

            plt.figure(figsize=(5,5))
            sns.violinplot(y=df[col])
            plt.title(col)
            plt.show()

            plt.figure(figsize=(5,5))
            stats.probplot(df[col],dist="norm",plot=plt)
            plt.title(col)
            plt.show()

        z=np.abs(stats.zscore(df[numeric],nan_policy="omit"))

        print("\nZ Score Outliers")

        for i,col in enumerate(numeric):
            print(col,":",np.sum(z[:,i]>3))

        ss=StandardScaler()

        standard=pd.DataFrame(
            ss.fit_transform(df[numeric]),
            columns=numeric
        )

        print("\nStandardized Data")
        print(standard.head())

        mm=MinMaxScaler()

        normal=pd.DataFrame(
            mm.fit_transform(df[numeric]),
            columns=numeric
        )

        print("\nNormalized Data")
        print(normal.head())

    if len(categorical)>0:

        for col in categorical:

            plt.figure(figsize=(8,4))
            sns.countplot(x=df[col])
            plt.xticks(rotation=45)
            plt.title(col)
            plt.tight_layout()
            plt.show()

    if len(numeric)>=2:

        plt.figure(figsize=(6,5))
        sns.scatterplot(x=df[numeric[0]],y=df[numeric[1]])
        plt.xlabel(numeric[0])
        plt.ylabel(numeric[1])
        plt.tight_layout()
        plt.show()