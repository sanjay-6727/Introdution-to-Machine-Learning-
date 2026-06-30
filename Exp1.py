import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


df = pd.read_csv("https://raw.githubusercontent.com/Apaulgithub/oibsip_taskno4/main/spam.csv", encoding='ISO-8859-1')
     
print("EDA ANALYSIS\n")
print(df.shape)
print(df.duplicated())
print(df.info(),"\n",df.describe(),"\nColumn Names\n",df.columns)

print("missing value check")

print(df.isnull().sum())
plt.subplot(4, 2, 1)
sns.heatmap(df.isnull(),cbar=False)
plt.title("Null values removal")


df = df.drop(columns=["Unnamed: 2", "Unnamed: 3", "Unnamed: 4"])
df.columns=["label","msg"]

plt.subplot(4, 2, 2)
sns.heatmap(df.isnull(),cbar=False)
plt.title("Null values removal")

print("DUPLICATES CHECK",df.duplicated().sum())
df=df.drop_duplicates()
print("DUPLICATES CHECK",df.duplicated().sum())

print(df['label'].value_counts())

df["message_length"]=df["msg"].apply(len)
plt.subplot(4, 2, 3)
sns.histplot(df["message_length"],bins=50)
plt.xlabel("Message Length")
plt.title("Distribution of message length")

plt.subplot(4, 2, 4)
sns.boxplot(x="label",y="message_length",data=df)
plt.title("Spam vs Ham Message Length")


df["word_count"]=df["msg"].apply(lambda x: len(x.split()))
print(df.head())

plt.subplot(4, 2, 5)
sns.boxplot(x="label",y="word_count",data=df)
plt.title("Spam VS Ham Word Count")

plt.subplot(4,2,6)
sns.histplot(data=df,x="word_count",hue="label",stat="density",bins=30)
plt.title("Correlation between word count and labels")

plt.subplot(4,2,7)
sns.scatterplot(x="message_length", y="word_count", hue="label", data=df, alpha=0.6)
plt.title("Pairplot View: Message Length vs Word Count")


plt.tight_layout()
plt.savefig("complete_eda_dashboard.eps", format="eps", dpi=600, bbox_inches="tight")
plt.show()
