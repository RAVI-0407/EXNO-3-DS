## EXNO-3-DS

# AIM:
To read the given data and perform Feature Encoding and Transformation process and save the data to a file.

# ALGORITHM:
STEP 1:Read the given Data.
STEP 2:Clean the Data Set using Data Cleaning Process.
STEP 3:Apply Feature Encoding for the feature in the data set.
STEP 4:Apply Feature Transformation for the feature in the data set.
STEP 5:Save the data to the file.

# FEATURE ENCODING:
1. Ordinal Encoding
An ordinal encoding involves mapping each unique label to an integer value. This type of encoding is really only appropriate if there is a known relationship between the categories. This relationship does exist for some of the variables in our dataset, and ideally, this should be harnessed when preparing the data.
2. Label Encoding
Label encoding is a simple and straight forward approach. This converts each value in a categorical column into a numerical value. Each value in a categorical column is called Label.
3. Binary Encoding
Binary encoding converts a category into binary digits. Each binary digit creates one feature column. If there are n unique categories, then binary encoding results in the only log(base 2)ⁿ features.
4. One Hot Encoding
We use this categorical data encoding technique when the features are nominal(do not have any order). In one hot encoding, for each level of a categorical feature, we create a new variable. Each category is mapped with a binary variable containing either 0 or 1. Here, 0 represents the absence, and 1 represents the presence of that category.

# Methods Used for Data Transformation:
  # 1. FUNCTION TRANSFORMATION
• Log Transformation
• Reciprocal Transformation
• Square Root Transformation
• Square Transformation
  # 2. POWER TRANSFORMATION
• Boxcox method
• Yeojohnson method

# CODING AND OUTPUT:

      from google.colab import drive
drive.mount('/content/drive')

ls drive/MyDrive/'Colab Notebooks'/

# **ENDODING**

import pandas as pd
import numpy as np

df=pd.read_csv('drive/MyDrive/Data Science/Encoding Data.csv')
df

### **ORDINAL ENCODER**

from sklearn.preprocessing import LabelEncoder, OrdinalEncoder

pm= ['Hot','Warm','Cold']

en1 = OrdinalEncoder(categories = [pm])

en1.fit_transform(df[["ord_2"]])

df['bo2']=en1.fit_transform(df[["ord_2"]])
df

### **LABLE ENCODER**

le=LabelEncoder()

dfc=df.copy()

dfc['ord_2'] = dfc['ord_2'].astype(str)

dfc

## **ONE HOT ENCODER**

from sklearn.preprocessing import OneHotEncoder

One=OneHotEncoder(sparse_output=False)
df2=df.copy()

enc=pd.DataFrame(One.fit_transform(df2[['nom_0']]))

df2=pd.concat([df2,enc],axis=1)
df2

pd.get_dummies(df2,columns=["nom_0"])

## **BINARY ENCODER**

pip install --upgrade category_encoders

from category_encoders import BinaryEncoder

df=pd.read_csv("drive/MyDrive/Data Science/data.csv")
df

be=BinaryEncoder()

nd=be.fit_transform(df['Ord_2'])
dfb=pd.concat([df,nd],axis=1)
dfb1=df.copy()

dfb1

## **TARGET ENCODER**

from category_encoders import TargetEncoder

te=TargetEncoder()

cc=df.copy()

new=te.fit_transform(X=cc["City"],y=cc["Target"])
cc=pd.concat([cc,new],axis=1)
cc

# **TRANSFORMATION**

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
import scipy.stats as stats

from sklearn.preprocessing import QuantileTransformer

df=pd.read_csv('drive/MyDrive/Data Science/Data_to_Transform.csv')
df

df.skew()

np.log(df["Highly Positive Skew"])

np.reciprocal(df["Moderate Positive Skew"])

np.sqrt(df["Highly Positive Skew"])

np.square(df["Highly Positive Skew"])

df["Highly Positive Skew_boxcox"],parameters=stats.boxcox(df["Highly Positive Skew"])
df

df["Moderate Negative Skew_yeojohnson"], lmbda = stats.yeojohnson(df["Moderate Negative Skew"])

df.skew()

df["Highly Negative Skew_yeojohnson"],parameters=stats.yeojohnson(df["Highly Negative Skew"])

df.skew()

from sklearn.preprocessing import QuantileTransformer

qt=QuantileTransformer(output_distribution='normal')

df["Moderate Negative Skew_1"]=qt.fit_transform(df[["Moderate Negative Skew"]])
df

import seaborn as sns
import statsmodels.api as sm
import matplotlib.pyplot as plt

sm.qqplot(df["Moderate Negative Skew"],line='45')
plt.show()

sm.qqplot(np.reciprocal(df["Moderate Negative Skew"]),line='45')
plt.show()

from sklearn.preprocessing import QuantileTransformer
qt=QuantileTransformer(output_distribution='normal',n_quantiles=891)

df["Moderate Negative Skew"]=qt.fit_transform(df[["Moderate Negative Skew"]])

sm.qqplot(df["Moderate Negative Skew"],line='45')
plt.show()

df["Highly Negative Skew_1"]=qt.fit_transform(df[["Highly Negative Skew"]])
sm.qqplot(df['Highly Negative Skew'],line='45')
plt.show()

sm.qqplot(df['Highly Negative Skew_1'],line='45')
plt.show()

dt=pd.read_csv("drive/MyDrive/Data Science/titanic_dataset.csv")

from sklearn.preprocessing import QuantileTransformer
qt=QuantileTransformer(output_distribution='normal',n_quantiles=891)

dt["Age_1"]=qt.fit_transform(dt[["Age"]])
sm.qqplot(dt['Age'],line='45')
plt.show()

sm.qqplot(dt['Age_1'],line='45')
plt.show()
![Screenshot 2024-11-21 184841](https://github.com/user-attachments/assets/0d519375-71e4-4cd8-a5ab-19500da83f0c)
![Screenshot 2024-11-21 184834](https://github.com/user-attachments/assets/7eed7500-5b01-4887-bf41-9c14890becbd)
![Screenshot 2024-11-21 184828](https://github.com/user-attachments/assets/c4871884-dc12-47f5-a49b-a44628d5268c)
![Screenshot 2024-11-21 184821](https://github.com/user-attachments/assets/bee06fc9-75d6-4b37-ae12-bbeb56993d2d)
![Screenshot 2024-11-21 184815](https://github.com/user-attachments/assets/930dff31-cc92-4d7b-99f6-80c8519b7672)
![Screenshot 2024-11-21 184805](https://github.com/user-attachments/assets/7387b0ee-2dbd-4008-82e7-f3a13ea59089)
![Screenshot 2024-11-21 184754](https://github.com/user-attachments/assets/0d5b4565-19c8-452c-9f54-aab0d127c091)
![Screenshot 2024-11-21 184707](https://github.com/user-attachments/assets/aebf2fc0-d89d-4ab2-808a-10270934577f)
![Screenshot 2024-11-21 184701](https://github.com/user-attachments/assets/de21f207-6ce9-4fc7-a705-e10bde22ce18)
![Screenshot 2024-11-21 184655](https://github.com/user-attachments/assets/741a9496-f536-4676-be23-b0d2758d2f28)
![Screenshot 2024-11-21 184649](https://github.com/user-attachments/assets/4899b74d-349f-45eb-b64e-d3992e67db7b)
![Screenshot 2024-11-21 184455](https://github.com/user-attachments/assets/0903281e-a5df-4a2a-8fcd-5de25189ac61)
![Screenshot 2024-11-21 185030](https://github.com/user-attachments/assets/d57abcda-7778-46e0-9d71-dd18ec5fcfcc)
![Screenshot 2024-11-21 185024](https://github.com/user-attachments/assets/777e2877-77d3-4b74-baa8-9706430764d5)
![Screenshot 2024-11-21 185019](https://github.com/user-attachments/assets/711e3331-1be0-4c0b-a638-6f9daaeea8e4)
![Screenshot 2024-11-21 185012](https://github.com/user-attachments/assets/c688e268-8ecc-466b-96eb-e69cb9cd4d64)
![Screenshot 2024-11-21 185007](https://github.com/user-attachments/assets/105a5042-2f93-4622-a9ea-2e63d9f7b1a4)
![Screenshot 2024-11-21 185001](https://github.com/user-attachments/assets/a8cc21a8-8849-4a3e-afaf-8c9cc5f5e0fd)
![Screenshot 2024-11-21 184955](https://github.com/user-attachments/assets/133d291d-2658-4fc6-84b5-b7c169cbe2c0)
![Screenshot 2024-11-21 184944](https://github.com/user-attachments/assets/1a8473db-08fc-4669-a8b0-fbbd83a8b904)
![Screenshot 2024-11-21 184933](https://github.com/user-attachments/assets/36082e18-1e9b-4584-bd1b-3a29579da22c)
![Screenshot 2024-11-21 184927](https://github.com/user-attachments/assets/69f1e973-3664-4bfc-8350-38e7a6a2f770)
![Screenshot 2024-11-21 184920](https://github.com/user-attachments/assets/cd4af0b3-6c02-4823-8287-282e055f22a0)
![Screenshot 2024-11-21 184909](https://github.com/user-attachments/assets/4bea72d5-42a1-48bb-82d1-b353dfecfff0)
![Screenshot 2024-11-21 184904](https://github.com/user-attachments/assets/c126dd65-1918-4b7c-99dd-7ff5675a0b30)
![Screenshot 2024-11-21 184857](https://github.com/user-attachments/assets/3aeaa324-d90c-4944-8a1d-b99f33899a4f)
![Screenshot 2024-11-21 184850](https://github.com/user-attachments/assets/7b9bf066-6cbb-4c86-961f-e0d9feb9e73a)
![Screenshot 2024-11-21 184846](https://github.com/user-attachments/assets/96d3e627-fe88-4e52-b761-b73f132242ea)

# RESULT:
       # INCLUDE YOUR RESULT HERE

       
