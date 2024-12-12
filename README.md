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

![image](https://github.com/user-attachments/assets/6f208659-9414-4ca6-8d8d-d1ad470e85b2)

### **ORDINAL ENCODER**

from sklearn.preprocessing import LabelEncoder, OrdinalEncoder

pm= ['Hot','Warm','Cold']

en1 = OrdinalEncoder(categories = [pm])

en1.fit_transform(df[["ord_2"]])

![image](https://github.com/user-attachments/assets/1de5b2a7-ba89-44aa-938a-25727bee77be)

df['bo2']=en1.fit_transform(df[["ord_2"]])
df


### **LABLE ENCODER**

le=LabelEncoder()

dfc=df.copy()

dfc['ord_2'] = dfc['ord_2'].astype(str)

dfc

![image](https://github.com/user-attachments/assets/8e740dc1-6a58-45b2-8746-343acc25891c)

## **ONE HOT ENCODER**

from sklearn.preprocessing import OneHotEncoder

One=OneHotEncoder(sparse_output=False)
df2=df.copy()

enc=pd.DataFrame(One.fit_transform(df2[['nom_0']]))

df2=pd.concat([df2,enc],axis=1)
df2

![image](https://github.com/user-attachments/assets/1190f9b5-aada-44de-a7cc-57989365fdbb)

pd.get_dummies(df2,columns=["nom_0"])

![image](https://github.com/user-attachments/assets/bda17d86-7261-4a4b-986c-d57b9059c842)

## **BINARY ENCODER**

pip install --upgrade category_encoders

from category_encoders import BinaryEncoder

df=pd.read_csv("drive/MyDrive/Data Science/data.csv")
df

![image](https://github.com/user-attachments/assets/6b014e06-34e2-4331-8a46-57973e44f2d2)

be=BinaryEncoder()

nd=be.fit_transform(df['Ord_2'])
dfb=pd.concat([df,nd],axis=1)
dfb1=df.copy()

dfb1

![image](https://github.com/user-attachments/assets/31cceed1-5b52-450c-9912-b698178acd42)

## **TARGET ENCODER**

from category_encoders import TargetEncoder

te=TargetEncoder()

cc=df.copy()

new=te.fit_transform(X=cc["City"],y=cc["Target"])
cc=pd.concat([cc,new],axis=1)
cc

![image](https://github.com/user-attachments/assets/2e6ed56b-cf26-4745-b794-233714d7b040)

# **TRANSFORMATION**

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
import scipy.stats as stats

from sklearn.preprocessing import QuantileTransformer

df=pd.read_csv('drive/MyDrive/Data Science/Data_to_Transform.csv')
df

![image](https://github.com/user-attachments/assets/2f37aaee-e7d5-4461-b1c6-8a1b68986025)

df.skew()

![image](https://github.com/user-attachments/assets/524e38c7-1a45-4492-8e59-688e1722dd1a)

np.log(df["Highly Positive Skew"])

![image](https://github.com/user-attachments/assets/fb23f998-cda4-44d9-8c3b-e60ae6aa150f)

np.reciprocal(df["Moderate Positive Skew"])

![image](https://github.com/user-attachments/assets/583bba9d-772a-4b51-8462-db36c671e173)

np.sqrt(df["Highly Positive Skew"])

![image](https://github.com/user-attachments/assets/b1af8c3b-077b-4eec-9e3f-4d1bb94b4bb1)

np.square(df["Highly Positive Skew"])

![image](https://github.com/user-attachments/assets/63e6d692-c80a-47d2-b2e9-96c9c162ebd3)

df["Highly Positive Skew_boxcox"],parameters=stats.boxcox(df["Highly Positive Skew"])
df

![image](https://github.com/user-attachments/assets/c03255d2-250b-45e3-9f06-dd8bdd5ed42a)

df["Moderate Negative Skew_yeojohnson"], lmbda = stats.yeojohnson(df["Moderate Negative Skew"])

df.skew()

![image](https://github.com/user-attachments/assets/28318a7a-2013-4033-99f7-6cc86cc858e0)

df["Highly Negative Skew_yeojohnson"],parameters=stats.yeojohnson(df["Highly Negative Skew"])

df.skew()

![image](https://github.com/user-attachments/assets/75da5d40-9fe2-4d20-be93-4e6cf7a20999)

from sklearn.preprocessing import QuantileTransformer

qt=QuantileTransformer(output_distribution='normal')

df["Moderate Negative Skew_1"]=qt.fit_transform(df[["Moderate Negative Skew"]])
df

![image](https://github.com/user-attachments/assets/93e00d44-6da9-4c98-9de7-64534f15eab5)

import seaborn as sns
import statsmodels.api as sm
import matplotlib.pyplot as plt

sm.qqplot(df["Moderate Negative Skew"],line='45')
plt.show()

![image](https://github.com/user-attachments/assets/9e6c45bc-6fc0-484f-801f-ff4fbb25a3c8)

sm.qqplot(np.reciprocal(df["Moderate Negative Skew"]),line='45')
plt.show()

![image](https://github.com/user-attachments/assets/1f89e6dd-bb04-41e4-a783-516d8b5cc749)

from sklearn.preprocessing import QuantileTransformer
qt=QuantileTransformer(output_distribution='normal',n_quantiles=891)

df["Moderate Negative Skew"]=qt.fit_transform(df[["Moderate Negative Skew"]])

sm.qqplot(df["Moderate Negative Skew"],line='45')
plt.show()

![image](https://github.com/user-attachments/assets/71336d4a-904b-4d33-8b5c-d7757c8bbc46)

df["Highly Negative Skew_1"]=qt.fit_transform(df[["Highly Negative Skew"]])
sm.qqplot(df['Highly Negative Skew'],line='45')
plt.show()

![image](https://github.com/user-attachments/assets/2f08652a-ad55-4eab-947d-70e9f0c69f12)

sm.qqplot(df['Highly Negative Skew_1'],line='45')
plt.show()

![image](https://github.com/user-attachments/assets/37f1fa0f-7df1-4b78-98d3-79b25baf66f2)

dt=pd.read_csv("drive/MyDrive/Data Science/titanic_dataset.csv")

from sklearn.preprocessing import QuantileTransformer
qt=QuantileTransformer(output_distribution='normal',n_quantiles=891)

dt["Age_1"]=qt.fit_transform(dt[["Age"]])
sm.qqplot(dt['Age'],line='45')
plt.show()

![image](https://github.com/user-attachments/assets/7fc2cc81-c072-4222-ab5c-ac68368f0bbd)

sm.qqplot(dt['Age_1'],line='45')
plt.show()

![image](https://github.com/user-attachments/assets/fc48d433-c0bd-4455-8c05-32311a51aeb7)

# RESULT:
       successfully performed Feature Encoding and Transformation process

       
