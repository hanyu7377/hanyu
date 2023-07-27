import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

from mlxtend.preprocessing import minmax_scaling

df = pd.read_csv("Building_Permits.csv",encoding="ISO-8859-1")
#print(df)
### get missing data in each column
missing_data = df.isnull().sum()
#print(missing_data[1:10])

#### calculation the proportion of missing data across all data points
all = np.product(df.shape)
total_missing = missing_data.sum()
proportion = round(total_missing/all * 100, 2)
#print(proportion,"pertange of data is missing")

###drop missing value 
#### axis = 1 means this operation is based on the column
column_drop_df = df.dropna(axis = 1)
#print(column_drop_df)
###calculate how many columns be removed
#### loss 31 columns and it means get rid of a lot data
#print(df.shape[1] - column_drop_df.shape[1])

####Various data imputation
###fill blank with NaN
df["Street Number Suffix"].replace("",np.nan ,inplace = True)
df["Unit"].replace("", np.nan, inplace = True)
#print(df["Street Number Suffix"])
#print(df["Unit"])


#######scale and normalization
# obtain data from Street Number column
min_value = df['Street Number'].min() 
max_value = df['Street Number'].max() 
scaled_data = (df['Street Number'] - min_value) / (max_value - min_value)
normalized_data = stats.boxcox(df["Street Number"])[0]

### plot both together to compare
### figsize (10,5) 代表加大圖的寬度, sharey代表共用y軸
fig, ax = plt.subplots(1, 3,figsize= (18,5))
####ax = ax[0] 是為了將第一筆data 畫在第一個子圖上
sns.histplot(df['Street Number'], ax=ax[0])
ax[0].set_title("Original Data")
ax[0].set_xlabel("Street Number")
####ax = ax[1] 是為了將第一筆data 畫在第二個子圖上
sns.histplot(scaled_data, ax=ax[1], color = "red")
ax[1].set_title("Scaled Data")
ax[1].set_xlabel("Street Number")
sns.histplot(normalized_data, ax=ax[2],color = "orange",legend = False)
ax[2].set_title("Normalized Data")
ax[2].set_xlabel("Street Number")
plt.show()

