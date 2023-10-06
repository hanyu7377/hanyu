import pandas as pd
import matplotlib.pyplot as plt
import numpy as np




class sanity_check():
    ###initialize dataframe
    def __init__(self,name):
        self.df = pd.read_csv(name)
    ###print the data frame
    def print_out(self):
        print(self.df)
    ###print the column
    def attributes(self):
        print("data frame has following columns: " )
        print(self.df.columns.to_list())
        return self.df.columns.to_list()
    ### dimension information
    def dimension(self):
        print("This df has ",self.df.shape[0],"rows and ", self.df.shape[1],"columns")
    #### check data type
    def check_data_type(self,column):
        print(f"Data in {column} is {self.df[column].dtypes} type")
    ### check the ratio of missing value in each column
    def check_null(self,column):
        null_value = self.df[column].isnull().sum()
        percentage = round(100 * null_value/len(self.df),4)
        print(f"There are {percentage} % missing value in column {column} AND" )
        return percentage
    #### find the outlier for int or float data type column
    def find_outlier(self,column):
        data = self.df[column]
        Q3 = np.percentile(data,75)
        Q1 = np.percentile(data,25)       
        IQR = Q3-Q1
        threshold = 1.5 * IQR
        outlier = [x for x in data if (x < Q1-threshold) or (x > Q3 > threshold)]
        print(f"There are {len(outlier)} outlier in {column} column")
        return outlier
class plot_setting():
    def size_setting(width, height):
        plt.figure(figsize=(width, height))
    def border_setting(width):
        ax = plt.gca()
        ax.spines['top'].set_linewidth(width)
        ax.spines['right'].set_linewidth(width)
        ax.spines['bottom'].set_linewidth(width)
        ax.spines['left'].set_linewidth(width)
        ax.xaxis.set_tick_params(width = width)
        ax.yaxis.set_tick_params(width = width)
    def label_setting(xlabel,ylabel):
        plt.xlabel(xlabel,fontsize = 24)
        plt.ylabel(ylabel,fontsize = 24)
    def title_setting(title):
        plt.title(title, fontsize = 30)
    def ticks_setting(x,y):
        plt.xticks(fontsize=x)
        plt.yticks(fontsize =y)
    def rotation(angle):
        plt.xticks(rotation = angle)




 
#### start to do sanity check in books_data df
df_data = sanity_check("Data/books_data.csv")
#df_data.print_out()
missing_ratio = []
column_list = df_data.attributes()
df_data.dimension()

for element in column_list:
    print("#########")
    ratio = df_data.check_null(element)
    missing_ratio.append(ratio)
    df_data.check_data_type(element)
df_data.find_outlier("ratingsCount")



plot_setting.size_setting(12,12)
plot_setting.border_setting(4)
plot_setting.ticks_setting(16,16)
plot_setting.rotation(5)
plot_setting.label_setting("Column","Missing value ratio (%)")
plot_setting.title_setting("Count Missing value ratio in books_data df")
plt.bar(column_list,missing_ratio)
plt.tight_layout()
plt.show()
##### start to do sanity check in books_data df
df_rating = sanity_check("Data/Books_rating.csv")
#df_rating.print_out()
missing_ratio = []
column_list = df_rating.attributes()
df_rating.dimension()
for element in column_list:
    print("######")
    ratio = df_rating.check_null(element)
    missing_ratio.append(ratio)
    df_rating.check_data_type(element)
df_rating.find_outlier("review/time")
df_rating.find_outlier("review/score")    

plot_setting.size_setting(12,12)
plot_setting.border_setting(4)
plot_setting.ticks_setting(16,16)
plot_setting.rotation(5)
plot_setting.label_setting("Column","Missing value ratio (%)")
plot_setting.title_setting("Count Missing value ratio in books_rating df")
plt.bar(column_list,missing_ratio)
plt.tight_layout()
plt.show()






