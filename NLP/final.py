import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from wordcloud import WordCloud
import nltk
import random as rd
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from sklearn.metrics import confusion_matrix
from sklearn.feature_extraction.text import TfidfVectorizer,CountVectorizer
from sklearn.linear_model import LogisticRegression
nltk.download('stopwords')

class sanity_check():
    ###initialize dataframe
    def __init__(self,name):
        self.df = pd.read_csv(name)
    ###return the data frame
    def get_df(self):
        return self.df
    ### get the value from the target column
    def get_column(self,column):
        return self.df[column]
    ###print the all column
    def attributes(self):
        print("data frame has following columns: " )
        print(self.df.columns.to_list())
        return self.df.columns.to_list()
    ### dimension information
    def dimension(self):
        print("This df has ",self.df.shape[0],"rows and ", 
              self.df.shape[1],"columns")
    #### check data type
    def check_data_type(self,column):
        print(f"Data in {column} is {self.df[column].dtypes} type")
    ### check the ratio of missing value in each column
    def check_null(self,column):
        null_value = self.df[column].isnull().sum()
        #percentage = round(100 * null_value/len(self.df),4)
        print(f"There are {null_value}  missing value in column {column} AND" )
        return null_value
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
    def find_duplicated(self):
        duplicated_data = self.df[self.df.duplicated()]
        print(duplicated_data)

###Here are some plot setting to make our figure more attractive
class plot_setting():
    ####size of figure setting
    def size_setting(self,width, height):
        plt.figure(figsize=(width, height))
    ### change width of the border
    def border_setting(self,width):
        ax = plt.gca()
        ax.spines['top'].set_linewidth(width)
        ax.spines['right'].set_linewidth(width)
        ax.spines['bottom'].set_linewidth(width)
        ax.spines['left'].set_linewidth(width)
        ax.xaxis.set_tick_params(width = width)
        ax.yaxis.set_tick_params(width = width)
    #### Named xlabel and ylabel
    def label_setting(self,xlabel,ylabel):
        plt.xlabel(xlabel,fontsize = 36)
        plt.ylabel(ylabel,fontsize = 36)
    ### Named title
    def title_setting(self,title):
        plt.title(title, fontsize = 60)
    ### modify xytick
    def ticks_label_setting(self,labelsize):
        plt.tick_params(axis='both', labelsize = labelsize)
    def ticks_setting(self,x,y):
        plt.xticks(fontsize = x)
        plt.yticks(fontsize = y)
    ### rotate xticks
    def rotation(self,angle):
        plt.xticks(rotation = angle)
plot_setting = plot_setting()   



#### start to do sanity check in books_data df
df_data = sanity_check("Data/books_data.csv")
missing_count = []
column_list = df_data.attributes()
df_data.dimension()
df_data.find_duplicated()
for element in column_list:
    print("#########")
    count = df_data.check_null(element)
    missing_count.append(count)
    df_data.check_data_type(element)
df_data.find_outlier("ratingsCount")
plot_setting.size_setting(20,12)
plot_setting.border_setting(4)
plot_setting.ticks_setting(16,16)
plot_setting.rotation(15)
plot_setting.ticks_label_setting(24)
plot_setting.label_setting("Column","Missing value count")
plot_setting.title_setting("Count Missing value in books_data df")
plt.bar(column_list,missing_count)
plt.tight_layout()
plt.savefig("Missing value ratio in data df.png")
plt.show()


##### start to do sanity check in books_data df
df_rating = sanity_check("Data/Books_rating.csv")
#df_rating.print_out()
missing_count = []
column_list = df_rating.attributes()
df_rating.dimension()
df_rating.find_duplicated()
for element in column_list:
    print("######")
    count = df_rating.check_null(element)
    missing_count.append(count)
    df_rating.check_data_type(element)
df_rating.find_outlier("review/score")    
df_rating.find_outlier("Price")    
plot_setting.size_setting(20,12)
plot_setting.border_setting(4)
plot_setting.ticks_setting(16,16)
plot_setting.rotation(15)
plot_setting.ticks_label_setting(24)
plot_setting.label_setting("Column","Missing value count")
plot_setting.title_setting("Count Missing value in books_rating df")
plt.gca().get_yaxis().get_major_formatter().set_scientific(False)
plt.bar(column_list,missing_count)
plt.tight_layout()
plt.savefig("Missing value ratio in rating df.png")
plt.show()


df_rating = sanity_check("Data/Books_rating.csv")
reviews_sub = df_rating.get_df()[["review/score","review/summary","review/text"]]
reviews_sub = reviews_sub.sample(frac=0.5, random_state=42)


####Classify reviews into two ‘sentiment’ categories 
####called positive and negative
reviews_sub['sentiment'] = ['positive' if x >= 3.0 else
'negative' for x in reviews_sub["review/score"]]


#####start to generate word cloud for review/text based positive sentiment category
nltk_stops = set(stopwords.words('english'))
#### we customizer more stopword to optimze the result and derease the computational loading
custom_stops = set(['I',"a", 'you', 'he', 'she', 'it', 'we',"the" ,"read","quot","could"
                    'they', 'me', 'him', 'her', 'us', 'them', 'my', 'your', "author","authors"
                    'his', 'its', 'our', 'their', 'mine', 'yours', 'theirs', 
                    'this', 'that', 'these', 'those', 'here', 'there', 'now', 
                    'then', 'when', 'where', 'why', 'how', 'what', 'which', 'who',
                      'whose', 'whom', 'because', 'since', 'although', 
                    'though', 'even', 'so', 'if', 'unless', 'until', 'while', 
                    'after', 'before', 'since', 'as',  'about', 'for', 
                    'with', 'without', 'by', 'through', 'into', 'onto', 'off', 
                    'on', 'up', 'down', 'above', 'below', 'over', 'under', 
                    'beside', 'near', 'far', 'away', 'behind', 'in front of', 
                    'inside', 'outside', 'between', 'among', 'through', 'across', 
                    'around', 'beside', 'along', 'against', 'before', 'after', 'during',
                    'while', 'since', 'until', 'about', 'against', 'toward', 'among', 'via', 
                    'toward', 'within', 'among', 'concerning',"book","one","would","books","to",
                    "of",'!', ',', '.', "'s", '...', '?', '(', ')', '[', ']', '{', '}', '-', '_', 
                    ':', ';', "'", '"', '/', '\\', '|', '@', '#', '$', '%', '^', '&', '*', '+', '=',
                    '<', '>', '`', '~',"n't","''","``"])

stops = nltk_stops.union(custom_stops)


#####start to generate word cloud for review/summary based positive sentiment category
positive_summary = ' '.join(str(summary) for summary in reviews_sub[reviews_sub['sentiment'] == 'positive']['review/summary'] if isinstance(summary, str))
wordcloud = WordCloud(stopwords=stops).generate(positive_summary)
wordcloud.to_file("positive_summary_wordcloud.png")

#####start to generate word cloud for review/summary based negative sentiment category
negative_summary = ' '.join(str(summary) for summary in reviews_sub[reviews_sub['sentiment'] == 'negative']['review/summary'] if isinstance(summary, str))
wordcloud = WordCloud(stopwords=stops).generate(negative_summary)
wordcloud.to_file("negative_summary_wordcloud.png")

#####start to generate word cloud for review/text based positive sentiment category
positive_text = ' '.join(str(summary) for summary in reviews_sub[reviews_sub['sentiment'] == 'positive']['review/text'] if isinstance(summary, str))
wordcloud = WordCloud(stopwords=stops).generate(positive_text)
wordcloud.to_file("positive_text_wordcloud.png")
  
#####start to generate word cloud for review/text based negative sentiment category
negative_text = ' '.join(str(summary) for summary in reviews_sub[reviews_sub['sentiment'] == 'negative']['review/text'] if isinstance(summary, str))
wordcloud = WordCloud(stopwords=stops).generate(negative_text)
wordcloud.to_file("negative_text_wordcloud.png")

####plot the and count the rating based on the review/score column
data = df_rating.get_column("review/score")
plot_setting.size_setting(20,12)
plot_setting.border_setting(4)
plot_setting.ticks_setting(16,16)
plot_setting.rotation(5)
plot_setting.ticks_label_setting(24)
plot_setting.label_setting("Value","Count")
plot_setting.title_setting("Rating value count")
plt.gca().get_yaxis().get_major_formatter().set_scientific(False)
plt.hist(data, bins=range(1,7), align='left', rwidth=0.8)
plt.xlim(0.5, 5.5) 
plt.savefig("Rating plot.png")
plt.show()

#####predict the sentiment category based on a textbased review
df = df_rating.get_df()[["review/score","review/summary","review/text"]]

def cleanData(reviews_df):
    """ Treat missing values in the review/score and review/summary variables. 
    In this case, dropna has been chosen as the strategy.   """
    reviews_df.dropna(subset=["review/score",'review/text'], inplace=True)
    return reviews_df

def remove_stopwords(token_list):
    tokensFiltered = [token for token in token_list if token not in stops]
    return tokensFiltered

def remove_punc_stopwords(text):
    text_tokens = word_tokenize(text)
    text_tokens = remove_stopwords(text_tokens)
    return " ".join(text_tokens)

def runTextPredictions(reviews_df,frac):
    reviews_df = reviews_df.sample(frac=frac, random_state=42)
    """ Remove punctuations and stopwords from the text data in review/text and review/summary"""
    #reviews_df['review/summary'] = [str(x) for x in reviews_df['review/summary']]
    reviews_df['review/text'] = [str(x) for x in reviews_df['review/text']]
    reviews_df['review/text'] = reviews_df['review/text'].apply(remove_punc_stopwords)

    
    #reviews_df['combined_text'] = reviews_df['review/summary'] + ' ' + reviews_df['review/text']

    # Apply remove_punc_stopwords function to combined_text
    #reviews_df['processed_text'] = reviews_df['combined_text'].apply(remove_punc_stopwords)

    #The following applies remove_punc_stopwords function to each value in the given column. The result is a column with lower case values
    # that have no punctuations, no stop words
    #reviews_df['review/summary'] = reviews_df['review/summary'].apply(remove_punc_stopwords)
    #reviews_df['review/text'] = reviews_df['review/text'].apply(remove_punc_stopwords)
    """ Add a new variable called sentiment; if Rating is greater than or equal to 3, then sentiment = 1, else sentiment = -1 """
    reviews_df['sentiment_value'] = [1 if x >= 3 else -1 for x in reviews_df["review/score"]]

    """ split the dataset into two: train (85% of the obs.) and test (15% of the obs.)"""
    reviews_sub_train =  reviews_df.sample(frac=0.85)
    reviews_sub_test  =  reviews_df.sample(frac=0.15)

    reviews_df['random_index'] = [rd.uniform(0, 1) for _ in range(len(reviews_df))]

    reviews_sub_train = reviews_df[reviews_df.random_index < 0.85][
        ["review/score",'review/text', 'sentiment_value']]
    reviews_sub_test = reviews_df[reviews_df.random_index >= 0.15][
        ["review/score",'review/text', 'sentiment_value']]

    print(reviews_sub_train.head(10))
    print(reviews_sub_test.head(10))
    vectorizer = CountVectorizer(token_pattern=r'\b\w+\b')
    #vectorizer = TfidfVectorizer(token_pattern=r'\b\w+\b')
    train_matrix = vectorizer.fit_transform(reviews_sub_train['review/text'])
    test_matrix = vectorizer.transform(reviews_sub_test['review/text'])

    """Perform Logistic Regression"""
    lr = LogisticRegression()
    X_train = train_matrix
    X_test = test_matrix
    y_train = reviews_sub_train['sentiment_value']
    y_test = reviews_sub_test['sentiment_value']
    lr.fit(X_train, y_train)
    print("Coefficients:")
    print(lr.coef_)
    print("Intercept:")
    print(lr.intercept_)

    """ Generate the predictions for the test dataset"""
    predictions = lr.predict(X_test)
    reviews_sub_test['predictions'] = predictions
    print(reviews_sub_test.head(30))

    """Calculate the prediction accuracy"""
    reviews_sub_test['match'] = y_test == reviews_sub_test['predictions']

    print("")
    print("Prediction Accuracy:")
    sentiment_accuracy = sum(reviews_sub_test['match']) / len(reviews_sub_test)
    print(sentiment_accuracy)

    ## Multinomial Logisitic Regression Model
    lr2 = LogisticRegression()

    X_train = train_matrix
    X_test = test_matrix
    y_train2 = reviews_sub_train["review/score"]
    y_test2 = reviews_sub_test["review/score"]

    lr2.fit(X_train, y_train2)
    print(lr2)

    """ Generate the predictions for the test dataset"""
    predictions2 = lr2.predict(X_test)
    reviews_sub_test['predictions_rating'] = predictions2
    print(reviews_sub_test.head(30))

    reviews_sub_test['match_rating'] = y_test2 == reviews_sub_test['predictions_rating']

    print("")
    print("Prediction Accuracy:")
    rating_accuracy = sum(reviews_sub_test['match_rating']) / len(reviews_sub_test)
    print(rating_accuracy)
    return sentiment_accuracy, rating_accuracy

print(runTextPredictions(df,0.5))


#### store all data from the prediction and visualize them.
methods = ['TfidfVectorizer', 'CountVectorizer', 'TfidfVectorizer', 'CountVectorizer', 'TfidfVectorizer', 'CountVectorizer']
sentiment_acc = [0.935805845, 0.934306505, 0.91465606, 0.914224796, 0.932225432, 0.934200781]
rating_acc = [0.69477824, 0.695081851, 0.650804308, 0.655848017, 0.689011635, 0.697822096]
x = np.arange(len(methods))
width = 0.4
fig, ax = plt.subplots(figsize=(20, 12))

rects1 = ax.bar(x - width/2, [x * 100 for x in sentiment_acc] , width, label='Sentiment Accuracy', color='b')
rects2 = ax.bar(x + width/2, [x * 100 for x in rating_acc], width , label='Rating Accuracy', color='r')
plot_setting.label_setting("Vectorizer","Accuracy (%)")
plot_setting.border_setting(4)
ax.set_xticks(x)
ax.set_xticklabels(methods)
ax.legend(prop={'size': 36})
plot_setting.ticks_setting(16,16)
plot_setting.rotation(15)
plot_setting.ticks_label_setting(24)
plt.tight_layout()
plt.savefig("prediction result.png")
plt.show()




















































