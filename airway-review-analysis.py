
from nltk.corpus import stopwords
import matplotlib.pyplot as plt
from nltk.tokenize import word_tokenize
from bs4 import BeautifulSoup
from textblob import TextBlob
from wordcloud import WordCloud
from string import digits
import requests
import re
import string
import seaborn as sns
import pandas as pd
import nltk
nltk.download('stopwords')

nltk.download('punkt')

"""Webscrape and Analysis"""

base_url = "https://www.airlinequality.com/airline-reviews/british-airways"
page=10
page_size=100
reviews=[]

for i in range(1,page+1):
  print(f'scrapping page {i}')
  url= url = f"{base_url}/page/{i}/?sortby=post_date%3ADesc&pagesize={page_size}"
  response=requests.get(url)
  content=response.content
  soup=BeautifulSoup(content,'html.parser')
  for para in soup.find_all('div',{"class":"text_content"}):
    reviews.append(para.get_text())
  print(f'    {len(reviews)}  total reviews')

df = pd.DataFrame()
df["reviews"] = reviews
df.head()

df.to_csv("airways_review")

df1=pd.read_csv('/content/airways_review')
df1.reset_index(drop=True,inplace=True)
print(df1['reviews'])

df1.info()
df1.describe()

"""Preprocessing of Data

"""

df1['reviews'] = df1['reviews'].str.strip()
df1['reviews']=df1['reviews'].str.lstrip('✅ Trip Verified |')
df1['reviews']=df1['reviews'].str.lstrip('Not Verified |')
df1['reviews']= df1['reviews'].str.lower()
print(df1)

df1['reviews'] = df1['reviews'].str.replace('[^\w\s]','')
print(df1['reviews'])

print(df1.iloc[1,1])
df1['reviews'] = df1.apply(lambda row: nltk.word_tokenize(row['reviews']), axis=1)
print(df1.iloc[0,1])

"""Remove StopWords"""

from nltk.corpus import stopwords
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))
df1['reviews'] = df1['reviews'].apply(lambda x: ' '.join([word for word in x if word not in (stop_words)]))
print(df1.head(20))

def polarity_calc(text):
    try:
        return TextBlob(text).sentiment.polarity
    except:
        return None



def tag_cal(num):
    if num<0:
        return 'Negative'
    elif num>0:
        return 'Positive'
    else:
        return 'Neutral'


df1['polarity'] = df1['reviews'].apply(polarity_calc)


df1['tag'] = df1['polarity'].apply(tag_cal)


print(df1)

(df1.groupby('tag').size()/df1['tag'].count())*100

"""Visualizing the results"""

df1['tag'].value_counts().plot(kind='bar', figsize=(7, 6), rot=0)
plt.xlabel("Sentiment")
plt.ylabel("No of reviews")
plt.show()

