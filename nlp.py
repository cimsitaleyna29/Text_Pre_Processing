
# Text PreProcessing

# import numpy as np
import pandas as pd
# import seaborn as sns
# import matplotlib.pyplot as plt

pd.set_option('display.max_columns', None)
pd.set_option('display.width', 500)

df = pd.read_csv("C:/Users/PC/Desktop/amazon_reviews.csv", sep=",")
df.head()
df.info()
df['reviewText']


########################
# Normalizing Case Folding
########################
# string ifade yer aldığı için bütün satırlar belirli bir standarta koyuldu ve büyük-küçük harf dönüşümü gerçekleşti.
df['reviewText'] = df['reviewText'].str.lower()


#########################
# Punctuations
#########################

# regular expression
# Metinde herhangi bir noktalama işareti görüldüğünde boşluk ile değiştir.
df['reviewText'] = df['reviewText'].str.replace('[^\w\s]', '')


######################
# Numbers
#####################

# ilgili text içerisindeki sayıları yakala sonra boşluk ile değiştir.
df['reviewText'] = df['reviewText'].str.replace('\d', '')


######################
# Stopwords
#####################

# metinlerde herhangi bir anlamı olmayan-barınan yaygın kullanılan kelimeleri at.

from nltk.corpus import stopwords
import nltk
nltk.download('stopwords')
sw = stopwords.words('english')
df['reviewText'] = df['reviewText'].apply(lambda x: " ".join(x for x in str(x).split() if x not in sw))

####################
# Rarewords
####################

# Nadir geçen kelimelerin örüntü oluşturamayacağını varsayarak onları çıkartma işlemi.
# bir kelime ne kadar sıklıkta geçiyor.

import pandas as pd
temp_df = pd.Series(' '.join(df['reviewText']).split()).value_counts()
drops = temp_df[temp_df <= 1]
df['reviewText'] = df['reviewText'].apply(lambda x: " ".join(x for x in str(x).split() if x not in drops))



#####################
# Tokenization
#####################
# cümleleri parçalamak birimleştirmek.
nltk.download('punkt')
from textblob import TextBlob
df['reviewText'].apply(lambda x: TextBlob(x).words).head()



######################
# Lemmatization
#####################

# kelimeleri köklerine indirgemek
# (stemming) ayrıca  buda bir köklerine ayırma işlemidir.
import nltk
nltk.download('wordnet')
from nltk.corpus import wordnet
from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()
df['reviewText'].apply(lambda x: " ".join([lemmatizer.lemmatize(word) for word in x.split()]))
df['reviewText'] = df['reviewText'].apply(lambda x: " ".join([lemmatizer.lemmatize(word) for word in x.split()]))




























