#!/usr/bin/env python
# coding: utf-8

# # Welcome to The Notebook
# ---
# 
# In this guided-project we are going to cover the following tasks: 
# 
# - Task 1: Loading the customer reviews dataset
# - Task 2: Clean and preprocess the reviews
# - Task 3: Tokenize the reviews and removing the stop words
# - Task 4: Exploring the tokens and product categories
# - Task 5: Extracting the adjectives in the reviews
# - Task 6: Finding the sentiment of each review 
# 

# Importing the modules

# In[1]:


import pandas as pd 
import string 

# Importing Natural Language Processing toolkit 
import nltk

# Downloading the NLTK english stop words
nltk.download('stopwords')

# Downloading the NLTK sentence tokenizer
nltk.download('punkt')

# Downloading the NLTK POS Tagger
nltk.download('averaged_perceptron_tagger')

# Downloading the NLTK Vader Lexicon
nltk.download('vader_lexicon')

# Importing the NLTK english stop words 
from nltk.corpus import stopwords

# Importing frequency distribution from NLTK
from nltk.probability import FreqDist

# Importing VADER dictionary. It is a rule-based sentiment analyzer
from nltk.sentiment import SentimentIntensityAnalyzer

# Importing data visualization modules 
from wordcloud import WordCloud
import plotly.express as px 
import matplotlib.pyplot as plt

print("Modules are imported! :)")


# ## Task 1
# 
# ### Loading the customer reviews dataset
# ---

# In[2]:


data = pd.read_csv('dataset.csv')
data.head()


# Let's check the shape of the data frame

# In[3]:


data.shape


# Let's check the first `product_review`

# In[4]:


data.product_review[0]


# Checking the number of reviews per product category

# In[5]:


data.product_category.value_counts()


# ## Task 2 
# 
# ### Clean and preprocess the reviews
# 
# - Lower casing
# - Removing the punctuations
# 
# ---

# In[6]:


data.head()


# Converting all the reveiews to lower case

# In[7]:


data.head()


# In[8]:


review = data.product_review[0]
review.lower()


# In[9]:


data.product_review = data.product_review.str.lower()
data.head()


# Removing the punctuations

# In[10]:


# punctionation are condiered as noise in data
data.product_review[0].translate(str.maketrans('','',string.punctuation))


# Let's remove the punctuations from all the reviews

# In[11]:


data.product_review = data.product_review.str.translate(str.maketrans('','',string.punctuation))
data.head()


# ## Task 3 
# 
# ### Tokenize the reviews and removing the stop words 
# ---
# - <b>Tokenization</b> is the process of breaking down a continuous stream of text, such as a sentence or a paragraph, into smaller units called tokens. These tokens typically correspond to words, but can also represent subword units like prefixes, suffixes, and stems.
# 
# - <b>Tokenization</b> facilitates the transformation of text into a format that machine learning algorithms can understand.

# In[12]:


nltk.word_tokenize('This a sentense plase tokenize me')
tokens = nltk.word_tokenize(data.product_review[0])
tokens


# Let's remove the Stop Words
#     
# <b>Stop words</b> are common words (e.g., "the," "and," "is") that appear frequently in a language and have little semantic value. Removing them is essential in natural language processing tasks to reduce data size, speed up processing, and improve the accuracy of algorithms by focusing on more informative words that convey the actual meaning of a text.
# 

# In[13]:


import nltk

# Downloading the NLTK english stop words
nltk.download('stopwords')

from nltk.corpus import stopwords


english_stopwords = stopwords.words("english")
english_stopwords.extend(['im','its','youre','thing','cant','dont','doesnt'])
english_stopwords


# Let's remove the stop words from the `tokens` list

# In[14]:


[t for t in tokens if t not in english_stopwords]


# Let's tokenize all the reviews 

# In[15]:


data['product_review_tokenize'] = data.product_review.apply(nltk.word_tokenize)
data.head()


# remove the stopwords from the tokenized reviews

# In[18]:


def remove_stopwords(tokens):
    return [t for t in tokens if t not in english_stopwords]

data['cleaned_tokens'] = data.product_review_tokenize.apply(remove_stopwords)
data.head()


# Let's recreate the reviews from the cleaned tokens again

# In[ ]:


# Key Takeaways
# Tokenization is the process of breaking down a continuous stream of text, such as a
# sentence or a paragraph, into smaller units called takers. These takers typically
# correspond to words, but can also represent subword units like prefixes, suffixes, and
# stems.
# Tokenization facilitates the transformation of text into a format that machine
# learning algorithms can understand.
# Stop words are common words (eg, "the" "and" "s") that appear frequently in a
# language and have little semantic value. Removing them is essential in natural
# language processing tasks to reduce data size speed up processing and improve the
# accuracy of algorithms by focusing on more informative words that convey the
# actual meaning of a text.


# In[21]:


# tokens = data.cleaned_tokens[0]
# " ".join(tokens)

data['product_review_cleaned'] = data.cleaned_tokens.apply(lambda x: " ".join(x))
data.head()


# In[ ]:





# ## Task 4
# 
# ### Exploring the tokens and product categories

# In[23]:


# data.head()
data.tail()


# Let's take a look at the product categories again

# In[24]:


data.product_category.value_counts()


# Let's combine all the tokens used in reviews for the `Tops`

# In[33]:


tops_tokens = []
for x in data[data.product_category == 'Tops'].cleaned_tokens:
    tops_tokens.extend(x)
len(tops_tokens)


# Let's find the 20 most common words in the `Tops` products' reviews 

# In[35]:


freq_dist = FreqDist(tops_tokens)
freq_dist.most_common (20)


# now find the 20 most common words in the `Dresses` products' reviews 
# 
# Utilized the `.most_common()` method to identify the most frequently used
# tokens within a list of tokens.

# In[45]:


all_tokens = []

# change the jackets to dresses for its most used words
for x in data[data.product_category == "Jackets"].cleaned_tokens:
    all_tokens.extend(x)
freq_dist = FreqDist(all_tokens)
freq_dist.most_common(20)


# ## Task 5
# 
# ### Extracting the adjectives used in the reviews

# In[46]:


data.head()


# ### Part of Speech Tagging
# 
# 
# <b>Part of Speech:</b> The grammatical role of a word in a sentence. A part of speech is one of the nine types of English words: VERB, NOUN, ADJECTIVE, ADVERB, PRONOUN, PREPOSITION, DETERMINER, CONJUNCTION, INTERJECTION.

# In[47]:


data.product_review[0]


# In[53]:


nltk.download('tagsets')
nltk.help.upenn_tagset()


# In[57]:


nltk.pos_tag(data.product_review_tokenize[0])


# Let's use the POS-tagger to assign part of speech to all the tokens of all of the reviews

# In[59]:


data['POS_tokens'] = data.product_review_tokenize.apply(nltk.pos_tag)
data.head()


# Let's extract the adjectives used in each review

# In[60]:


def extract_adj(tokens):
    adjectives = []
    for x in tokens:
        if x[1] in ['JJ', 'JJR', 'JJS']:
            adjectives.append(x[0])
    return adjectives

data['adjectives'] = data.POS_tokens.apply(extract_adj)
data.head()


# Let's combine all the `adjectives` for the `Tops`.

# In[62]:


adj_tops = ""
for x in data[data.product_category == 'Tops'].adjectives:
    adj_tops += " ".join(x) + " "

adj_tops
    


# Let's visualize the adjectives using a wordcloud

# In[65]:


word_cloud = WordCloud(width = 800, height = 600, background_color = 'white').generate(adj_tops)
plt.imshow(word_cloud)

plt.axis('off')
plt.show()


# <b>Exercise</b>: Write a python method that gets a product category name and combine the adjectives used in the reviews related to the input category and visualize them using a word cloud.

# In[68]:


def visualize_adjectives(category):
    adjectives = ""
    for x in data[data.product_category == category].adjectives:
        adjectives += " ".join(x) + " "
    word_cloud = WordCloud(width=800, height=600, background_color="white").generate(adjectives)
plt.imshow(word_cloud)
plt.axis("off")
plt.show()
visualize_adjectives("Dresses")


# ## Task 6 
# 
# ### Finding the sentiment of each review 

# In[ ]:


data.head()


# finding the sentiment of the reviews

# In[71]:


sent = SentimentIntensityAnalyzer()
review = data.product_review_cleaned[0]
print(review)

scores = sent.polarity_scores(review)
print(scores)


# ### Sentiment scores:
# 
# - `pos`: The probability of `positive` sentiment
# - `neu`: The probability of `neutral` sentiment
# - `neg`: The probability of `negative` sentiment
# - `compound`: The normalized `compound` score that takes values from -1 to 1
# 
# We can use the `compound` score to find the sentiment of each review.
# 
# - if compound score>=0.05 then `positive` 
# - if compound score between -0.05 and 0.05 then `neutral` 
# - if compound score<=-0.05 then `negative` 
# 
# 
# Now let's create a method to find the sentiment of a review using the compound score

# In[74]:


def polarity_score(review):
    # Initilizing the Sentiment Analyzer
    sent = SentimentIntensityAnalyzer()
   
    # Extracting the sentiment polarity scores of a review
    scores = sent.polarity_scores(review)
    
    # Getting the compound score
    compound = scores['compound']
    
    if compound > 0.05:
        return "positive"
    elif compound < -0.5:
        return "negative"
    else:
        return "neutral"

polarity_score("this product is Amazing, the quality is good")


# Let's label all the reviews with sentiment

# In[76]:


data['sentiment'] = data.product_review_cleaned.apply(polarity_score)
data.head()


# In[79]:


df = data.groupby(["product_category", "sentiment"]).size().reset_index(name="counts")
df


# In[80]:


px.bar(df, x = "product_category", y="counts", color = "sentiment", barmode = "group")


# In[ ]:




