import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud, STOPWORDS
import string
import re
import random
import missingno as msno
import tensorflow as tf
# from transformers import TFAutoModel, AutoTokenizer
import os
# from tqdm.notebook import tqdm
import tensorflow_hub as hub

train1 = pd.read_csv("data/jigsaw-toxic-comment-train.csv", usecols=[1, 2])
print(train1.head())
print(len(train1))
train2 = pd.read_csv("data/jigsaw-unintended-bias-train.csv", usecols=[1, 2])
print(train2.head())
print(len(train2))

# data = pd.concat([train1[['comment_text', 'toxic']],
#                   train2[['comment_text', 'toxic']].query('toxic==1'),
#                   train2[['comment_text', 'toxic']].query('toxic==0').sample(n=100000,random_state=0)
#                   ], ignore_index=True)
data = pd.concat([train1,
                  train2.query('toxic==1'),
                  train2.query('toxic==0').sample(n=100000,random_state=0)
                  ], ignore_index=True)
print(len(data))
print(data.head())
print(data.loc[223546:223550])
print(data.dtypes)
data[['toxic']] = data[['toxic']].astype(int)
print(data.dtypes)
print(data.head())
# train2.toxic = train2.toxic.round().astype(int)
#
# valid = pd.read_csv('data/validation.csv')
# test = pd.read_csv('data/test.csv')
# sub = pd.read_csv('data/sample_submission.csv')
#
# train = pd.concat([
#     train1[['comment_text', 'toxic']],
#     train2[['comment_text', 'toxic']]
# ])

# plt.figure(figsize=(15, 8))
# plt.title("Count of labels 0 vs 1")
# plt.xlabel("Toxic")
# plt.ylabel("Count")
# sns.countplot(x="toxic", data=train)
#
# train = pd.concat([
#     train1[['comment_text', 'toxic']],
#     train2[['comment_text', 'toxic']].query('toxic==1'),
#     train2[['comment_text', 'toxic']].query('toxic==0').sample(n=100000, random_state=0)
# ])
# plt.figure(figsize=(15, 8))
# plt.title("Count of labels 0 vs 1")
# plt.xlabel("Toxic")
# plt.ylabel("Count")
# sns.countplot(x="toxic", data=train)
# plt.show()
#
# plt.figure(figsize = (12, 8))
# msno.bar(train)
#
# stopwords = set(STOPWORDS)
#
#
# def word_cloud(data, title=None):
#     data = data.apply(lambda x: x.lower())
#     cloud = WordCloud(
#         background_color="black",
#         stopwords=stopwords,
#         max_words=200,
#         max_font_size=40,
#         scale=3).generate(str(data))
#
#     fig = plt.figure(figsize=(15, 15))
#     plt.axis("off")
#     if title:
#         fig.suptitle(title, fontsize=20)
#         fig.subplots_adjust(top=2.25)
#
#     plt.imshow(cloud)
#     plt.show()
#
# word_cloud(train["comment_text"], "WordCloud for train data")
# word_cloud(valid["comment_text"].apply(str), "WordCloud for valid data")
# word_cloud(test["content"], "WordCloud for test data")
# plt.figure(figsize = (15, 8))
# len_sent = train["comment_text"].apply(lambda x : len(x.split()))
# sns.distplot(len_sent.values)
# plt.title("Distribution of length of words")
# plt.xlabel("Length of words")
# plt.ylabel("Probability of occurance")
