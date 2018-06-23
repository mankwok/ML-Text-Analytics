import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import numpy as np
import pandas as pd

train = pd.read_csv('./input/train.csv').fillna(' ')
i=1
filltered_comment_list = []
for comment in train.comment_text:
    filtered_words = [w for w in word_tokenize(comment) if not w.lower() in stopwords.words('english')]
    sentence = ' '.join(word for word in filtered_words)
    filltered_comment_list.append(sentence)
    i = i + 1
    print(i)

output = pd.DataFrame(columns=['comment_text'])

output = pd.DataFrame(filltered_comment_list, columns=['comment_text'])
output.to_csv('filtered.csv', index=False)


