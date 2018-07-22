import datetime

import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_union

if __name__ == "__main__" :
	target_classes = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']

	train_data = pd.read_csv('./input/train.csv').fillna(' ')
	test_data = pd.read_csv('./input/test.csv').fillna(' ')

	train_text = train_data['comment_text']
	test_text = test_data['comment_text']

	one_word_vectorizer = TfidfVectorizer(
		sublinear_tf=True,
		strip_accents='unicode',
		analyzer='word',
		token_pattern=r'\w{1,}',
		ngram_range=(1, 1),
		max_features=30000)
	two_word_vectorizer = TfidfVectorizer(
		sublinear_tf=True,
		strip_accents='unicode',
		analyzer='word',
		token_pattern=r'\w{1,}',
		ngram_range=(1, 2),
		max_features=30000)
	char_vectorizer = TfidfVectorizer(
		sublinear_tf=True,
		strip_accents='unicode',
		analyzer='char',
		ngram_range=(1, 4),
		max_features=30000)

	print('Start: {}'.format(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')))
	combined_vectorizer = make_union(one_word_vectorizer, two_word_vectorizer, char_vectorizer, n_jobs=3)

	combined_vectorizer.fit(train_text)
	train_features = combined_vectorizer.transform(train_text)
	test_features = combined_vectorizer.transform(test_text)

	submission = pd.DataFrame.from_dict({'id': test_data['id']})
	for target_class in target_classes:
		train_target = train_data[target_class]
		logit_classifier = LogisticRegression(solver='sag')
		print('Predicting {}: {}'.format(target_class, datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')))
		logit_classifier.fit(train_features, train_target)
		submission[target_class] = logit_classifier.predict_proba(test_features)[:, 1]

	submission.to_csv('./output/submission_{}.csv'.format(datetime.datetime.now().strftime('%Y%m%d%H%M')), index=False)
	print('End: {}'.format(str(datetime.datetime.now())))