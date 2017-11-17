from sklearn import datasets
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer

span_train_data = [line.rstrip() for line in open('spam_train.txt')]

labels = [message[0] for message in span_train_data]
messages = [message[2:] for message in span_train_data]

#print(labels)
#print(messages)
#print (len(span_train_data))

import numpy
from sklearn.feature_extraction.text import CountVectorizer

count_vectorizer = CountVectorizer()
counts = count_vectorizer.fit_transform(messages)

from sklearn.naive_bayes import MultinomialNB

classifier = MultinomialNB()
classifier.fit(counts, labels)

span_test_data = [line.rstrip() for line in open('spam_test.txt')]

labels = [message[0] for message in span_train_data]
messages = [message[2:] for message in span_train_data]

example_counts = count_vectorizer.transform(messages)
predictions = classifier.predict(example_counts)

from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix

print('Total emails classified:', len(messages))
print('Score:', accuracy_score(predictions, labels))
print('Confusion Matrix: \n', confusion_matrix(predictions, labels))
