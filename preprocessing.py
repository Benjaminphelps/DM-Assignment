from sklearn.feature_extraction.text import CountVectorizer
import numpy as np
import os

negative_real_path = 'op_spam_v1.4/negative_polarity/deceptive_from_MTurk'

# negative_false_path = 'op_spam_v1.4/negative_polarity/deceptive_from_MTurk'

print ("TEST:", os.listdir(negative_real_path))

def load_data(training_arr, testing_arr, path):
    # Load training set
    for folder in os.listdir(path):
        extended_path = path+"/"+folder
        if folder != 'fold5':
            for filename in os.listdir(extended_path):
                if filename.endswith('.txt'):
                    with open(os.path.join(extended_path, filename), encoding='utf-8') as f:
                        training_arr.append(f.read())
        else:
            for filename in os.listdir(extended_path):
                if filename.endswith('.txt'):
                    with open(os.path.join(extended_path, filename), encoding='utf-8') as f:
                        testing_arr.append(f.read())

neg_real_train = []
neg_real_test = []
neg_fake_train = []
neg_fake_test = []

neg_real_path = 'op_spam_v1.4/negative_polarity/truthful_from_Web'
neg_fake_path = 'op_spam_v1.4/negative_polarity/deceptive_from_MTurk'

load_data(neg_real_train, neg_real_test, neg_real_path)
load_data(neg_fake_train, neg_fake_test, neg_fake_path)

corpus = np.concatenate(neg_real_train, neg_real_test, neg_fake_train, neg_fake_test)

vectorizer = CountVectorizer()

# Fit and transform the corpus
X = vectorizer.fit_transform(corpus)

# Display the vocabulary
print("Vocabulary:", vectorizer.vocabulary_)

# Display the Bag of Words frequency matrix
print("Bag of Words (as array):\n", X.toarray())

