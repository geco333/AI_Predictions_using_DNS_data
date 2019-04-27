import os
import csv
import numpy as np
from sklearn import metrics
import matplotlib.pyplot as plt
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
from sklearn.feature_selection import RFECV
from sklearn.tree import DecisionTreeClassifier
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.model_selection import *
from sklearn.linear_model import SGDClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB


def extract_labels():
    labels = []

    # Get the training data.
    with open('resources/partial_labels.csv') as file:
        lines = csv.reader(file)

        # Skip csv header.
        next(lines)

        for row in lines:
            labels = labels + row[1:]

        return labels


def extract_features():
    user_files_dir = 'C:/Users/Geco/PycharmProjects/K_Means/resources/FraudedRawData-20190326T124424Z-001/'
    segments = []

    # Do for each training user file.
    for f in os.listdir(user_files_dir):
        with open(user_files_dir + f) as user_file:
            # List the commands in the user file.
            commands = [line.strip() for line in user_file.readlines()]

        # Split the commands to 150 segments.
        #  Each segment is a single string of 100 commands separated by a space(' ').
        for i in range(0, len(commands), 100):
            segments.append(' '.join(commands[i:i + 100]))

    return [(labels[i], segments[i].split()) for i in range(len(segments))]


def extract_ngrams(segments, n):
    vocabulary = {}

    for seg_i, seg in enumerate(segments):
        label = seg[0]
        segment = seg[1]

        for str_i, command in enumerate(segment):
            for m in n:
                gram = ' '.join(segment[str_i:str_i + m])

                if gram in vocabulary.keys():
                    if label == '0':
                        vocabulary[gram][0] += 1
                    else:
                        vocabulary[gram][1] += 1
                else:
                    if label == '0':
                        vocabulary[gram] = [1, 0]
                    else:
                        vocabulary[gram] = [0, 1]

    return vocabulary


def plot_ngrams(segments, n):
    sum_good = 280000
    sum_bad = 20000

    # Create a {ngram: count} dict form the training segments.
    ngrams = extract_ngrams(segments[:1500], n)

    # Create a {ngram: frequency} dict form the training segments.
    ngrams_freq = {k: [v[0] / sum_good, v[1] / sum_bad] for k, v in ngrams.items()}

    more_good = [i for i in ngrams_freq.values() if i[0] > i[1]]
    more_bad = [i for i in ngrams_freq.values() if i[0] < i[1]]
    only_good = [i for i in ngrams_freq.values() if i[1] == 0]
    only_bad = [i for i in ngrams_freq.values() if i[0] == 0]

    plt.xlabel('Good')
    plt.ylabel('Bad')
    plt.scatter([i[0] for i in more_good], [i[1] for i in more_good], s=3, label='More good')
    plt.scatter([i[0] for i in more_bad], [i[1] for i in more_bad], s=3, label='More bad')
    plt.scatter([i[0] for i in only_good], [i[1] for i in only_good], s=5, label='Only good')
    plt.scatter([i[0] for i in only_bad], [i[1] for i in only_bad], s=5, label='Only bad')
    plt.legend()


def find_best_classifier(pipe, classifiers):
    scores = []

    # Run each classifier in the pipe and get its score value.
    for clf in classifiers:
        rfecv = RFECV(clf, step=0.10, cv=StratifiedKFold(n_splits=10), verbose=True).fit(pipe, train_labels)
        score = rfecv.score(pipe, train_labels)
        scores.append(score)

    print(scores)


def split_train_segments():
    test_segments, test_labels = [], []
    split_segments = StratifiedShuffleSplit(n_splits=1).split(train_segments, train_labels)

    for i, j in split_segments:
        test_index = j

    for i in test_index:
        test_segments.append(all_segments[i])
        test_labels.append(labels[i])

    return test_segments, test_labels


# Extract the label for each segment to a single list.
labels = extract_labels()

# Extract the segments to a list,
#  each list item is a tuple containing the segment label and the commands.
segments = extract_features()

# Extract ngrams from the segments list and plot to a scatter graph.
'''plot_ngrams(segments, (2, 3))'''

# Create a list of training segments for use in the classification pipe.
#  Extract the the first 1500 segments from the segments list (the first 10 users)
#  and convert it to a single string for ease of use.
all_segments = [' '.join(i[1]) for i in segments[:]]
train_labels = labels[:1500]
train_segments = all_segments[:1500]

# A list of classifiers to use.
classifiers = [SGDClassifier(tol=1e-2, max_iter=1000, alpha=1e-4),
               MultinomialNB(alpha=1e-10),
               LogisticRegression(solver='saga', tol=1e-1, max_iter=500, C=50),
               LinearSVC(C=0.5, max_iter=1000, tol=1e-1),
               DecisionTreeClassifier(criterion='entropy', splitter='best')]

# Running the data through a pipeline: first vectorize the segments
#  getting a count per ngram then transforming the count matrix
#  to a tfidf matrix.
pipe = Pipeline([('counter', CountVectorizer(ngram_range=(2, 3))),
                 ('tfidf', TfidfTransformer(use_idf=True))]).fit(train_segments)
train = pipe.transform(train_segments)
clf = SGDClassifier(tol=1e-2, max_iter=1000, alpha=1e-4).fit(train, train_labels)

predictions = []

for i in range(1550, 6000, 150):
    predict = pipe.transform(all_segments[i:i + 100])
    predictions.append(clf.predict(predict))

with open('./resources/predictions.csv', 'w') as pred_csv:
    csv.writer(pred_csv).writerows(predictions)
