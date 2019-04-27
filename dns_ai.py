import os
import queue
import csv
import threading
import matplotlib.pyplot as plt
from collections import Counter
import numpy as np
from scipy.sparse.csr import csr_matrix
from sklearn.utils.multiclass import unique_labels
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.feature_selection import RFECV
from sklearn.svm import LinearSVC
from sklearn.linear_model import SGDClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.tree import DecisionTreeClassifier


def extract_files_data():
    # A path to the dns user files.
    dir = 'resources/DNS-REQ-RES/'
    # An empty list to be filled with the urls from each user file.
    dns_qry_names = list()
    # An empty list to be filled with the user names: the classification labels.
    labels = list()

    print('\033[1mExtracting features and labels from the dataset:\033[0m')

    # Iterate each user file in the directory,
    # every 100 lines append the dns_qry_names list with the users urls.
    # Eventually the list is populated with
    for file_i in os.listdir(dir):
        with open(dir + file_i) as f:
            lines = csv.reader(f)
            # Skip the first row(the column names row).
            next(lines)
            # Create an empty list to be filled with the user urls,
            part = list()

            for i, row in enumerate(lines):
                part.append(row[10])

                # Append the dns_qry_names list with the part list,
                # and clear the part list,
                # append the labels list with the right user label for the part.
                if i > 0 and i % 100 == 0:
                    dns_qry_names.append(' '.join(part))
                    labels.append(file_i[11:18])

                    # Update progress.
                    print('\tExtracting features from {}: {:.0f}%'.format(file_i, 100 * (len(dns_qry_names) / 31565)),
                          end='\r')

                    part.clear()

    print('\tExtracting features from user files: \033[32mDone.\033[0m')

    # Vectorize the strings to a TfIdf sparse matrix.
    # Filter the strings using regex to get the full url strings.
    print('\tCreating a TfidfVectorizer object...', end='\r')
    vectorizer = TfidfVectorizer(use_idf=True, token_pattern='(\\S+\\.\\w+[^ ])').fit(dns_qry_names)
    print('\tCreating a TfidfVectorizer object: \033[32mDone.\033[0m')

    print('\tCreating the features list...', end='\r')
    features = vectorizer.transform(dns_qry_names)
    print('\tCreating the features list: \033[32mDone.\033[0m')

    print('\tCreating a Counter object using a CountVectorizer object...', end='\r')
    counter = CountVectorizer(token_pattern='(\\S+\\.\\w+[^ ])').fit(dns_qry_names)
    counter = Counter(counter.vocabulary_)
    print('\tCreating a Counter object using a CountVectorizer object: \033[32mDone.\033[0m')

    return_queue.put(vectorizer)
    return_queue.put(features)
    return_queue.put(labels)
    return_queue.put(counter)
    return_queue.join()

    print('\033[1mFinished features extraction.\033[0m')


def cross_validation(cv=3, step=0.25):
    scores = {clf[0]: None for clf in classifiers}

    print("\033[1mPerforming feature elimination cross-validation:\033[0m\n"
          "* Running {} validations.\n"
          "* Eliminating {}% features per iteration.".format(cv, step))

    for clf in classifiers:
        print('\tCurrently running: {}...'.format(clf[0]), end='\r')

        selected = RFECV(estimator=clf[1], step=step, cv=cv).fit(features, labels)
        scores[clf[0]] = selected.score(features, labels)

        print('\t{}: \033[32mDone.\033[0m'.format(clf[0]))

    print('\t\033[1mResults:')

    for k, v in scores.items():
        print('\t\t\033[1m{}: {:.2f}%\033[0m'.format(k, v * 100))


def plot_confusion_matrices():
    # Create the plot setup.
    figure, axs = plt.subplots(2, 2, constrained_layout=True)
    axs = axs.flatten()

    color_maps = (plt.cm.Blues_r, plt.cm.Oranges_r, plt.cm.Greens_r, plt.cm.Reds_r)

    # Split the features and labels to training and test parts.
    X_train, X_test, Y_train, Y_test = train_test_split(features, labels, shuffle=True, test_size=0.1)

    for i, clf in enumerate(classifiers):
        # Train the classifier using the training data
        # then predict the test data.
        prediction = clf[1].fit(X_train, Y_train).predict(X_test)

        # Create the confusion matrix.
        cm = confusion_matrix(prediction, Y_test)
        # Normalize
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

        ax = axs[i]
        im = ax.imshow(cm, interpolation='nearest', cmap=color_maps[i])
        classes = unique_labels(labels, prediction)
        ax.figure.colorbar(im, ax=ax)
        ax.set(title=clf[0], xticks=np.arange(cm.shape[1]), yticks=np.arange(cm.shape[0]), xticklabels=classes,
               yticklabels=classes, ylabel='True label', xlabel='Predicted label')
        plt.setp(ax.get_xticklabels(), rotation=90, ha="right", rotation_mode="anchor")


def menu():
    user_input = None

    while user_input != '3':
        print('\033[1mWelcome.\033[0m')
        print('\t1. Score classifiers.')
        print('\t2. Plot confusion matrices.')
        print('\t3. Quit program.')

        user_input = input('Please choose an option')

        if user_input.isnumeric():
            if user_input == '1':
                cv = input('\tHow many iterations(press enter for default)?')
                step = input('\tChoose percent of features to eliminate each iteration(press enter for default)?')

                if cv == '':
                    cv = 3
                if step == '':
                    step = 0.25

                t = threading.Thread(target=cross_validation, args=[cv, step])
                t.start()
                t.join()
            if user_input == '2':
                pass
            if user_input == '3':
                print('Terminating program.\nGood bye.')
        else:
            print('Please choose a number from the menu.')


'''Critical Variables:
    
    classifiers: A list of classifiers to use.
                    LinearSVC - Linear Support Vector Classification.
                    MultinomialNB - Naive Bayes classifier for multinomial models.
                    SGDClassifier - Stochastic Gradient Descent Classifier.
                    DecisionTreeClassifier - A decision tree classifier :)                    
    return_queue: A queue to catch threaded return values.    
    vectorizer: A TfidfVectorizer object to be fit with the features.
    features: The features used in the classification process.
    labels: The user names are used as class label.
    counter: A counter instance for debugging and data analysis.
'''
# If either classifiers or the return_queue variables
# are not defined create them.
try:
    classifiers
    return_queue
except NameError:
    classifiers = [('LinearSVC', LinearSVC(loss='squared_hinge', tol=0.1, max_iter=500, C=0.1)),
                   ('MultinomialNB', MultinomialNB(alpha=0.1)),
                   ('SGDClassifier',
                    SGDClassifier(loss='modified_huber', penalty='l1', alpha=1e-5, max_iter=2000, tol=0.01)),
                   ('DecisionTreeClassifier', DecisionTreeClassifier())]
    return_queue = queue.Queue()

# If any of the critical variables (vectorizer, features, labels, counter)
# are not defined properly run the extract_files_data function
# in a separate thread to define them.
try:
    vectorizer
    features
    labels
    counter
except NameError:
    # Extract the features from the user files using a separate thread
    # and catch the returned values using the return_queue.
    threading.Thread(target=extract_files_data).start()

    vectorizer = return_queue.get()
    return_queue.task_done()

    # The urls(features) from each user file in to a dictionary.
    features = return_queue.get()
    return_queue.task_done()

    # The user corresponding to the urls(features) extracted.
    labels = return_queue.get()
    return_queue.task_done()

    # A CountVectorizer object fit with the features.
    counter = return_queue.get()
    return_queue.task_done()

# Show the menu.
menu()
