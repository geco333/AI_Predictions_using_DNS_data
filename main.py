import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from user import User
from consts import *
from sklearn.metrics import roc_auc_score, confusion_matrix
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import *
from sklearn.manifold import TSNE
from sklearn.feature_extraction import DictVectorizer
from sklearn.model_selection import train_test_split, cross_val_score, cross_val_predict, StratifiedShuffleSplit


global users, global_unique_domains, train_test_samples, train_test_labels


def get_global_unique_domains() -> 'A dataframe of the top 100 domains':
    """Crate the dictionary: If the domain name is not
     in the global_unique_domains dictionary
     add the domain name and set its Appearance to 1
     and set its Count value.
     if the domain name is in the dictionary add one to its Appearance
     and add its Count value.

     Create the Dataframe: From the dictionary, filter it to show
     only domain that all user requested, then, sort it descending by count.
    """

    global_unique_domains = dict()

    # Fill the global_unique_domains dictionary with
    # domain name as key and appearances and count value as values.
    # Only add sites that responded to the dns request.
    for user in users:
        for domain in user.unique_domains.index:
            if domain in global_unique_domains:
                global_unique_domains[domain][0] += 1
                global_unique_domains[domain][1] += user.unique_domains.loc[domain]['Count']
            else:
                global_unique_domains[domain] = [1, user.unique_domains.loc[domain]['Count']]

    df = pd.DataFrame(global_unique_domains.values(),
                      index=global_unique_domains.keys(),
                      columns=['Appearance', 'Count'])

    # Only add sites that all 15 users visited.
    df = df[df['Appearance'] == 15][df['Count'] > COUNT_LIMIT].sort_values(by=['Count'], ascending=False)

    return df['Count']


def setup():
    """Define global variables to hold the users list,
        a list of unique domain names across all users
        and the users features for the classifier.

    Create the users list using the data in each users csv file.
    """

    global users  # A list of user objects.
    global global_unique_domains  # A list of top unique domain names visited by all users.
    global train_test_samples
    global train_test_labels

    users = list()

    # Create a list of users with data from the csv files.
    for i in range(USERS_COUNT):
        users.append(User(i))

    global_unique_domains = get_global_unique_domains()
    train_test_samples, train_test_labels = get_features(users)


def get_features(users: 'List of User objects') -> 'Top domains for each user':
    """Create for each user a list of percentages, each index in the list represents
    a unique domain from the global_unique_domains global variable.

    The result is a matrix of length 15 (number of users),
    each row in the matrix is a list of equal length - the number of unique domain names.
    """

    features = list()
    labels = list()
    dict_vectorizer = DictVectorizer(sparse=True).fit([global_unique_domains])

    for user in users:
        # Shuffle the users dataframe.
        df = user.df['dns.qry.name'].sample(frac=1, replace=True)
        # Get
        samples = [df.sample(n=TRAIN_SAMPLES_COUNT) for i in range(TRAIN_SAMPLES_COUNT)]
        unique_df = [User.get_unique_domains(sample)['Percentage'] for sample in samples]

        for df in unique_df:
            sample_features = df.filter(items=global_unique_domains.keys())
            features.append(sample_features)
            labels.append(user.i)

    features = dict_vectorizer.transform(features)

    return features, labels


def plot_data_graphs():
    """Plot varies data graphs."""

    fig, axs = plt.subplots(1, 2)

    # Top 10 domains count for each user.
    for i, user in enumerate(users):
        for domain in global_unique_domains.keys()[:10]:
            axs[1].bar(i, user.unique_domains['Count'][domain])

    axs[1].set_title('Top domains count per user')
    axs[1].grid(axis='y', linewidth=.2, linestyle='dashed')
    axs[1].set_xticks(np.arange(15))
    axs[1].set_xticklabels(User.user_file_numbers, rotation=90)

    axs[0].set_title('Top domains')
    axs[0].bar(global_unique_domains.keys()[:10], global_unique_domains.values[:10])
    for tick in axs[0].get_xticklabels():
        tick.set_rotation(90)

    fig.legend(global_unique_domains.keys()[:10])
    plt.tight_layout()


def run_classification():
    # A dictionary of classifiers to use.
    classifiers = {'GaussianNB': GaussianNB(),
                   'LinearSVC': LinearSVC(),
                   'KNeighborsClassifier': KNeighborsClassifier()}
    X_train, X_test, y_train, y_test = train_test_split(train_test_samples, train_test_labels, train_size=0.6,
                                                        test_size=0.4, shuffle=True)

    # The cross validation will be stratified shuffleSplit,
    # 100 splits each 60% to train and 40% for test.
    cv = StratifiedShuffleSplit(n_splits=10, train_size=0.6, test_size=0.4)

    cv_scores = list()

    # Fit each classifier, get its cross validation score
    # and use it to predict the test data.
    for k, v in classifiers.items():
        if k == 'GaussianNB':
            v.fit(X_train.toarray(), y_train)
            cv_scores.append(cross_val_score(v, train_test_samples.toarray(), train_test_labels, cv=cv).mean())
            predictions = v.predict(X_test.toarray())
        else:
            v.fit(X_train, y_train)
            cv_scores.append(cross_val_score(v, train_test_samples, train_test_labels, cv=cv).mean())
            predictions = v.predict(X_test)
            prob = v.predict_proba(X_test) if k != 'LinearSVC' else v.decision_function(X_test)

    for i, clf in enumerate(classifiers.keys()):
        print(f'{clf}: {cv_scores[i]}')

    # Different color for each user.
    colors = {'292': 'xkcd:purple', '301': 'xkcd:green', '303': 'xkcd:blue', '305': 'xkcd:pink',
              '306': 'xkcd:brown', '308': 'xkcd:red', '316': 'xkcd:light blue', '334': 'xkcd:teal',
              '341': 'xkcd:orange', '343': 'xkcd:light green', '348': 'xkcd:magenta', '354': 'xkcd:yellow',
              '372': 'xkcd:sky blue', '387': 'xkcd:grey', '392': 'xkcd:lime green'}

    # Create a 2D array of samples from the train data
    # for easier visualization as a 2D X vs Y graph.
    tsne_train = TSNE(n_components=2).fit_transform(train_test_samples.toarray())

    clf2d = KNeighborsClassifier().fit(tsne_train, train_test_labels)

    x_min, x_max, y_min, y_max = min(tsne_train[:, 0]), max(tsne_train[:, 0]), min(tsne_train[:, 1]), max(
        tsne_train[:, 1])
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1), np.arange(y_min, y_max, 0.1))
    z = clf2d.predict(np.c_[xx.ravel(), yy.ravel()])
    z = z.reshape(xx.shape)

    plt.contour(xx, yy, z)

    for i in range(7500):
        plt.scatter(tsne_train[i, 0], tsne_train[i, 1], c=colors[str(predictions[i])], s=3, marker='s')


setup()
run_classification()
