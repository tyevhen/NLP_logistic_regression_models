import nltk
from nltk.corpus import brown
import string
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction import DictVectorizer
import numpy as np

def data_split():
    sents = brown.tagged_sents()

    num_train = int(len(sents)*0.7)
    num_test = int(len(sents)*0.15)

    train_set = sents[:num_train]
    test_set = sents[num_train:num_train+num_test]
    dev_set = sents[num_train+num_test:]

    return train_set, test_set, dev_set

def build_vocabulary(train_set, th):
    words = [word[0].lower() for sent in train_set for word in sent]
    freq_dist = nltk.FreqDist(words)
    filtered = list(filter(lambda x: x[1] >= th, freq_dist.items()))
    vocabulary = [filtered[i][0] for i in range(len(filtered))]

    return vocabulary


def last_word(pair, sent, vocabulary):
    feature = {}
    label = str()
    if pair[0].lower() in vocabulary:
        idx = sent.index(pair)
        prev_pair = sent[idx - 1]
        feature['word'] = prev_pair[0].lower()
        label = pair[1]

    return feature, label

def second_last_word(pair, sent, vocabulary):
    feature = {}
    label = str()
    if pair[0].lower() in vocabulary:
        idx = sent.index(pair)
        sprev_pair = sent[idx - 2]
        feature['word'] = sprev_pair[0].lower()
        label = pair[1]

    return feature, label

def ngram_prefix(pair, sent, vocabulary, n):
    feature = {}
    label = str()
    if pair[0].lower() in vocabulary:
        feature['pref'] = pair[0][:n].lower()
        label = pair[1]

    return feature, label

def ngram_suffix(pair, sent, vocabulary, n):
    feature = {}
    label = str()
    if pair[0].lower() in vocabulary:
        feature['pref'] = pair[0][-n:].lower()
        label = pair[1]

    return feature, label

def curr_stem(pair, sent, vocabulary):
    label = str()
    feature = {}
    if pair[0].lower() in vocabulary:
        feature['stem'] = stemmer.stem(pair[0].lower())
        label = pair[1]

    return feature, label

def prev_stem(pair, sent, vocabulary):
    feature = {}
    label = str()
    if pair[0].lower() in vocabulary:
        idx = sent.index(pair)
        prev_pair = sent[idx - 1]
        feature['stem'] = stemmer.stem(prev_pair[0].lower())
        label = pair[1]

    return feature, label

def prev_to_last_stem(pair, sent, vocabulary):
    feature = {}
    label = str()
    if pair[0].lower() in vocabulary:
        idx = sent.index(pair)
        sprev_pair = sent[idx - 2]
        feature['stem'] = stemmer.stem(sprev_pair[0].lower())
        label = pair[1]

    return feature, label

def starts_capital(pair, sent, vocabulary):
    feature = {}
    label = str()
    if pair[0].lower() in vocabulary:
        if pair[0][0].isupper():
            feature['upper'] = 1
        else:
            feature['upper'] = 0
        label = pair[1]

    return feature, label

def has_punctuation(pair, sent, vocabulary):
    feature = {}
    label = str()
    punct = string.punctuation
    if pair[0].lower() in vocabulary:
        if any(char in punct for char in pair[0].lower()):
            feature['punct'] = 1
        else:
            feature['punct'] = 0
        label = pair[1]

    return feature, label

def build_features(data_set, vocabulary, feature_pattern):
    feature_dicts = []
    labels = []
    for sent in data_set:
        if len(sent) > 1:
            for pair in sent:
                feature, label = feature_pattern(pair, sent, vocabulary)
                feature_dicts.append(feature)
                labels.append(label)
                assert len(feature_dicts) == len(labels)

    return feature_dicts, labels

def build_ngram_features(data_set, vocabulary, feature_pattern, n):
    feature_dicts = []
    labels = []
    for sent in data_set:
        if len(sent) > 1:
            for pair in sent:
                if len(pair[0]) >= n:
                    feature, label = feature_pattern(pair, sent, vocabulary, n)
                    feature_dicts.append(feature)
                    labels.append(label)
                assert len(feature_dicts) == len(labels)

    return feature_dicts, labels


def build_feature_matrix(feature_dicts):
    vec = DictVectorizer()
    feature_matrix = vec.fit_transform(feature_dicts)

    return feature_matrix, vec



if __name__ == "__main__":
    templates = [last_word, second_last_word, curr_stem, prev_stem,
                 prev_to_last_stem, has_punctuation, starts_capital]
    ngram_templates = [ngram_prefix, ngram_suffix]

    regularization_params = [0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1, 2, 5, 10, 20, 50]
    n = [1, 2, 3]

    stemmer = nltk.PorterStemmer()

    train_set, test_set, dev_set = data_split()

    vocabulary = build_vocabulary(train_set, 2)

    for template in templates:
        for reg in regularization_params:
            train_dict, train_labels = build_features(train_set, vocabulary, template)
            dev_dict, dev_labels = build_features(dev_set, vocabulary, template)

            feature_matrix, vec = build_feature_matrix(train_dict)
            model = LogisticRegression(solver='lbfgs', multi_class='multinomial', C=reg)
            model.fit(feature_matrix, train_labels)

            train_predict = model.predict(feature_matrix)
            train_acc = np.mean(train_predict == train_labels)
            print("#####################################################")
            print("Regularization strength=",reg)
            print("Train set accuracy for feature template %s is %.4f" % (str(template), train_acc))

            dev_X = vec.transform(dev_dict)
            dev_predict = model.predict(dev_X)
            dev_acc = np.mean(dev_predict == dev_labels)

            print("Dev set accuracy for feature template %s is %.4f" % (str(template), dev_acc))
            print("#####################################################")
            print()

    for ngram_template in ngram_templates:
        for i in n:
            for reg in regularization_params:
                train_dict, train_labels = build_ngram_features(train_set, vocabulary, template, i)
                dev_dict, dev_labels = build_ngram_features(dev_set, vocabulary, template, i)

                feature_matrix, vec = build_feature_matrix(train_dict)
                model = LogisticRegression(solver='lbfgs', multi_class='multinomial', C=reg)
                model.fit(feature_matrix, train_labels)

                train_predict = model.predict(feature_matrix)
                train_acc = np.mean(train_predict == train_labels)
                print("#####################################################")
                print("Regularization strength C=", reg)
                print("Train set accuracy for feature template %s is %.4f" % (str(template), train_acc))

                dev_X = vec.transform(dev_dict)
                dev_predict = model.predict(dev_X)
                dev_acc = np.mean(dev_predict == dev_labels)

                print("Dev set accuracy for feature template %s is %.4f" % (str(template), dev_acc))
                print("#####################################################")
                print()



