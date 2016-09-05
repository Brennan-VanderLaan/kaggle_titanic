import tensorflow as tf
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pylab

mpl.style.use('ggplot')

def readFile(name):
    """
    Read in a CSV File
    :param name: filename
    :return: pandas DataFrame
    """
    with open(name, 'r') as fs:
        csv_fs = pd.read_csv(fs)
        return csv_fs

def findTitles(data):
    """
    Find the titles that are present in the names
    add them to the given DataFrame
    :param data: DataFrame with Titanic info
    :return: list of titles
    """
    titles = []
    for row in data['Name']:
        title = row.split(",")[1].strip().split(".")[0] + '.'
        # if title not in titles:
        #     print "Found new title: %s" % title
        #     print "Row: %s" % row
        titles.append(title)

    data['Title'] = pd.Series(titles, index=data.index)
    titles = list(set(titles))
    titles.sort()
    data['TitleIndex'] = data['Title'].map(titles.index).astype(int)

    return titles

def findLastNames(data):
    """
    Find the last names in the dataset
    :param data: DataFrame with Titanic info
    :return: Set of names
    """
    names = []
    for row in data['Name']:
        name = row.split(",")[0]
        names.append(name)

    data['LastName'] = pd.Series(names, index=data.index)
    return set(names)

def fillAges(data):
    """
    Fills in ages in the dataset based on the median ages in
    each class of the population. Adds filled ages to 'AgeFill' column
    :param data: DataFrame with Titanic dataset
    :return: None
    """

    #male/female for all classes
    med_ages = np.zeros((2,3))

    data['Gender'] = data['Sex'].map({'female': 0, 'male': 1}).astype(int)
    data['PclassStr'] = data['Pclass'].map({1: '1', 2: '2', 3: '3'}).astype(str)

    for i in [0,1]:
        print "Checking %s" % ['female', 'male'][i]
        for j in [0,1,2]:
            med_ages[i,j] = data[(data['Gender'] == i) & (data['Pclass'] == j+1)]['Age'].dropna().median()
            print "Class: %d, median age: %2f" % ((j+1), float(med_ages[i,j]))

    print "Pulled median ages for classes"

    data['AgeFill'] = data['Age']

    for i in [0,1]:
        for j in [0,1,2]:
            data.loc[(data.Age.isnull()) & (data.Gender == i) & (data.Pclass == j+1), 'AgeFill'] = med_ages[i,j]

def processGroup(data):
    """
    Do common preprocessing for a group in place
    :param data: DataFrame with Titanic data
    :return: dict of titles and last names for the dataset
    """
    fillAges(data)
    titles = findTitles(data)
    names = findLastNames(data)

    return {'titles': titles, 'names': names}

def weight_var(shape):
    initial = tf.truncated_normal(shape, stddev=.1)
    return tf.Variable(initial)

def bias_var(shape):
    initial = tf.constant(.1, shape=shape)
    return tf.Variable(initial)

def convRangeToValues(numPoints, value):
    base = []
    for i in range(numPoints):
        base.append(0)
    base[value] = 1
    return base

def constructFeatures(data, catCol, conCol):
    """
    Takes a DataFrame and builds out a list of features for use as input to TF
    :param data: processed DataFrame (needs to have Gender and AgeFill)
    :return: list of features in the form of [catCol..., conCol...]
    """

    test_data = []
    for i in data.iterrows():
        i = i[1]

        px = []
        for col in catCol:
            px.append(i[col])

        for col in conCol:
            px.append(i[col])

        print "Input: " + str(px)
        test_data.append(px)

    return test_data

def getSurvived(data):
    """
    Grabs Survived and turns it into a [x,y] form for classification
    :param data:
    :return: list of survival
    """

    survived = []

    for i in data.iterrows():
        i = i[1]

        py = convRangeToValues(2, i['Survived'])
        survived.append(py)

    return survived


def buildLinearColumns(data, catCols, conCols):
    """
    Builds out a dict for tf input
    :param data: processed DataFrame
    :param catCols: columns to categorize
    :param conCols: columns that are continuous
    :return: dict of features
    """

    df = pd.DataFrame()

    for col in catCols:
        df[col] = data[col]

    for col in conCols:
        df[col] = data[col]

    conDict = {k: tf.constant(df[k].values) for k in conCols}

    catDict = {k: tf.SparseTensor(
        indices=[[i, 0] for i in range(df[k].size)],
        values=df[k].values,
        shape=[df[k].size, 1])
        for k in catCols
    }

    features = dict(conDict.items() + catDict.items())

    return features

def getLabels(data):
    return tf.constant(data["Survived"].values)





train_data = readFile("train.csv")
test_data = readFile("test.csv")
combined = test_data.append(train_data.drop('Survived', 1))

print "Loaded training and test files..."

catCols = ["Title", "PclassStr", "Sex"]
conCols = ["AgeFill", "SibSp", "Parch", "Fare"]

trainingProcessed = processGroup(train_data)
processGroup(test_data)

def trainingInput():
    features = buildLinearColumns(train_data, catCols, conCols)
    labels = getLabels(train_data)
    return features, labels

def testInput():
    features = buildLinearColumns(train_data, catCols, conCols)
    labels = getLabels(train_data)
    return features, labels

def submitInput():
    features = buildLinearColumns(test_data, catCols, conCols)
    return features

def buildLinearModel(data, catCols, conCols, titles):
    """
    Build a linear model based on the input
    :param features: feature input
    :param labels: labels for training
    :param titles: titles pulled from the data
    :return: trained linear classifier
    """

    titles = tf.contrib.layers.sparse_column_with_keys(column_name="Title", keys=titles)
    gender = tf.contrib.layers.sparse_column_with_keys(column_name="Sex", keys=['female', 'male'])
    pclass = tf.contrib.layers.sparse_column_with_keys(column_name="PclassStr", keys=['1', '2', '3'])

    age = tf.contrib.layers.real_valued_column(column_name="AgeFill")
    age_buckets = tf.contrib.layers.bucketized_column(age, boundaries=[2,5,10,15,18,20,22,24,26,28,30,33,36,40,45,50,60,70])
    siblings = tf.contrib.layers.real_valued_column(column_name="SibSp")
    parents = tf.contrib.layers.real_valued_column(column_name="Parch")
    fare = tf.contrib.layers.real_valued_column(column_name="Fare")

    titles_x_pclass = tf.contrib.layers.crossed_column([titles,pclass], hash_bucket_size=10000)
    gender_x_pclass = tf.contrib.layers.crossed_column([gender,pclass], hash_bucket_size=10)
    titles_x_gender_x_pclass = tf.contrib.layers.crossed_column([titles, gender, pclass], hash_bucket_size=1000000)
    age_x_gender = tf.contrib.layers.crossed_column([age_buckets, gender], hash_bucket_size=10000)

    deep_columns = [
        tf.contrib.layers.embedding_column(titles, dimension=48),
        tf.contrib.layers.embedding_column(gender, dimension=12),
        tf.contrib.layers.embedding_column(pclass, dimension=9),
        tf.contrib.layers.embedding_column(titles_x_pclass, dimension=24),
        tf.contrib.layers.embedding_column(gender_x_pclass, dimension=12),
        tf.contrib.layers.embedding_column(titles_x_gender_x_pclass, dimension=48),
        tf.contrib.layers.embedding_column(age_x_gender, dimension=48),
        age_buckets,
        fare,
        siblings,
        parents
    ]

    # model = tf.contrib.learn.DNNLinearCombinedClassifier(
    #     model_dir='./model',
    #     linear_feature_columns=[
    #         titles,
    #         gender,
    #         pclass,
    #         age_buckets,
    #         fare
    #     ],
    #     dnn_feature_columns=deep_columns,
    #     dnn_hidden_units=[400, 200, 100, 10],
    # )

    model = tf.contrib.learn.DNNClassifier(
        hidden_units=[400,200,100,100,10],
        feature_columns=deep_columns,
        model_dir="./model"
    )


    for i in range(280):
        model.fit(input_fn=trainingInput, steps=100)

        if i % 5 == 0:
            results = model.evaluate(input_fn=testInput, steps=1)
            print "Accuracy: " + str(results['accuracy'])


    # model = tf.contrib.learn.LinearClassifier(feature_columns=[
    #     titles, gender, pclass,
    #     age, age_buckets, siblings, parents, fare,
    #     titles_x_pclass,
    #     gender_x_pclass,
    #     titles_x_gender_x_pclass,
    #     age_x_gender
    # ], model_dir="./model")
    #
    # for i in range(10):
    #     model.fit(input_fn=trainingInput, steps=80)

    return model


model = buildLinearModel(train_data, catCols, conCols, trainingProcessed['titles'])

results = model.evaluate(input_fn=testInput, steps=1)

for key in sorted(results):
    print "%s: %s" % (key, results[key])

output = model.predict(input_fn=submitInput)

for i in range(len(output)):
    print "%d,%d" % (i+892, output[i])

