###Titanic survival analysis
import math
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

np.random.seed(15329)

data = pd.read_csv('./data/train.csv')

del data['PassengerId']
del data['Cabin']
del data['Ticket']

# print(data.head())
# print(data['Pclass'].describe())
# print(data['Sex'].describe())
# print(data['Age'].describe())
# print('- Data types '+  str(data.dtypes))
data['s'] = data['Sex'].map(lambda e: 0. if e == 'female' else 1.).astype(np.float32)
# print('- Data types '+  str(data.dtypes))
# print(data.head())

# print(data[data['s'] == 0]['Sex'].describe())
# print(data[data['s'] == 1]['Sex'].describe())

for i in range(len(data.columns)):
    if (data.dtypes[i] == 'object') | (data.columns[i] == 'Pclass'):
        name = data.columns[i]
        data.iloc[:,i] = data.iloc[:,i].astype('category')

print('- Data types ' +  str(data.dtypes))
'''
Survived       int64
Pclass      category
Name        category
Sex         category
Age          float64
SibSp          int64
Parch          int64
Fare         float64
Embarked    category
s            float32
'''
# print(np.mean(data['Survived'])) # 0.3838
survived_procent = np.mean(data['Survived'])
undersampling_idx = pd.Series(np.random.rand(len(data))) > survived_procent
data = data[(undersampling_idx) | (data['Survived'])].reset_index()

data['Status'] = data['Survived'].map(lambda e: 'alive' if e == 1 else 'Dead').astype(np.str)

data['AgeUnknown'] = data['Age'].map(lambda e: 1. if math.isnan(e) else 0).astype(np.float32)
data['AgeEstimated'] = data['Age'].map(lambda e: 1. if ~math.isnan(e) & (e > 1) & ~e.is_integer() else 0).astype(np.float32)
data['Age'] = data['Age'].map(lambda e: 0. if math.isnan(e) else e ).astype(np.float32)
data['Age'] = data['Age'] / np.max(data['Age'])

train_idx = pd.Series(np.random.rand(len(data))) > 0.7
processed = pd.get_dummies(data.loc[:, ['Age', 'Pclass', 'Sex']])

y_processed = pd.get_dummies(data.loc[:,['Status']])

x_train = processed[train_idx].as_matrix().astype('float32')
y_train = y_processed[train_idx].as_matrix().astype('float32')
x_test = processed[~train_idx].as_matrix().astype('float32')
y_test = data[~train_idx].as_matrix(['Survived']).astype('float32')

x_shape = len(processed.columns)


def model_fn(features, labels, mode):
    hidden_nodes = 10
    W_h = tf.Variable(np.random.randn(x_shape, hidden_nodes), dtype=tf.float32) 
    b_h = tf.Variable(np.zeros(hidden_nodes), dtype=tf.float32)
    W_o = tf.Variable(np.random.randn(hidden_nodes, 2), dtype=tf.float32)
    b_o = tf.Variable(np.zeros(2), dtype=tf.float32)

    y_h = tf.sigmoid(tf.matmul(features['x'],W_h) + b_h)

    y_raw = tf.matmul(y_h,W_o) + b_o

    y_ = tf.nn.softmax(y_raw)

    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(
            mode = mode,
            predictions = y_)

    loss = tf.reduce_sum(tf.nn.softmax_cross_entropy_with_logits(logits=y_raw, labels=labels))

    global_step = tf.train.get_global_step()
    optimizer = tf.train.GradientDescentOptimizer(0.01)
    train = tf.group(optimizer.minimize(loss), tf.assign_add(global_step, 1))
    return tf.estimator.EstimatorSpec(
        mode = mode,
        predictions = y_,
        loss = loss,
        train_op = train)

input_train_fn = tf.estimator.inputs.numpy_input_fn({'x': x_train}, y_train,
    batch_size=4, num_epochs=None, shuffle=True)
input_test_fn = tf.estimator.inputs.numpy_input_fn({'x': x_test},
    batch_size=1, num_epochs=1, shuffle=False)

estimator = tf.estimator.Estimator(model_fn = model_fn, model_dir='./models')
estimator.train(input_fn = input_train_fn, steps=10000)

test_labels = np.argmax(list(estimator.predict(input_fn = input_test_fn)), 1)
y_test = np.transpose(y_test)[0]

print(test_labels[0:15])
print(y_test[0:15].astype('int32'))
print(np.mean(np.abs(test_labels - y_test)))

# train_metrics = estimator.evaluate(input_fn = input_train_fn)
# print('Eval metrics: %r'%train_metrics)
