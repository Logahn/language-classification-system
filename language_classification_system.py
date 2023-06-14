import matplotlib.pyplot as plt
import pickle
from sklearn.metrics import log_loss
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn import metrics
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
import math
import collections
from itertools import permutations
import seaborn as sns

# Read the dataset
df = pd.read_csv('language.csv')
df = df.dropna()
df['text'] = df['text'].astype(str)
df['language'] = df['language'].astype(str)

"""### Creating set of features"""

# Define punctuation marks and vowel combinations
pun = ("?", "-", ".", ",", "\'", ';', "/", '!')
vow = ['a', 'e', 'i', 'o', 'u']
dou_vow = ['aa', 'ee', 'ii', 'oo', 'uu']
con_vow = [''.join(p) for p in permutations(vow, 2)]
dutch_com = ['ij']

# Create features based on text data
df['word_count'] = df['text'].apply(lambda x: len(x.split()))
df['character_count'] = df['text'].apply(lambda x: len(x.replace(" ", "")))
df['word_density'] = df['word_count'] / (df['character_count'] + 1)
df['punc_count'] = df['text'].apply(lambda x: len([a for a in x if a in pun]))
df['v_char_count'] = df['text'].apply(
    lambda x: len([a for a in x if a.casefold() == 'v']))
df['w_char_count'] = df['text'].apply(
    lambda x: len([a for a in x if a.casefold() == 'w']))
df['ij_char_count'] = df['text'].apply(lambda x: sum(
    [any(a_b in a for a_b in dutch_com) for a in x.split()]))
df['dou_vow_count'] = df['text'].apply(lambda x: sum(
    [any(a_b in a for a_b in dou_vow) for a in x.split()]))
df['con_vow_count'] = df['text'].apply(lambda x: sum(
    [any(a_b in a for a_b in con_vow) for a in x.split()]))
df['vow_char_count'] = df['text'].apply(
    lambda x: len([a for a in x if a in vow]))
df['vow_density'] = df['vow_char_count'] / df['word_count']
df['capitals'] = df['text'].apply(lambda x: sum(1 for c in x if c.isupper()))
df['cap_vs_character'] = df['capitals'] / df['character_count']
df['exclamation_count'] = df['text'].apply(lambda x: x.count('!'))
df['Question_count'] = df['text'].apply(lambda x: x.count('?'))
df['unique_words'] = df['text'].apply(lambda x: len(set(w for w in x.split())))
df['repeat words'] = df['text'].apply(lambda x: len(
    [w for w in collections.Counter(x.split()).values() if w > 1]))
df['words_vs_unique'] = df['unique_words'] / df['word_count']
df['encode'] = np.nan

for i in range(len(df)):
    try:
        df['text'].iloc[i].encode(encoding='utf-8').decode('ascii')
    except UnicodeDecodeError:
        df['encode'].iloc[i] = 0
    else:
        df['encode'].iloc[i] = 1

df

# Calculate the mean of features by language
df.groupby('language').mean().T

"""### Plotting degree of correlation between the characteristics"""

# Calculate the correlation matrix
df.corr(method='pearson')

# Create a pair plot to visualize the correlations
sns.pairplot(df)

"""### Splitting The Data"""


# Split the dataset into training and testing sets
features_col = list(df.columns)[2:]
x = df[features_col]
y = df[['language']]
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

x_test.shape

"""### Correlation reduction"""


# Scale the features
scale = StandardScaler()
scale.fit(x_train)

x_train = scale.transform(x_train)
x_test = scale.transform(x_test)

# Apply Principal Component Analysis (PCA)
pca = PCA(0.95)
pca.fit(x_train)

print('no of principle component=' + str(pca.n_components_))

x_train = pca.transform(x_train)
x_test = pca.transform(x_test)

x_test.shape

"""### Apply Machine Learning model"""


# Train a Decision Tree Classifier
dt_clf = DecisionTreeClassifier()
dt_clf = dt_clf.fit(x_train, y_train)

# Save the trained model
dt_model = "dt_model.pkl"
with open(dt_model, 'wb') as file:
    pickle.dump(dt_clf, file)

# Load the saved model
with open(dt_model, 'rb') as file:
    dt_clf = pickle.load(file)

dt_clf

# Make predictions on the test set
y_predict = dt_clf.predict(x_test)

# Calculate accuracy score
acc = accuracy_score(y_test, y_predict)

print(acc)

label = ['Afrikaans', 'English', 'Nederlands']

# Create a confusion matrix
confusion_matrix_dt = confusion_matrix(y_test, y_predict)
fig = plt.figure(figsize=(10, 10))

ax = fig.add_subplot(111)
sns.heatmap(confusion_matrix_dt, annot=True)
plt.ylabel('Actual')
plt.xlabel('Predicted')

ax.set_xticklabels(label)
ax.set_yticklabels(label)

title = 'Decision tree Accuracy score=' + str(round(acc * 100, 3)) + "%"
plt.title(title, size=20)

confusion_matrix_dt
