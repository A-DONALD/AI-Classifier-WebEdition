from flask import Flask, render_template, jsonify, request
import pickle
import os
import re
import webbrowser
import nltk
from bs4 import BeautifulSoup
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from nltk.corpus import stopwords
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer

app = Flask(__name__)

nltk.download('wordnet')
nltk.download('stopwords')

data_dir = os.path.dirname(os.path.abspath(__file__))
dataSets_dir = os.path.join(os.path.dirname(data_dir), 'data_sets')
model_dir = os.path.join(data_dir, 'model')
data = []
RandomForest_classifier = ''
KNeighbors_classifier = ''
multinomial_classifier = ''
RandomForest_accuracy = 0
KNeighbors_accuracy = 0
multinomial_accuracy = 0

# Load dictionary
print("Loading files ", end=" ")
for filename in os.listdir(dataSets_dir):
    if filename.endswith('.sgm'):
        try:
            print("*", end=" ")
            with open(os.path.join(dataSets_dir, filename), 'r', encoding='utf-8') as f:
                content = f.read()
        except UnicodeDecodeError:
            continue

        soup = BeautifulSoup(content, 'html.parser')
        reuters = soup.findAll('reuters')

        for reuter in reuters:
            if reuter['topics'] == "YES" and reuter.topics.text != '' and reuter.body is not None:
                data.append({'content': reuter.body.text, 'target': reuter.topics.d.text, 'lewissplit': reuter['lewissplit']})

X, y = [item['content'] for item in data], [item['target'] for item in data]
# Text preprocessing
documents = []

stemmer = WordNetLemmatizer()

for sen in range(0, len(X)):
    # Remove all the special characters
    document = re.sub(r'\W', ' ', str(X[sen]))

    # remove all single characters
    document = re.sub(r'\s+[a-zA-Z]\s+', ' ', document)

    # Remove single characters from the start
    document = re.sub(r'\^[a-zA-Z]\s+', ' ', document)

    # Converting to Lowercase
    document = document.lower()

    # Lemmatization
    document = document.split()

    document = [stemmer.lemmatize(word) for word in document]
    document = ' '.join(document)

    documents.append(document)

# Convert word to number: type bag of words
vectorizer = CountVectorizer(max_features=1500, min_df=5, max_df=0.7, stop_words=stopwords.words('english'))
X = vectorizer.fit_transform(documents).toarray()

# Find TFIDF
tfidfconverter = TfidfTransformer()
X = tfidfconverter.fit_transform(X).toarray()

# Training and Testing Sets
X_train, X_test, y_train, y_test = [], [], [], []
for i, x in enumerate(X):
    if data[i]['lewissplit'].lower() == 'train':
        X_train.append(x)
        y_train.append(y[i])
    elif data[i]['lewissplit'].lower() == 'test':
        X_test.append(x)
        y_test.append(y[i])

if os.path.exists(os.path.join(model_dir, "RandomForest_classifier.pkl"))\
        and os.path.exists(os.path.join(model_dir, "multinomial_classifier.pkl")) \
        and os.path.exists(os.path.join(model_dir, "KNeighbors_classifier.pkl")):
    # Open the model
    print(' Loading model')

    with open(os.path.join(model_dir, 'multinomial_classifier.pkl'), 'rb') as training_model:
        model = pickle.load(training_model)
    multinomial_classifier = model
    y_pred = multinomial_classifier.predict(X_test)
    multinomial_accuracy = accuracy_score(y_test, y_pred)

    with open(os.path.join(model_dir, 'KNeighbors_classifier.pkl'), 'rb') as training_model:
        model = pickle.load(training_model)
    KNeighbors_classifier = model
    y_pred = KNeighbors_classifier.predict(X_test)
    KNeighbors_accuracy = accuracy_score(y_test, y_pred)

    with open(os.path.join(model_dir, 'RandomForest_classifier.pkl'), 'rb') as training_model:
        model = pickle.load(training_model)
    RandomForest_classifier = model
    y_pred = RandomForest_classifier.predict(X_test)
    RandomForest_accuracy = accuracy_score(y_test, y_pred)

# if in model we doesn't have all models of the classifier, we will create the model of that one
else:
    print('Creating model')

    # Multinomial
    multinomial_classifier = MultinomialNB()
    multinomial_classifier.fit(X_train, y_train)
    y_pred = multinomial_classifier.predict(X_test)
    multinomial_accuracy = accuracy_score(y_test, y_pred)
    with open(os.path.join(model_dir, 'multinomial_classifier.pkl'), 'wb') as picklefile:
        pickle.dump(multinomial_classifier, picklefile)

    # KNeighborsClassifier
    KNeighbors_classifier = KNeighborsClassifier(n_neighbors=5)
    KNeighbors_classifier.fit(X_train, y_train)
    y_pred = KNeighbors_classifier.predict(X_test)
    KNeighbors_accuracy = accuracy_score(y_test, y_pred)
    with open(os.path.join(model_dir, 'KNeighbors_classifier.pkl'), 'wb') as picklefile:
        pickle.dump(KNeighbors_classifier, picklefile)

    # Algorithm of random forest
    RandomForest_classifier = RandomForestClassifier(n_estimators=1000, random_state=0)
    RandomForest_classifier.fit(X_train, y_train)
    y_pred = RandomForest_classifier.predict(X_test)
    RandomForest_accuracy = accuracy_score(y_test, y_pred)
    with open(os.path.join(model_dir, 'RandomForest_classifier.pkl'), 'wb') as picklefile:
        pickle.dump(RandomForest_classifier, picklefile)


def preprocess_text(text):
    from nltk.stem import WordNetLemmatizer

    stemmer = WordNetLemmatizer()

    document = re.sub(r'\W', ' ', text)
    document = re.sub(r'\s+[a-zA-Z]\s+', ' ', document)
    document = re.sub(r'\^[a-zA-Z]\s+', ' ', document)
    document = document.lower()
    document = document.split()
    document = [stemmer.lemmatize(word) for word in document]
    document = ' '.join(document)

    return document


@app.route('/')
def hello():
    return render_template('index.html')


@app.route('/txt-classify', methods=['POST'])
def classify_text():
    text = request.form.get('text')
    preprocessed_text = preprocess_text(text)
    features = vectorizer.transform([preprocessed_text]).toarray()
    randomforest = RandomForest_classifier.predict(features)
    kneighbors = KNeighbors_classifier.predict(features)
    multinomial = multinomial_classifier.predict(features)
    words = [randomforest[0], kneighbors[0], multinomial[0]]
    print(words)

    return jsonify(words=words)


def main():
    url = f"http://localhost:5000"
    webbrowser.open(url)
    app.run(debug=False, port='5000')