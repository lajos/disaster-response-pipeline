# import libraries
import sys
import re
import pandas as pd
import numpy as np
import nltk
import pickle

from sqlalchemy import create_engine
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV
from sklearn.base import BaseEstimator,TransformerMixin

# download nltk data
nltk.download(['punkt', 'wordnet', 'omw-1.4', 'stopwords'])


def load_data(database_filepath):
    '''
    Load data from SQLite database.

    Input:
      database_filepath: path and filename of database to load
    Output:
      X: DataFrame of features
      y: DataFrame of labels
      category_names: list of category names
    '''
    engine = create_engine(f'sqlite:///{database_filepath}')
    df = pd.read_sql_table('disaster_messages', engine)
    X = df['message']
    y = df.drop(['message', 'genre', 'id', 'original'], axis=1)
    category_names = y.columns

    return X, y, category_names


def tokenize(text):
    '''
    Tokenize text.

    Input:
      text: text to tokenize
    Output:
      words: list of lemmatized words
    '''
    text = text.lower()
    text = re.sub(r'[^a-z]', " ", text)

    tokens = nltk.word_tokenize(text)

    lemmatizer = nltk.WordNetLemmatizer()
    stop_words = nltk.corpus.stopwords.words("english")

    tokens = [lemmatizer.lemmatize(w) for w in tokens if w not in stop_words]

    return tokens


class StartingVerbExtractor(BaseEstimator, TransformerMixin):
    '''
    A custom transformer class that tests if the
    the sentence starts with a verb.
    '''
    def starting_verb(self, text):
        sentence_list = nltk.sent_tokenize(text)
        for sentence in sentence_list:
            pos_tags = nltk.pos_tag(tokenize(sentence))
            if not len(pos_tags):
                return False
            first_word, first_tag = pos_tags[0]
            if first_tag in ['VB', 'VBP'] or first_word == 'RT':
                return True
        return False

    def fit(self, x, y=None):
        return self

    def transform(self, X):
        X_tagged = pd.Series(X).apply(self.starting_verb)
        return pd.DataFrame(X_tagged)


def build_model():
    '''
    Build model pipeline

    Output:
      pipeline: a Scikit ML pipeline 
    '''
    pipeline = Pipeline([
        ('features', FeatureUnion([
            ('text_pipeline', Pipeline([
                ('vect', CountVectorizer(tokenizer=tokenize)),
                ('tfidf', TfidfTransformer())
            ])),
            ('stverb', StartingVerbExtractor())
        ])),
        ('clf', MultiOutputClassifier(AdaBoostClassifier(random_state=42)))
    ])
    return pipeline


def evaluate_model(model, X_test, Y_test, category_names):
    y_test_pred = model.predict(X_test)
    print(classification_report(Y_test.values, y_test_pred, zero_division=0, target_names=category_names))


def save_model(model, model_filepath):
    '''
    Save the trained model

    Input:
      model: SciKit ML model
      model_filepath: path and name of file to save model to
    '''
    with open(model_filepath, 'wb') as pickle_file:
        pickle.dump(model, pickle_file)


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()