import pandas as pd
import re
from string import punctuation
import spacy
import en_core_web_sm
import numpy as np
# from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
# from sklearn.model_selection import train_test_split, GridSearchCV
# from sklearn.metrics import accuracy_score
# from joblib import dump, load
# from sklearn.preprocessing import MinMaxScaler
# from sklearn import svm, tree
# from sklearn.neighbors import KNeighborsClassifier
# from sklearn.pipeline import Pipeline
# from sklearn.base import TransformerMixin, BaseEstimator
# from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
# from sklearn.naive_bayes import MultinomialNB, BernoulliNB
# from sklearn.linear_model import LogisticRegressionCV, LogisticRegression
# from sklearn.neighbors import KNeighborsClassifier


class customtransformer(TransformerMixin, BaseEstimator):
    """This is the custom transformer for the pipeline.Please export export this file with the pipleline 
    
    Notes
    -----
    Remember to import this file when you load the pickle of the pipeline, else the pickle will not run
    

    Args:
        TransformerMixin ([sklearnclass]):
        BaseEstimator ([sklearnclass]):
    """

    def __init__(self, negation, stopwords, cleaning, lema):
        self.negation = negation
        self.stopwords = stopwords
        self.cleaning = cleaning
        self.lema = lema
        self.nlp = spacy.load("en_core_web_sm")

    def stopwordsremove(self, data):
        #     This function removes stopwords using spacy stopwords list
        finaltweet = []
        for tweet in data:
            text = self.nlp(tweet)

            cleansent = []
            for token in text:
                #             checking for stopwords
                stopword = token.is_stop
                if stopword:
                    pass
                else:
                    cleansent.append(token.text)

            finaltweet.append(" ".join(cleansent))

        return finaltweet

    def lemmatizingtweet(self, data):
        """this function lemmatize the words in the tweet
        """

        finaltweet = []
        for tweet in data:
            processedtweet = self.nlp(tweet)
            lem = []
            for text in processedtweet:
                textLema = text.lemma_

                #             if it is a pronoun maintain the text
                if textLema == "-PRON-":
                    textLema = text.text

                lem.append(textLema)
            finaltweet.append(" ".join(lem))
        return finaltweet

    def cleaningtweet(self, data):
        #     cleaning tweets, removing stopwords, emojis, links,@mentions and lemmatizing tweets

        clean_tweet = []
        for text in data:
            text = text.lower()

            # converting hashtags to sentences
            hashtags = re.findall("#\w+\d?", text)
            if hashtags:
                for hash_tag in hashtags:
                    text = re.sub(hash_tag, " ".join(re.split("([A-Z][a-z]+)", hash_tag)).replace("#", ""), text)

            #     removing links
            text = re.sub("(?P<url>https?://[^\s]+)", "", text)
            # removing @mentions
            text = re.sub("\@\d*?\w+\d?", "MENTION", text)
            # removing numbers
            text = re.sub("\d+", "NUMBER", text)

            clean_tweet.append(text.strip())

        return clean_tweet

    def negations(self, data):
        nlp = spacy.load("en_core_web_sm")
        #     This function converts negations to one word for bag of words model eg. "do not sing" will be "do notsing"
        finaltweet = []
        for tweet in data:

            tweet = re.sub("\s{2,}", " ", tweet)
            #         using positive lookahead to get the word after "not"
            notfind = re.findall("(?<=not\s)\w+", tweet, re.IGNORECASE)

            #     check for match
            if notfind:

                for text in notfind:
                    check = nlp(text)

                    # check for stopwords eg. a,am,i else " I am not a boy" will be converted to "I am nota boy"
                    # instead of "I am notboy"

                    if check[0].is_stop:
                        try:
                            check = str(check[0])
                            cleantweet = re.sub(f"\\b{check}\\b", "", tweet)
                            cleantweet = re.sub("\s{2,}", " ", cleantweet)

                            clean = re.findall("(?<=not\s)\w+", cleantweet, re.IGNORECASE)

                            if len(clean) < 1:
                                tweet = re.sub(text, f'not{text}', tweet)
                            else:
                                tweet = re.sub(str(clean[0]), f"not{str(clean[0])}", cleantweet)
                        except:
                            raise
                    else:
                        #  if match and the hit is not a stopword, loop through the matched list and replace the text
                        text1 = f"\\b{text}\\b"
                        tweet = re.sub(text1, f"not{str(text)}", tweet)

                #     append the final tweet

                tweet = re.sub("not ", "", tweet, re.IGNORECASE)
                tweet = re.sub("\s{2,}", " ", tweet)
                finaltweet.append(tweet)
            else:
                # if not in sentence send it back
                finaltweet.append(tweet)
        return finaltweet

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        x = X.copy()
        if self.negation and self.lema and self.cleaning:

            x = self.lemmatizingtweet(x)
            x = self.negations(x)
            x = self.cleaningtweet(x)

        elif self.lema and self.cleaning:

            x = self.lemmatizingtweet(x)
            x = self.cleaningtweet(x)

        elif self.negation and self.cleaning:

            x = self.negations(x)
            x = self.cleaningtweet(x)

        elif self.negation and self.lema:
            x = self.negation(x)
            x = self.lemmatizingtweet(x)

        elif self.lema:
            x = self.lemmatizingtweet(x)

        if self.stopwords:
            x = self.stopwordsremove(x)

        return x


class arrayconverter(TransformerMixin, BaseEstimator):
    """This class convert the vectorizer to an array format for scaling
    """

    def __init__(self, arrays=True):
        self.array = arrays

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        if self.array:
            x = X.copy()
            x = x.toarray()
            return x
