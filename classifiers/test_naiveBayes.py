import pandas as pd
from classifiers import NaiveBayes
from sklearn.model_selection import cross_val_score
import os
import dill
from unittest import TestCase

cur_dir = os.path.dirname(__file__)

class TestNaiveBayes(TestCase):
    classifier_path = 'clf.p'
    def setUp(self):
        self.trainx, self.trainy = NaiveBayes.transform_data(
            pd.DataFrame.from_csv(os.path.join(cur_dir, '..', 'tweets_train.csv')))
        if os.path.exists(self.classifier_path):
            self.clf = dill.load(open(self.classifier_path, 'rb'))
        else:
            self.clf = NaiveBayes.NaiveBayes()
            self.clf.fit(self.trainx, self.trainy)
            dill.dump(self.clf, open(self.classifier_path, 'wb'))

    def test_classify_accuracy(self):
        res = sum(cross_val_score(self.clf, self.trainx, self.trainy))/3
        self.assertGreater(res, 0.75)

    def test_model_sane(self):
        self.assertGreater(self.clf.score(self.trainx, self.trainy), 0.9)

    def test_prediction_legal(self):
        results = pd.Series(self.clf.classify(self.trainx))
        self.assertEquals(0, results.isnull().sum())