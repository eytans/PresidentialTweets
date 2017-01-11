import pandas as pd
import matplotlib.pyplot as plt
import sklearn as sk

test = pd.read_csv('./tweets_test.csv')
train = pd.read_csv('./tweets_train.csv')

sk.feature_extraction.text.CountVectorizer(?
