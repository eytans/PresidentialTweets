import pandas
from collections import defaultdict
import argparse
from sklearn.model_selection import train_test_split, KFold, cross_val_score
import sklearn.base


class NaiveBayes(sklearn.base.BaseEstimator):
    def __init__(self, priora=1.0/50000):
        self.data_ = None
        self.classes_ = None
        # for key == y value is probability
        self.priora = priora
        self.py_ = {}
        self.pxiy_ = {}
        self.regulisers_ = {}

    def fit(self, X, y, copy=False):
        self.classes_ = y
        self.data_ = X
        self.py_ = {}
        self.pxiy_ = {}
        self.regulisers_ = {}

        if self.classes_ is None or self.data_ is None:
            raise RuntimeError("Need to initialize date and classes before train")
        if copy:
            res = NaiveBayes(self.priora)
            return res.fit(X, y)

        if not isinstance(self.data_, pandas.DataFrame):
            self.data_ = pandas.DataFrame(self.data_)
        if not isinstance(self.classes_, pandas.Series):
            self.classes_ = pandas.Series(self.classes_)
        # MLE calculation of py is number of seen divided by total number of samples
        # TODO: probably dont need to regularise value
        self.py_ = self.classes_.value_counts().map(lambda count: count / float(self.classes_.count())).to_dict()

        # MAP calculation of pxiy is a dictionary of y values then x values which is theta_j_i meaning in our case
        # P(X = x_i | Y=yj) where x_i is the i word from vocabulary and yj is the j classification
        # by working with word count and priora we calculate probabilities by addition
        # theta_j_i = count of xi with yj + prioria (alpha_i which is the same for all)

        # split data by classes and then just bag up the words. then we will have the wanted counts.
        y_data = {}
        for yj in self.py_.keys():
            y_data[yj] = self.data_.loc[self.classes_ == yj]
            self.regulisers_[yj] = sum(y_data[yj].count()) + self.priora * 30
            self.pxiy_[yj] = defaultdict(lambda: self.priora)
            for c in y_data[yj]:
                for i, count in y_data[yj][c].value_counts().iteritems():
                    self.pxiy_[yj][i] += count
            # might end up huge when claculating probs so regulate the value a bit
            for k in self.pxiy_[yj]:
                self.pxiy_[yj][k] /= self.regulisers_[yj]

        return self

    def classify(self, data):
        max_prob = 0
        max_y = None
        for y, prob in self.py_.items():
            for w in data:
                if w not in self.pxiy_[y]:
                    self.pxiy_[y][w] /= self.regulisers_[y]
                prob *= self.pxiy_[y][w]
            if prob > max_prob:
                max_prob = prob
                max_y = y
        return max_y

    def score(self, testx, testy):
        total = 0
        res = 0
        for i, tup in enumerate(testx.itertuples()):
            data = [w for w in tup if isinstance(w, str)]
            y = testy.iloc[i]
            total += 1
            res += y == self.classify(data)
        return float(res)/total

    def predict(self, x):
        return self.classify(x)

def __main__():
    parser = argparse.ArgumentParser()
    parser.add_argument('--validations', type=int, default=5, help='how many sets to use for cross validation')
    parser.add_argument('-f', '--folds', type=int, default=4, help='amount of folds')
    parser.add_argument('-t', '--test', type=int, default=1, help='fold in test')
    parser.add_argument('-p', '--prior', type=float, default=0.05, help='size of jump for prior')
    parser.add_argument('path', help='path to data csv')
    args = parser.parse_args()

    data = pandas.DataFrame.from_csv(args.path)
    classes = data['handle']
    cols = list(data.columns)
    cols.remove('text')
    data.drop(cols)

    data['tl'] = data['text'].map(lambda x: len(x.split()))
    learning_data = data['text'].apply(lambda x: pandas.Series(x.split()))
    priora = 0.00001
    while priora <= 1:
        priora += args.prior
        if priora > 1:
            continue
        cv = 10
        print("prior - {} . accuracy - {}".format(priora,
                            sum(cross_val_score(NaiveBayes(priora=priora), learning_data, classes, cv=cv))/float(cv)))

if __name__ == '__main__':
    __main__()
