import pandas
from collections import defaultdict
import argparse
from sklearn.model_selection import train_test_split


class NaiveBayes(object):
    def __init__(self, priora=1.0/50000, data=None, classes=None):
        self.data = data
        self.classes = classes
        # for key == y value is probability
        self._py = {}
        self._pxiy = {}
        self.priora = priora
        self._regulisers = {}

    def train(self, copy=False):
        if self.classes is None or self.data is None:
            raise RuntimeError("Need to initialize date and classes before train")
        if copy:
            res = NaiveBayes(self.priora, self.data, self.classes)
            return res.train(copy=False)

        if not isinstance(self.data, pandas.DataFrame):
            self.data = pandas.DataFrame(self.data)
        if not isinstance(self.classes, pandas.Series):
            self.classes = pandas.Series(self.classes)

        # MLE calculation of py is number of seen divided by total number of samples
        # TODO: probably dont need to regularise value
        self._py = self.classes.value_counts().map(lambda count: count/float(self.classes.count())).to_dict()

        # MAP calculation of pxiy is a dictionary of y values then x values which is theta_j_i meaning in our case
        # P(X = x_i | Y=yj) where x_i is the i word from vocabulary and yj is the j classification
        # by working with word count and priora we calculate probabilities by addition
        # theta_j_i = count of xi with yj + prioria (alpha_i which is the same for all)

        # split data by classes and then just bag up the words. then we will have the wanted counts.
        self.data['classification'] = self.classes
        y_data = {}
        for yj in self._py.keys():
            y_data[yj] = self.data.loc[self.data['classification'] == yj].drop('classification')
            self._regulisers[yj] = sum(y_data[yj].count()) + self.priora*30
            self._pxiy[yj] = defaultdict(lambda: self.priora)
            for c in y_data[yj]:
                for i, count in y_data[yj][c].value_counts().iteritems():
                    self._pxiy[yj][i] += count
            # might end up huge when claculating probs so regulate the value a bit
            for k in self._pxiy[yj]:
                self._pxiy[yj][k] /= self._regulisers[yj]

        return self

    def classify(self, data):
        max_prob = 0
        max_y = None
        for y, prob in self._py.items():
            for w in data:
                if w not in self._pxiy[y]:
                    self._pxiy[y][w] /= self._regulisers[y]
                prob *= self._pxiy[y][w]
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


def __main__():
    parser = argparse.ArgumentParser()
    parser.add_argument('--validations', type=int, default=5, help='how many sets to use for cross validation')
    parser.add_argument('-p', '--prior', type=float, default=0.1, help='size of jump for prior')
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
        for i in range(args.validations):
            xtrain, xtest, ytrain, ytest = train_test_split(learning_data, classes, test_size=0.25)
            clf = NaiveBayes(priora=priora, data=xtrain, classes=ytrain)
            clf.train()
            print('prior: {}. validation: {}. score: {}'.format(priora, i+1, clf.score(xtest, ytest)))


if __name__ == '__main__':
    __main__()
