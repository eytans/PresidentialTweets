import pandas
from collections import defaultdict
import argparse
from sklearn.model_selection import train_test_split, KFold


class NaiveBayes(object):
    def __init__(self, priora=1.0/50000, data=None, classes=None):
        self.data = data
        self.classes = classes
        # for key == y value is probability
        self._init_inner()
        self.priora = priora


    def _init_inner(self):
        self._py = {}
        self._pxiy = {}
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
        self._init_inner()
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

    def fit(self, x, y):
        self.data = x
        self.classes = y
        self.train()

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
        print("random set valiidation:")
        for i in range(args.validations):
            xtrain, xtest, ytrain, ytest = train_test_split(learning_data, classes, test_size=args.test/float(args.folds))
            clf = NaiveBayes(priora=priora, data=xtrain, classes=ytrain)
            clf.train()
            print('\tprior: {}. validation: {}. score: {}'.format(priora, i+1, clf.score(xtest, ytest)))
        print("kfold validations:")

        for i in range(args.validations):
            kf = KFold(args.folds, shuffle=True)
            total = 0
            length = 0
            for train_indices, test_indices in kf.split(learning_data):
                train_X = learning_data.loc[learning_data.index[train_indices]]
                train_Y = classes.loc[learning_data.index[train_indices]]
                test_X = learning_data.loc[learning_data.index[test_indices]]
                test_Y = classes.loc[learning_data.index[test_indices]]

                # Train the model, and evaluate it
                clf = NaiveBayes(priora=priora, data=train_X, classes=train_Y)
                clf.train()

                predictions = [clf.classify(x[1]) for x in test_X.iterrows()]
                testing_classes = [y for y in test_Y]
                acc = sum(map(lambda x: x[0] == x[1], zip(predictions, testing_classes)))
                total += float(acc)/len(test_Y)
                length += 1
            print("after {} runs got {} accuracy".format(length, total/length))



if __name__ == '__main__':
    __main__()
