import pandas
from collections import defaultdict


class NaiveBayes(object):
    def __init__(self, priora=0.5, data=None, classes=None):
        self.data = data
        self.classes = classes
        # for key == y value is probability
        self._py = {}
        self._pxiy = None
        self.priora = priora
        self._regulizer = 0

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
        self._regulizer = float(self.data.shape[0])
        y_data = {}
        for yj in self._py.keys():
            y_data[yj] = self.data.loc[self.data['classification'] == yj].drop('classification')
            self._pxiy[yj] = defaultdict(lambda: self.priora)
            for c in y_data[yj]:
                for i, count in y_data[yj][c].value_counts().iteritems():
                    self._pxiy[yj][i] += count
            # might end up huge when claculating probs so regulate the value a bit
            for k in self._pxiy[yj]:
                self._pxiy[yj][k] /= self._regulizer

        return self

    def classify(self, data):
        max_prob = 0
        max_y = None
        for y, prob in self._py.items():
            for w in data:
                if w not in self._pxiy[y]:
                    self._pxiy[y][w] /= self._regulizer
                prob *= self._pxiy[y][w]
            if prob > max_prob:
                max_prob = prob
                max_y = y
        return max_y
