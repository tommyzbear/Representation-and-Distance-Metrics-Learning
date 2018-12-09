from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import Normalizer
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import MaxAbsScaler
from sklearn.preprocessing import RobustScaler


class Normaliser:
    def __init__(self):
        self.normaliser = None
        self.transformed_train_data = None
        self.transformed_test_data = None

    def fit(self, train_data, method='Std'):
        if method == 'Std':
            self.normaliser = StandardScaler()
        elif method == 'l1':
            self.normaliser = Normalizer(norm='l1')
        elif method == 'l2':
            self.normaliser = Normalizer()
        elif method == 'max':
            self.normaliser = Normalizer(norm='max')
        elif method == 'MinMax':
            self.normaliser = MinMaxScaler()
        elif method == 'MaxAbs':
            self.normaliser = MaxAbsScaler()
        elif method == 'Robust':
            self.normaliser = RobustScaler()
        else:
            raise ValueError("Method %s not supported." % method)

        self.normaliser.fit(train_data)

    def fit_transform(self, train_data, method='Std'):
        if method == 'Std':
            self.normaliser = StandardScaler()
        elif method == 'l1':
            self.normaliser = Normalizer(norm='l1')
        elif method == 'l2':
            self.normaliser = Normalizer()
        elif method == 'max':
            self.normaliser = Normalizer(norm='max')
        elif method == 'MinMax':
            self.normaliser = MinMaxScaler()
        elif method == 'MaxAbs':
            self.normaliser = MaxAbsScaler()
        elif method == 'Robust':
            self.normaliser = RobustScaler()
        else:
            raise ValueError("Method %s not supported." % method)

        self.transformed_train_data = self.normaliser.fit_transform(train_data)
        return self.transformed_train_data

    def transform(self, test_data):
        self.transformed_test_data = self.normaliser.transform(test_data)
        return self.transformed_test_data
