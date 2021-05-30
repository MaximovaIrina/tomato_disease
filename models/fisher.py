from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB


class Fisher:
    def __init__(self):
        self.LDA = None
        self.Bayes = False

    def fit(self, x, y):
        self.LDA = LinearDiscriminantAnalysis()
        x = self.LDA.fit_transform(x, y)
        self.Bayes = GaussianNB().fit(x, y)

    def predict(self, x):
        x = self.LDA.transform(x.copy())
        prediction = self.Bayes.predict(x)
        return prediction
