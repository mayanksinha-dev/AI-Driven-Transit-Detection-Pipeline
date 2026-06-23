from sklearn.base import BaseEstimator, TransformerMixin

class QualityFilter(BaseEstimator, TransformerMixin):

    def fit(self, X, y=None):
        return self

    def transform(self, data):

        output = []

        for d in data:

            mask = d["quality"] == 0

            d["time"] = d["time"][mask]
            d["flux"] = d["flux"][mask]

            output.append(d)

        return output