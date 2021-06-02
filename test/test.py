import unittest

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

from app.algo import Coordinator, Client

def parse_input(path):
    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
    X = pd.read_csv(path, sep=",").select_dtypes(include=numerics).dropna()
    y = X.loc[:, "TARGET_deathRate"]
    X = X.drop("TARGET_deathRate", axis=1)

    return X, y


class TestLinearRegression(unittest.TestCase):
    def setUp(self):

        X, y = parse_input("client1/client1.csv")
        client = Coordinator()
        client.X = X
        client.y = y

        X, y = parse_input("client2/client2.csv")
        client2 = Client()
        client2.X = X
        client2.y = y

        client.local_preprocessing()
        client2.local_preprocessing()
        data = client.aggregate_preprocessing(
            [[client.X_offset_local, client.y_offset_local, client.X_scale_local, client.X.shape[0]],
             [client2.X_offset_local, client2.y_offset_local, client2.X_scale_local, client2.X.shape[0]]])
        client.set_global_offsets(data)
        client2.set_global_offsets(data)

        xtx, xty = client.local_computation()
        xtx2, xty2 = client2.local_computation()
        global_coefs = client.aggregate_beta([[xtx, xty], [xtx2, xty2]])
        client.set_coefs(global_coefs)
        client2.set_coefs(global_coefs)

        self.model1 = client.model
        self.model2 = client2.model

        numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
        X = pd.read_csv("cancer_reg.csv", sep=",").select_dtypes(include=numerics).dropna()
        y = X.loc[:, "TARGET_deathRate"]
        X = X.drop("TARGET_deathRate", axis=1)
        self.global_model = LinearRegression()
        self.global_model.fit(X, y)

        X = pd.read_csv("client1/client1.csv", sep=",").select_dtypes(include=numerics).dropna()
        self.y_test = X.loc[:, "TARGET_deathRate"]
        self.X_test = X.drop("TARGET_deathRate", axis=1)

    def test_intercept(self):
        np.allclose(self.global_model.intercept_, self.model1.intercept_)
        np.allclose(self.global_model.intercept_, self.model2.intercept_)
        np.allclose(self.model2.intercept_, self.model1.intercept_)

    def test_coef(self):
        np.allclose(self.global_model.coef_, self.model1.coef_)
        np.allclose(self.global_model.coef_, self.model2.coef_)
        np.allclose(self.model2.coef_, self.model1.coef_)

    def test_prediction(self):
        y_pred_global = self.global_model.predict(self.X_test)
        y_pred1 = self.model1.predict(self.X_test)
        y_pred2 = self.model2.predict(self.X_test)

        np.allclose(y_pred_global, y_pred1)
        np.allclose(y_pred_global, y_pred2)
        np.allclose(y_pred2, y_pred1)


if __name__ == "__main__":
    unittest.main()
