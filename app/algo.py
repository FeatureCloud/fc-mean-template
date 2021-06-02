import numpy as np
from sklearn.linear_model import LinearRegression


class Client:
    model = LinearRegression()
    X = None
    y = None
    X_test = None
    y_test = None
    X_offset_local = None
    y_offset_local = None
    X_scale_local = None
    X_offset_global = None
    y_offset_global = None
    X_scale_global = None

    def set_coefs(self, coef):
        self.model.coef_ = coef
        self.model._set_intercept(self.X_offset_global, self.y_offset_global, self.X_scale_global)

    def set_global_offsets(self, aggregated_preprocessing):
        self.X_offset_global = aggregated_preprocessing[0]
        self.y_offset_global = aggregated_preprocessing[1]
        self.X_scale_global = aggregated_preprocessing[2]

        self.X -= self.X_offset_global
        self.y -= self.y_offset_global

    def local_preprocessing(self):
        accept_sparse = False if self.model.positive else ['csr', 'csc', 'coo']

        self.X, self.y = self.model._validate_data(self.X, self.y, accept_sparse=accept_sparse, y_numeric=True,
                                                   multi_output=True)

        # if regr.sample_weight is not None:
        #    sample_weight = regr._check_sample_weight(sample_weight, X,dtype=X.dtype)
        _, _, self.X_offset_local, self.y_offset_local, self.X_scale_local = self.model._preprocess_data(
            self.X, self.y, fit_intercept=self.model.fit_intercept, normalize=False,
            copy=self.model.copy_X, sample_weight=None, return_mean=True)

    def local_computation(self):
        XT_X_matrix = np.dot(self.X.T, self.X)
        XT_y_vector = np.dot(self.X.T, self.y)

        return XT_X_matrix, XT_y_vector

class Coordinator(Client):

    def aggregate_offsets_(self, offsets, samples_per_client, overall_sample_size):
        weighted_X_offsets = offsets[0] * samples_per_client[0]
        for i in range(1, len(offsets)):
            weighted_X_offsets = weighted_X_offsets + (offsets[i] * samples_per_client[i])
        X_offset_global = weighted_X_offsets / overall_sample_size

        return X_offset_global

    def aggregate_matrices_(self, matrices):
        matrix = matrices[0]
        for i in range(1, len(matrices)):
            matrix = np.add(matrix, matrices[i])
        matrix_global = matrix

        return matrix_global

    def aggregate_preprocessing(self, preprocessing):
        X_offsets = [client[0] for client in preprocessing]
        X_offsets = [el for el in X_offsets]
        y_offsets = [client[1] for client in preprocessing]
        y_offsets = [el for el in y_offsets]
        X_scales = [client[2] for client in preprocessing]
        X_scales = [el for el in X_scales]

        samples_per_client = [client[3] for client in preprocessing]
        samples_per_client = [int(el) for el in samples_per_client]
        overall_sample_size = np.sum(samples_per_client)
        X_offset_global = self.aggregate_offsets_(X_offsets, samples_per_client, overall_sample_size)
        y_offset_global = self.aggregate_offsets_(y_offsets, samples_per_client, overall_sample_size)
        X_scale_global = self.aggregate_offsets_(X_scales, samples_per_client, overall_sample_size)

        return X_offset_global, y_offset_global, X_scale_global

    def aggregate_beta(self, local_results):
        XT_X_matrices = [client[0] for client in local_results]
        XT_X_matrix_global = self.aggregate_matrices_(XT_X_matrices)

        XT_y_vectors = [client[1] for client in local_results]
        XT_y_vector_global = self.aggregate_matrices_(XT_y_vectors)

        XT_X_matrix_inverse = np.linalg.inv(XT_X_matrix_global)
        beta_vector = np.dot(XT_X_matrix_inverse, XT_y_vector_global)

        return beta_vector
