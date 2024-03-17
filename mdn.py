"""
Mixed Density Network
Using Gaussian Mixture

This is for testing against the EBM
Currently not written right for what we want to do but generated some base code

BROKEN
"""

import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
from sklearn.mixture import GaussianMixture

class MDN:
    def __init__(self, input_dim, num_components):
        self.input_dim = input_dim
        self.num_components = num_components
        self.model = self.build_model()
        self.initial_mu = None
        self.initial_sigma = None
        self.initial_alpha = None
    
    def build_model(self):
        input_layer = tf.keras.layers.Input(shape=(self.input_dim,))
        dense = tf.keras.layers.Dense(64, activation='relu')(input_layer)
        dense = tf.keras.layers.Dense(64, activation='relu')(dense)

        alpha = tf.keras.layers.Dense(self.num_components, activation='softmax')(dense)
        mu = tf.keras.layers.Dense(self.num_components)(dense)
        sigma = tf.keras.layers.Dense(self.num_components, activation='softplus')(dense)

        model = tf.keras.models.Model(inputs=input_layer, outputs=[alpha, mu, sigma])
        return model
    
    def mdn_loss(self, alpha, mu, sigma, y_true):
        gm = tfp.distributions.MixtureSameFamily(
            mixture_distribution=tfp.distributions.Categorical(probs=alpha),
            components_distribution=tfp.distributions.Normal(loc=mu, scale=sigma))
        return -tf.reduce_mean(gm.log_prob(y_true))
    
    def fit(self, x_train, y_train, epochs=100, batch_size=32):
        self.model.compile(optimizer='adam', loss=self.mdn_loss)
        self.model.fit(x_train, y_train, epochs=epochs, batch_size=batch_size)
    
    def predict(self, x_test):
        alpha_pred, mu_pred, sigma_pred = self.model.predict(x_test)
        samples = np.array([np.random.choice(self.num_components, p=alpha_pred[i]) for i in range(len(x_test))])
        y_pred = np.array([np.random.normal(mu_pred[i, samples[i]], sigma_pred[i, samples[i]]) for i in range(len(x_test))])
        return y_pred
    
    def fit_initial_gmm(self, y_train):
        gmm = GaussianMixture(n_components=self.num_components)
        gmm.fit(y_train)
        self.initial_mu = gmm.means_.flatten()
        self.initial_sigma = np.sqrt(gmm.covariances_).flatten()
        self.initial_alpha = gmm.weights_

    def train(self):
        self.fit_initial_gmm(self.input_dim)
        self.fit()