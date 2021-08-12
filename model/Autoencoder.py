import keras
from keras.layers import Input, Dense
from keras.models import Model
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')

class Autoencoder:
    """
    Autoencoder for learning latent space representation architecture simplified for only one layer
    """
    def __init__(self, latent_dim=32, activation= 'relu', epochs=200, batch_size=128):
        self.latent_dim = latent_dim
        self.activation = activation
        self.epochs = epochs
        self.batch_size = batch_size
        self.autoencoder = None
        self.encoder = None
        self.decoder = None
        self.his= None

    def _compile(self, input_dim):
        """
        Compile the computational graph.
        :param input_dim:
        """

        input_vec= Input(shape=(input_dim,))
        encoded_input= Dense(self.latent_dim, activation= self.activation)(input_vec)
        decoded_layer= Dense(input_dim, activation= self.activation)(encoded_input)

        self.decoder= Model(encoded_input, self.autoencoder.layers[-1](encoded_input))
        self.autoencoder.compile(optimizer= 'adam', loss=keras.lossses.mean_squared_error)

    def fit(self, X):
        if not self.autoencoder:
            self._compile(X.shape[1])

        X_train, X_test= train_test_split(X)
        self.his= self.autoencoder.fit(X_train, X_train,
                                       epochs=200,
                                       batch_size= 128,
                                       shuffle= True,
                                       validation_data=(X_test, X_test), verbose=0)
