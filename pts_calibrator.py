import os
import tensorflow as tf
import numpy as np

class PTS_calibrator():
    """Class for Parameterized Temperature Scaling (PTS)"""
    def __init__(
        self,
        epochs,
        lr,
        weight_decay,
        batch_size,
        nlayers,
        n_nodes,
        length_logits,
        top_k_logits
    ):
        """
        Args:
            epochs (int): number of epochs for PTS model tuning
            lr (float): learning rate of PTS model
            weight_decay (float): lambda for weight decay in loss function
            batch_size (int): batch_size for tuning
            n_layers (int): number of layers of PTS model
            n_nodes (int): number of nodes of each hidden layer
            length_logits (int): length of logits vector
            top_k_logits (int): top k logits used for tuning
        """

        self.epochs = epochs
        self.lr = lr
        self.weight_decay = weight_decay
        self.batch_size = batch_size
        self.nlayers = nlayers
        self.n_nodes = n_nodes
        self.length_logits = length_logits
        self.top_k_logits = top_k_logits

        #Build model
        input_logits = tf.keras.Input(shape=(self.length_logits))
        l2_reg = tf.keras.regularizers.l2(self.weight_decay)

        #Sort logits in descending order and keep top k logits
        t = tf.reshape(
            tf.sort(input_logits, axis=-1,
                    direction='DESCENDING', name='sort'),
            (-1,self.length_logits)
        )
        t = t[:,:self.top_k_logits]

        for _ in range(nlayers):
            t = tf.keras.layers.Dense(self.n_nodes, activation='relu',
                                      kernel_regularizer=l2_reg)(t)

        t = tf.keras.layers.Dense(1, activation=None, name="temperature")(t)
        temperature = tf.math.abs(t)
        x = input_logits/tf.clip_by_value(temperature,1e-12,1e12)
        x = tf.keras.layers.Softmax()(x)

        self.model = tf.keras.Model(input_logits, x, name="model")

        self.model.compile(
            optimizer=tf.keras.optimizers.Adam(self.lr),
            loss=tf.keras.losses.MeanSquaredError())
        self.model.summary()


    def tune(self, logits, labels, clip=1e2):
        """
        Tune PTS model
        Args:
            logits (tf.tensor or np.array): logits of shape (N,length_logits)
            labels (tf.tensor or np.array): labels of shape (N,length_logits)
        """

        if not tf.is_tensor(logits):
            logits = tf.convert_to_tensor(logits)
        if not tf.is_tensor(labels):
            labels = tf.convert_to_tensor(labels)

        assert logits.get_shape()[1] == self.length_logits, "logits need to have same length as length_logits!"
        assert labels.get_shape()[1] == self.length_logits, "labels need to have same length as length_logits!"

        logits = np.clip(logits, -clip, clip)

        self.model.fit(logits, labels, epochs=self.epochs, batch_size=self.batch_size)


    def calibrate(self, logits, clip=1e2):
        """
        Calibrate logits with PTS model
        Args:
            logits (tf.tensor): logits of shape (N,length_logits)
        Return:
            calibrated softmax probability distribution (np.array)
        """

        if not tf.is_tensor(logits):
            logits = tf.convert_to_tensor(logits)

        assert logits.get_shape()[1] == self.length_logits, "logits need to have same length as length_logits!"

        calibrated_probs = self.model.predict(tf.clip_by_value(logits, -clip, clip))

        return calibrated_probs


    def save(self, path = "./"):
        """Save PTS model parameters"""

        if not os.path.exists(path):
            os.makedirs(path)

        print("Save PTS model to: ", os.path.join(path, "pts_model.h5"))
        self.model.save_weights(os.path.join(path, "pts_model.h5"))


    def load(self, path = "./"):
        """Load PTS model parameters"""

        print("Load PTS model from: ", os.path.join(path, "pts_model.h5"))
        self.model.load_weights(os.path.join(path, "pts_model.h5"))
