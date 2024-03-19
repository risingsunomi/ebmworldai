"""
Energy Based Model

Using NCE loss
"""

import tensorflow as tf
from cv2.typing import MatLike

class EBM(tf.keras.Model):
    def __init__(self, image_array: MatLike, video_capture: any, batch_size: int=64):
        super(EBM, self).__init__()
        self.batch_size = batch_size
        self.image_array = image_array

        # use AlexNet CNN as a feature extractor/encoder
        self.encoder = tf.keras.Sequential([
            tf.keras.layers.experimental.preprocessing.Resizing(224, 224, interpolation="bilinear", input_shape=self.image_array.shape[1:]),
            tf.keras.layers.Conv2D(96, 11, strides=4, padding='same'),
            tf.keras.layers.Conv2D(96, 11, strides=4, padding='same'),
            tf.keras.layers.Lambda(tf.nn.local_response_normalization),
            tf.keras.layers.Activation('relu'),
            tf.keras.layers.MaxPooling2D(3, strides=2),
            tf.keras.layers.Conv2D(256, 5, strides=4, padding='same'),
            tf.keras.layers.Lambda(tf.nn.local_response_normalization),
            tf.keras.layers.Activation('relu'),
            tf.keras.layers.MaxPooling2D(3, strides=2),
            tf.keras.layers.Conv2D(384, 3, strides=4, padding='same'),
            tf.keras.layers.Activation('relu'),
            tf.keras.layers.Conv2D(384, 3, strides=4, padding='same'),
            tf.keras.layers.Activation('relu'),
            tf.keras.layers.Conv2D(256, 3, strides=4, padding='same'),
            tf.keras.layers.Activation('relu'),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(4096, activation='relu'),
            tf.keras.layers.Dropout(0.5),
            tf.keras.layers.Dense(4096, activation='relu'),
            tf.keras.layers.Dropout(0.5),
            tf.keras.layers.Dense(1, activation='softmax'),
        ])

        self.weights_and_biases = []
        self.get_weights_and_biases()

        self.features = None
        self.labels = None 
        self.nce_loss = None 

    def call(self):
        self.features = self.encoder(self.image_array)
    
    def get_weights_and_biases(self):
        for layer in self.encoder:
            if isinstance(layer, (tf.keras.layers.Conv2D, tf.keras.layers.Dense)):
                weights = layer.get_weights()[0]
                biases = layer.get_weights()[1]
                self.weights_and_biases.append((weights, biases))

    def train(self, epochs: int, learning_rate: float):
        # Optimizer
        optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

        # Training loop
        num_epochs = epochs if epochs else 100

        for epoch in range(num_epochs):
            for batch_frames in self.video_capture.get_batches(self.batch_size):
                for batch_idx, img_array in enumerate(batch_frames):
                    with tf.GradientTape() as tape:
                        # Forward pass
                        self.image_array = img_array
                        features = self.call()

                        num_features = self.encoder.layers[-1].output_shape[1]

                        # Create random labels for NCE loss
                        labels = tf.random.uniform(
                            shape=(self.batch_size, 1), 
                            minval=0, 
                            maxval=num_features, 
                            dtype=tf.int32
                        )

                        # Compute the NCE loss
                        loss = tf.nn.nce_loss(
                            weights=self.weights_and_biases[-1][0],
                            biases=self.weights_and_biases[-1][1],
                            labels=labels,
                            inputs=features,
                            num_sampled=self.batch_size,
                            num_classes=num_features,
                            num_true=1,
                            remove_accidental_hits=True
                        )

                        # Compute the mean loss
                        loss = tf.reduce_mean(loss)

                    # Backward pass and optimization
                    gradients = tape.gradient(loss, self.trainable_variables)
                    optimizer.apply_gradients(zip(gradients, self.trainable_variables))

                    # Print the loss for monitoring
                    if batch_idx % 10 == 0:
                        print(f"Epoch [{epoch+1}/{num_epochs}], Batch [{batch_idx+1}/{len(batch_frames)}], Loss: {loss.numpy():.4f}")