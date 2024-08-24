import os

import tensorflow as tf

new_model = tf.keras.models.load_model(os.path.join('model', 'my_model.tf'))
new_model.summary()
