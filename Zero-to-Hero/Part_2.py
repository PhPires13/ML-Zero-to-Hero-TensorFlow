import keras
import tensorflow as tf

fashion_mnist = keras.datasets.fashion_mnist
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

model = keras.Sequential([
    keras.layers.Flatten(input_shape=(28, 28)),
    keras.layers.Dense(128, activation=tf.nn.relu),
    keras.layers.Dense(10, activation=tf.nn.softmax)
])

model.compile(optimizer=tf.optimizers.Adam(),  # tf.train.AdamOptimizer() -> tf.optimizers.Adam()
              loss='sparse_categorical_crossentropy')

model.fit(train_images, train_labels, epochs=5)

test_loss = model.evaluate(test_images, test_labels)  # It does not return test_acc
print(f'Loss: {test_loss}')

# To get predictions for new images
# predictions = model.predict(my_images)
