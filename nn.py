import tensorflow as tf
import tensorflow
import numpy as np
import matplotlib.pyplot as plt


(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

x_train, x_test = x_train / 255.0, x_test / 255.0


model = keras.Sequential([
    keras.layers.Flatten(input_shape=(28, 28)), 
    keras.layers.Dense(128, activation='relu'), 
    keras.layers.Dense(10, activation='softmax')
])


model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])


history = model.fit(x_train, y_train, epochs=5, validation_data=(x_test, y_test))


test_loss, test_acc = model.evaluate(x_test, y_test)


predictions = model.predict(np.expand_dims(x_test[0], axis=0))

plt.imshow(x_test[0], cmap=plt.cm.binary)
plt.title(f"Prediction: {np.argmax(predictions)}")
plt.show()


test_acc
