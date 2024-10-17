from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,Dropout
from tensorflow.keras import optimizers
from tensorflow.keras.utils import plot_model
from tensorflow.keras.utils import to_categorical
import numpy as np
import matplotlib.pyplot as plt

(X_train, y_train), (X_test, y_test) = mnist.load_data()

X_train = X_train.reshape(X_train.shape[0], 784)
X_test = X_test.reshape(X_test.shape[0], 784)
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

model = Sequential()

model.add(Dense(256, activation='sigmoid', input_dim=784))
model.add(Dropout(rate=0.2))
model.add(Dense(128, activation='relu'))
model.add(Dropout(rate=0.2))
model.add(Dense(10, activation='softmax'))

sgd = optimizers.SGD(learning_rate=0.04)

model.compile(optimizer='rmsprop',loss='binary_crossentropy',metrics=['accuracy'])

model.summary()

history = model.fit(X_train, y_train, verbose=1, batch_size=64,epochs=10)

score = model.evaluate(X_test, y_test,verbose=0)
print("evaluate loss: {0[0]}\nevaluate acc: {0[1]}".format(score))

plt.plot(history.history["accuracy"], label="accuracy")
plt.ylabel("accuracy")
plt.xlabel("epoch")
plt.legend(loc="best")
plt.show()

pred = np.argmax(model.predict(X_test[0:20]), axis=1)
print(pred)

for i in range(20):
    plt.subplot(1, 20, i+1)
    plt.imshow(X_test[i].reshape((28,28)), "gray")
plt.show()
