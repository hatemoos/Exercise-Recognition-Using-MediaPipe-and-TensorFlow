import pickle
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split

data_dict = pickle.load(open('data_lunges.pickle', 'rb'))
data = np.asarray(data_dict['data'])
labels = np.asarray(data_dict['labels'])

x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, shuffle=True, stratify=labels)

model = tf.keras.models.Sequential()
model.add(Dense(128, input_dim=66, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])


early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True)
history = model.fit(x_train, y_train, epochs=100, validation_data=(x_test, y_test), callbacks=[early_stopping])
loss, accuracy = model.evaluate(x_test, y_test, verbose=2)

loss, accuracy = model.evaluate(x_test, y_test)
print(f'Accuracy: {accuracy * 100:.2f}%')

f = open('model_lunges.p', 'wb')
pickle.dump({'model': model}, f)
f.close()
