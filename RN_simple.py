import csv
from tqdm import tqdm
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

epochs = 50
batch_size = 250


data = pd.read_csv("201276.csv", delimiter=";")


X = data[['id', 'X1', 'X2', 'X3']].values
Y = data['Y'].values.reshape(-1, 1)


scaler = StandardScaler()
X = scaler.fit_transform(X)


X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.1, random_state=42)


model = tf.keras.models.Sequential([
    tf.keras.layers.Dense(64, activation='relu', input_shape=(4,), kernel_initializer='glorot_uniform'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(64, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.001)),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(1)
])


model.compile(loss='mean_squared_error', optimizer=tf.keras.optimizers.Adam(learning_rate=0.05), metrics=['mean_absolute_error'])


early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10)


history = model.fit(X_train, Y_train, epochs=epochs, batch_size=batch_size, validation_data=(X_test, Y_test), callbacks=[early_stopping])
model.fit(X_train, Y_train, epochs=epochs)


mse, mae = model.evaluate(X_test, Y_test)
print("MSE on testing set: ", mse)
print("MAE on testing set: ", mae)

X_new = np.array([[1, 30.28, 937.08, 745.98]])
X_new = scaler.transform(X_new)
y_pred = model.predict(X_new)
print("Prediction: ", y_pred)

model.save('modelo.h5')


tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir='./logs')
tf.keras.utils.plot_model(model, to_file='modelo.png', show_shapes=True, show_layer_names=True, rankdir='TB', dpi=96, expand_nested=True, layer_range=None)

history = model.fit(X_train, Y_train, epochs=epochs, batch_size=batch_size, validation_data=(X_test, Y_test), callbacks=[early_stopping])

fig, ax = plt.subplots()


ax.plot([], [], 'b', label='Training loss')
ax.plot([], [], 'r', label='Validation loss')
ax.legend()

ax.set_title('Model error')
ax.set_ylabel('Error')
ax.set_xlabel('Epoch')


epochs_range = range(epochs)
with tqdm(total=epochs, desc='Training') as pbar:
    for epoch in epochs_range:
       
        model.fit(X_train, Y_train, batch_size=batch_size, epochs=epochs, verbose=0)

       
        train_loss = model.evaluate(X_train, Y_train, verbose=0)[0]
        val_loss = model.evaluate(X_test, Y_test, verbose=0)[0]

        
        ax.plot(epoch, train_loss, 'bo')
        ax.plot(epoch, val_loss, 'ro')

        
        pbar.update(1)

model.fit(X_train, Y_train, epochs=epochs)
weights_history = model.get_weights()


fig, ax = plt.subplots(figsize=(10, 6))


ax.set_title('Evolución de los pesos')
ax.set_xlabel('Epocas')
ax.set_ylabel('Pesos')


for i, layer_weights in enumerate(weights_history):
    for j, weight in enumerate(layer_weights.flatten()):
        ax.plot(i, weight, 'o', label='Capa {} Peso {}'.format(i+1, j+1))


ax.legend()
plt.show()



y_pred = model.predict(X_test)


fig, axs = plt.subplots(2, sharex=True, figsize=(10, 6))

axs[0].plot(Y_test)
axs[0].set_ylabel('Salida deseada')


axs[1].plot(y_pred)
axs[1].set_ylabel('Salida calculada')
axs[1].set_xlabel('Identificador de la muestra')


fig.suptitle('Salida deseada vs. Salida calculada')

plt.show()

y_pred = model.predict(X_test)

errors = Y_test - y_pred

plt.scatter(range(len(errors)), errors)
plt.title('Identificador de la muestra vs Error')
plt.xlabel('Identificador de la muestra')
plt.ylabel('Error')
plt.show()


def generate_report(model, X_train, Y_train, epochs, max_error, file_name):
  
    initial_weights = model.get_weights()


    history = model.fit(X_train, Y_train, epochs=epochs, verbose=0)
    final_weights = model.get_weights()
    observed_error = history.history['loss'][-1]

    permisible_error = max_error / Y_train.max()

    with open(file_name, mode='w', newline='') as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(['Pesos iniciales'] + list(initial_weights))
        writer.writerow(['Pesos finales'] + list(final_weights))
        writer.writerow(['Error permisible', permisible_error])
        writer.writerow(['Error observado', observed_error])
        writer.writerow(['Cantidad de épocas de entrenamiento', len(history.history['loss'])])
        writer.writerow(['Maximo error observado', max(history.history['loss'])])

generate_report(model, X_train, Y_train, epochs=epochs, max_error=0.1, file_name='reporte.csv')


