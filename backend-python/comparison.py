from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Bidirectional, LSTM, Dense, Dropout, Conv1D, Flatten, Reshape, GRU, MultiHeadAttention, Input, Permute
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, accuracy_score, confusion_matrix, classification_report
import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler, LabelEncoder
from keras.optimizers import Adam
from keras.losses import MeanSquaredError
import pickle
import random
from statistics import mean, median

def lstm(n_steps, n_features):
    model = Sequential()
    model.add(LSTM(50, activation="relu", input_shape=(n_steps, n_features)))
    model.add(Dropout(0.3))
    model.add(Dense(50, activation="relu"))
    model.add(Dense(25, activation="relu"))
    model.add(Dense(50, activation="relu"))
    model.add(Dense(25, activation="relu"))
    model.add(
        Dense(n_features, activation="relu")
    )  # Linear activation for regression tasks
    custom_optimizer = Adam(learning_rate=0.001)
    model.compile(
        optimizer=custom_optimizer, loss=MeanSquaredError(), metrics=["mse", "accuracy"]
    )
    return model

def bilstm(n_steps, n_features):
    model = Sequential()
    model.add(Bidirectional(LSTM(50, activation="relu"), input_shape=(n_steps, n_features)))
    model.add(Dropout(0.3))
    model.add(Dense(50, activation="relu"))
    model.add(Dense(25, activation="relu"))
    model.add(
        Dense(n_features, activation="relu")
    )  # Linear activation for regression tasks
    custom_optimizer = Adam(learning_rate=0.001)
    model.compile(
        optimizer=custom_optimizer, loss=MeanSquaredError(), metrics=["mse", "accuracy"]
    )
    return model


def ann(n_steps, n_features):
    model = Sequential()
    model.add(Flatten(input_shape=(n_steps, n_features)))
    model.add(Dense(50, activation="relu"))
    model.add(Dropout(0.3))
    model.add(Dense(50, activation="relu"))
    model.add(Dense(25, activation="relu"))
    model.add(Dense(50, activation="relu"))
    model.add(Dense(25, activation="relu"))
    model.add(Dense(n_features, activation="linear"))
    # model.add(Dense(n_features))  # Adjust the number of units to match the number of features
    custom_optimizer = Adam(learning_rate=0.001)
    model.compile(
        optimizer=custom_optimizer, loss=MeanSquaredError(), metrics=["mse", "accuracy"]
    )
    return model


def gan(n_steps, n_features):
    latent_dim = 100  # Adjust the latent dimension as needed
    generator = Sequential()
    generator.add(Dense(128, activation="relu", input_shape=(n_steps, n_features)))
    generator.add(Dropout(0.3))
    generator.add(Dense(50, activation="relu"))
    generator.add(Dense(25, activation="relu"))
    generator.add(Dense(n_features, activation="relu"))
    generator.add(Reshape((n_steps, n_features)))

    discriminator = Sequential()
    discriminator.add(Flatten(input_shape=(n_steps, n_features)))
    discriminator.add(Dense(128, activation="relu"))
    discriminator.add(Dropout(0.3))
    discriminator.add(Dense(50, activation="relu"))
    discriminator.add(Dense(25, activation="relu"))
    discriminator.add(Dense(n_features, activation="relu"))

    discriminator.trainable = False
    gan_model = Sequential()
    gan_model.add(generator)
    gan_model.add(discriminator)

    custom_optimizer = Adam(learning_rate=0.001, beta_1=0.5)
    gan_model.compile(
        optimizer=custom_optimizer,
        loss="binary_crossentropy",
        metrics=["mse", "accuracy"],
    )

    return gan_model

def cnn(n_steps, n_features):
    model = Sequential()
    model.add(Conv1D(64, kernel_size=3, activation="relu", input_shape=(n_steps, n_features)))
    model.add(Conv1D(50, kernel_size=3, activation="relu"))
    model.add(Conv1D(25, kernel_size=3, activation="relu"))
    model.add(Flatten())
    model.add(Dense(50, activation="relu"))
    model.add(Dropout(0.3))
    model.add(Dense(50, activation="relu"))
    model.add(Dense(25, activation="relu"))
    model.add(Dense(n_features, activation="linear"))
    custom_optimizer = Adam(learning_rate=0.001)
    model.compile(
        optimizer=custom_optimizer, loss=MeanSquaredError(), metrics=["mse", "accuracy"]
    )
    return model

def gru(n_steps, n_features):
    model = Sequential()
    model.add(GRU(50, activation="relu", input_shape=(n_steps, n_features)))
    model.add(Dense(50, activation="relu"))
    model.add(Dropout(0.3))
    model.add(Dense(50, activation="relu"))
    model.add(Dense(n_features, activation="linear"))
    custom_optimizer = Adam(learning_rate=0.001)
    model.compile(
        optimizer=custom_optimizer, loss=MeanSquaredError(), metrics=["mse", "accuracy"]
    )
    return model

def create_sequences(data, n_steps):
    X, y = [], []
    for i in range(len(data)):
        end_ix = i + n_steps
        if end_ix > len(data) - 1:
            break
        seq_x, seq_y = data[i:end_ix], data[end_ix]
        X.append(seq_x)
        y.append(seq_y)
    return np.array(X), np.array(y)

def transpose(l1):
    l2 = []
    # iterate over list l1 to the length of an item
    for i in range(len(l1[0])):
        # print(i)
        row = []
        for item in l1:
            # appending to new list with values and index positions
            # i contains index position and item contains values
            row.append(item[i])
        l2.append(row)
    return l2

n_steps = 12
n_features = 8
num_models = 10

def generate_sine_curve_x_values(num_points=100, peak_value=10000):
    # Create a sine curve ranging from -pi/2 to pi/2
    x_values_sine = np.linspace(-np.pi / 2, np.pi / 2, num_points)
    # Scale and shift the sine curve to fit the desired range (0 to 10000)
    x_values_scaled = (np.sin(x_values_sine) + 1) * (peak_value / 2)
    # Convert to floats with max 2 decimal places
    x_values_float = np.round(x_values_scaled, 2)
    return [x + (peak_value / 2) + random.uniform(-peak_value, peak_value) for x in x_values_float]

def generate_cos_curve_x_values(num_points=100, peak_value=10000):
    # Create a sine curve ranging from -pi/2 to pi/2
    x_values_sine = np.linspace(-np.pi / 2, np.pi / 2, num_points)
    # Scale and shift the sine curve to fit the desired range (0 to 10000)
    x_values_scaled = (np.cos(x_values_sine) + 1) * (peak_value / 2)
    # Convert to floats with max 2 decimal places
    x_values_float = np.round(x_values_scaled, 2)
    return [x + (peak_value / 2) + random.uniform(-peak_value, peak_value) for x in x_values_float]

n_estimators = 3  # Choose the number of estimators as needed
dropout_rate = 0.3  # Choose the dropout rate as needed


mse_l = [[], [], [], [], [], []]
loss_l = [[], [], [], [], [], []]
for j in range(1, 101):
    print("(((((((((((((((((((((((((((((((((((())))))))))))))))))))))))))))))))))))")
    print("ITERATION", j)
    print("(((((((((((((((((((((((((((((((((((())))))))))))))))))))))))))))))))))))")
    # Example usage:
    backward = np.linspace(10000, 0, 10000)
    forward = np.linspace(0, 10000, 10000)
    data1 = generate_sine_curve_x_values(num_points=10000, peak_value=10000)
    data2 = generate_cos_curve_x_values(num_points=10000, peak_value=10000)
    data3 = [x + (10000 / 2) + random.uniform(-10000, 10000) for x in backward]
    data4 = [x + (10000 / 2) + random.uniform(-10000, 10000) for x in forward]
    data5 = generate_sine_curve_x_values(num_points=10000, peak_value=10000)
    data6 = generate_cos_curve_x_values(num_points=10000, peak_value=10000)
    data7 = [x + (10000 / 2) + random.uniform(-10000, 10000) for x in backward]
    data8 = [x + (10000 / 2) + random.uniform(-10000, 10000) for x in forward]
    for i in range(10000):
        if data1[i] < 0:
            data1[i] = 0
        if data2[i] < 0:
            data2[i] = 0
        if data3[i] < 0:
            data3[i] = 0
        if data4[i] < 0:
            data4[i] = 0
        if data5[i] < 0:
            data5[i] = 0
        if data6[i] < 0:
            data6[i] = 0
        if data7[i] < 0:
            data7[i] = 0
        if data8[i] < 0:
            data8[i] = 0

    preds = []
    for i in range(10000):
        preds.append([data1[i], data2[i], data3[i], data4[i], data5[i], data6[i], data7[i], data8[i]])
    '''
    with open("fake_data.pickle", 'rb') as file:
        preds = pickle.load(file)
    '''
    X_train, y_train = create_sequences(preds[:8000], n_steps)
    scaler = MinMaxScaler(feature_range=(0, 1))
    X_train_scaled = scaler.fit_transform(X_train.reshape(-1, 1)).reshape(
        X_train.shape
    )
    y_train_scaled = scaler.transform(y_train.reshape(-1, 1)).reshape(y_train.shape)
    # Assuming n_features is the number of categories
    n_features = 8
    # Reshape data for LSTM input (samples, timesteps, features)
    X_train_reshaped = X_train_scaled.reshape(
        (X_train_scaled.shape[0], n_steps, n_features)
    )
    # Ensure that the reshaping is compatible with the LSTM input
    # The second dimension (n_steps) should match the actual number of past months used for prediction
    assert X_train_reshaped.shape[1] == n_steps, "Incorrect reshaping dimensions"

    X_test, y_test = create_sequences(preds[8000:], n_steps)
    scaler = MinMaxScaler(feature_range=(0, 1))
    X_test_scaled = scaler.fit_transform(X_test.reshape(-1, 1)).reshape(
        X_test.shape
    )
    y_test_scaled = scaler.transform(y_test.reshape(-1, 1)).reshape(y_test.shape)
    # Reshape data for LSTM input (samples, timesteps, features)
    X_test_reshaped = X_test_scaled.reshape(
        (X_test_scaled.shape[0], n_steps, n_features)
    )
    # Ensure that the reshaping is compatible with the LSTM input
    # The second dimension (n_steps) should match the actual number of past months used for prediction
    assert X_test_reshaped.shape[1] == n_steps, "Incorrect reshaping dimensions"
    
    lstm_model = lstm(n_steps, n_features)
    ann_model = ann(n_steps, n_features)
    gan_model = gan(n_steps, n_features)
    bilstm_model = bilstm(n_steps, n_features)
    gru_model = gru(n_steps, n_features)
    cnn_model = cnn(n_steps, n_features)

    models = [lstm_model, ann_model, gan_model, bilstm_model, gru_model, cnn_model]
    model_names = ["LSTM", "ANN", "GAN", "BiLSTM", "GRU", "CNN"]
    # Fit each model to the training data
    for model, model_name in zip(models, model_names):
        print(f"Training {model_name}...")
        model.fit(X_train_reshaped, y_train_scaled, epochs=5, batch_size=120, verbose=1)
    # Evaluate each model on the testing data
    model_scores = []
    for model, model_name in zip(models, model_names):
        #print(f"Evaluating {model_name}...")
        loss, mse, _ = model.evaluate(X_test_reshaped, y_test_scaled, verbose=0)
        model_scores.append({"model": model_name, "mse": mse, "loss": loss})
    # Rank models based on mean squared error (lower is better)
    #model_scores.sort(key=lambda x: x["mse"])
    # Display the ranking
    #print("\nModel Ranking:")
    m, l = [], []
    for i, model_score in enumerate(model_scores, start=0):
        #print(f"{i}. {model_score['model']}: MSE = {model_score['mse']}, Loss = {model_score['loss']}")
        mse_l[i].append(model_score['mse'])
        loss_l[i].append(model_score['loss'])

for i in range(6):
    mse_l[i] = [mean(mse_l[i]), median(mse_l[i])]
    loss_l[i] = [mean(loss_l[i]), median(loss_l[i])]
        
for i in range(6):
    print(model_names[i] + ":")
    print("MSE Mean =", mse_l[i][0], ", MSE Median =", mse_l[i][1], ", LOSS Mean =", loss_l[i][0], ", LOSS Median =", loss_l[i][1])
    print("=====================================================================================================================")
    #pass

with open("comaprison.pickle", 'wb') as file:
    pickle.dump([model_names[i], mse_l[i], loss_l[i]], file)