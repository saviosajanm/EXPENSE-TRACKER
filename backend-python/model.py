import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler, LabelEncoder
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Bidirectional, LSTM, Dense, Dropout, Conv1D, Flatten, Reshape, GRU
from keras.optimizers import Adam
from keras.losses import MeanSquaredError
from decouple import config
from mongoengine import connect
from pymongo.errors import ConnectionFailure
import pickle

from controllers.expense import getExpense
from controllers.income import getIncomes

class InvalidLBException(Exception):
    "Raised when look back is too big"
    pass


def smooth_line(x, y, factor=10):
    x_smooth = np.linspace(0, len(x) - 1, len(x) * factor)
    y_smooth = np.interp(x_smooth, range(len(x)), y)
    return x_smooth, y_smooth


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


def prepare_input_data(data):
    # Assuming data is a DataFrame with columns: 'date', 'amount', 'type'

    # Convert 'date' column to a numerical feature, e.g., days since the start date
    data["days_since_start"] = (data["date"] - data["date"].min()).dt.days

    # Convert categorical feature 'type' to numerical using Label Encoding
    label_encoder = LabelEncoder()
    data["type_encoded"] = label_encoder.fit_transform(data["type"])

    # Standardize numerical features using Standard Scaler
    scaler = StandardScaler()
    data[["days_since_start", "amount"]] = scaler.fit_transform(
        data[["days_since_start", "amount"]]
    )

    # Select relevant features for input
    input_features = data[["days_since_start", "amount", "type_encoded"]]

    return input_features


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


def ann(n_steps, n_features):
    model = Sequential()
    model.add(Flatten(input_shape=(n_steps, n_features)))
    model.add(Dense(50, activation="relu"))
    model.add(Dropout(0.3))
    model.add(Dense(50, activation="relu"))
    model.add(Dense(25, activation="relu"))
    model.add(Dense(50, activation="relu"))
    model.add(Dense(25, activation="relu"))
    model.add(Dense(8, activation="linear"))
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


def build_expense_prediction_model(expenses_df, md="LSTM", X=5, n_steps=3):
    # Assuming 'expenses_df' is your DataFrame containing expenses
    expenses_df["date"] = pd.to_datetime(
        expenses_df["date"]
    )  # Convert 'date' column to datetime
    categories = [
        "education",
        "groceries",
        "health",
        "subscriptions",
        "takeaways",
        "clothing",
        "travelling",
        "other",
    ]

    # Create a DataFrame with all months in the dataset
    all_months = pd.date_range(
        start=expenses_df["date"].min().replace(day=1),
        end=expenses_df["date"].max().replace(day=1) + pd.DateOffset(months=1),
        freq="M",
    )
    last_month = all_months[-1]
    # Initialize an empty list to store monthly totals
    monthly_totals_list = []

    # Iterate over months
    for month in all_months:
        # Filter data for the current month
        month_data = expenses_df[expenses_df["date"].dt.month == month.month]

        # If data exists for the month
        if not month_data.empty:
            # Group by 'category' and sum 'amount_scaled'
            month_totals = (
                month_data.groupby("category")["amount"]
                .sum()
                .reindex(categories, fill_value=0)
                .tolist()
            )
        else:
            # If no data for the month, create an array of zeros
            month_totals = [0] * len(categories)

        # Add the monthly totals to the list
        monthly_totals_list.append(month_totals)

    # Display the resulting list
    for i, month_totals in zip(all_months.strftime("%B"), monthly_totals_list):
        print(i, ":", month_totals)

    data = np.array(monthly_totals_list)
    if len(data) <= 12:
        n_steps = 1
    elif len(data <= 120):
        n_steps = 12
    else:
        n_steps = 12
    # Create sequences for training
    if len(data) == 1:
        return -1
    else:
        X_train, y_train = create_sequences(data, n_steps)

        # Normalize data
        scaler = MinMaxScaler(feature_range=(0, 1))
        X_train_scaled = scaler.fit_transform(X_train.reshape(-1, 1)).reshape(X_train.shape)
        y_train_scaled = scaler.transform(y_train.reshape(-1, 1)).reshape(y_train.shape)

        # Assuming n_features is the number of categories
        n_features = len(categories)
        # Reshape data for LSTM input (samples, timesteps, features)
        X_train_reshaped = X_train_scaled.reshape(
            (X_train_scaled.shape[0], n_steps, n_features)
        )

        # Ensure that the reshaping is compatible with the LSTM input
        # The second dimension (n_steps) should match the actual number of past months used for prediction
        assert X_train_reshaped.shape[1] == n_steps, "Incorrect reshaping dimensions"

        # model.compile(optimizer='adam', loss='mse', metrics=['accuracy'])

        if md == "LSTM":
            model = lstm(n_steps, n_features)
        elif md == "ANN":
            # y_train_scaled = y_train_scaled.reshape(-1, 3)
            model = ann(n_steps, n_features)
        elif md == "GAN":
            model = gan(n_steps, n_features)
        elif md == "CNN":
            model = cnn(n_steps, n_features)
        elif md == "GRU":
            model = gru(n_steps, n_features)
        elif md == "BILSTM":
            model = bilstm(n_steps, n_features)
        else:
            raise ValueError(
                "Invalid model type."
            )

        # Train the model
        model.fit(X_train_reshaped, y_train_scaled, epochs=5, batch_size=10)

        # Assuming future_sequence is a 2D numpy array
        # Initialize future_sequence with the first prediction
        future_sequence = X_train_reshaped[-1].reshape(1, n_steps, n_features)
        # print(future_sequence.shape)
        preds = []
        # Inside your loop for generating predictions
        for _ in range(X):
            # Make the prediction
            prediction = model.predict(future_sequence)
            preds.append(
                scaler.inverse_transform(prediction.reshape(-1, 1)).reshape(
                    prediction.shape
                )[0]
            )
            # Ensure the prediction is a 1D array or a scalar value

            # Concatenate the prediction to future_sequence
            future_sequence = future_sequence.tolist()
            future_sequence[0].pop(0)
            future_sequence[0].append(prediction.tolist()[0])
            future_sequence = np.array(future_sequence)
            # X_train_reshaped = np.concatenate(X_train_reshaped, future_sequence)
            X_train_reshaped = X_train_reshaped + future_sequence
            #print(X_train_reshaped.shape, "+++++++++++++++++++++++++")
            model.fit(X_train_reshaped, y_train_scaled, epochs=5, batch_size=10)

        return preds, last_month


def build_income_prediction_model(incomes_df, md="LSTM", X=5, n_steps=5):
    
    incomes_df["date"] = pd.to_datetime(
        incomes_df["date"]
    )  # Convert 'date' column to datetime
    categories = [
        "salary",
        "freelancing",
        "investments",
        "stocks",
        "bitcoin",
        "bank",
        "youtube",
        "other",
    ]
    # Create a DataFrame with all months in the dataset
    all_months = pd.date_range(
        start=incomes_df["date"].min().replace(day=1),
        end=incomes_df["date"].max().replace(day=1) + pd.DateOffset(months=1),
        freq="M",
    )
    last_month = all_months[-1]
    #print(last_month, "++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
    # Initialize an empty list to store monthly totals
    monthly_totals_list = []

    # Iterate over months
    for month in all_months:
        # Filter data for the current month
        month_data = incomes_df[incomes_df["date"].dt.month == month.month]

        # If data exists for the month
        if not month_data.empty:
            # Group by 'category' and sum 'amount_scaled'
            month_totals = (
                month_data.groupby("category")["amount"]
                .sum()
                .reindex(categories, fill_value=0)
                .tolist()
            )
        else:
            # If no data for the month, create an array of zeros
            month_totals = [0] * len(categories)

        # Add the monthly totals to the list
        monthly_totals_list.append(month_totals)

    # Display the resulting list
    for i, month_totals in zip(all_months.strftime("%B"), monthly_totals_list):
        print(i, ":", month_totals)

    data = np.array(monthly_totals_list)
    if len(data) <= 12:
        n_steps = 1
    elif len(data <= 120):
        n_steps = 12
    else:
        n_steps = 12
    #print(monthly_totals_list)

    # Create sequences for training
    #print(len(data))
    if len(data) == 1:
        return -1
    else:
        X_train, y_train = create_sequences(data, n_steps)
        #print(X_train)
        # Normalize data
        scaler = MinMaxScaler(feature_range=(0, 1))
        X_train_scaled = scaler.fit_transform(X_train.reshape(-1, 1)).reshape(
            X_train.shape
        )
        y_train_scaled = scaler.transform(y_train.reshape(-1, 1)).reshape(y_train.shape)

        # Assuming n_features is the number of categories
        n_features = len(categories)
        # Reshape data for LSTM input (samples, timesteps, features)
        X_train_reshaped = X_train_scaled.reshape(
            (X_train_scaled.shape[0], n_steps, n_features)
        )

        # Ensure that the reshaping is compatible with the LSTM input
        # The second dimension (n_steps) should match the actual number of past months used for prediction
        assert X_train_reshaped.shape[1] == n_steps, "Incorrect reshaping dimensions"

        # model.compile(optimizer='adam', loss='mse', metrics=['accuracy'])

        if md == "LSTM":
            model = lstm(n_steps, n_features)
        elif md == "ANN":
            # y_train_scaled = y_train_scaled.reshape(-1, 3)
            model = ann(n_steps, n_features)
        elif md == "GAN":
            model = gan(n_steps, n_features)
        elif md == "CNN":
            model = cnn(n_steps, n_features)
        elif md == "GRU":
            model = gru(n_steps, n_features)
        elif md == "BILSTM":
            model = bilstm(n_steps, n_features)
        else:
            raise ValueError(
                "Invalid model type."
            )

        # Train the model
        # print(X_train_reshaped.shape, y_train_scaled.shape)
        #print(n_steps, n_features, ")))()()(()()()()()()()(((((()))))))")
        #print(X_train_reshaped, "000000000000000000000000000000000000000000000000000000000", y_train_scaled, "|||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||")
        model.fit(X_train_reshaped, y_train_scaled, epochs=5, batch_size=10)

        # Assuming future_sequence is a 2D numpy array
        # Initialize future_sequence with the first prediction
        future_sequence = X_train_reshaped[-1].reshape(1, n_steps, n_features)
        preds = []
        # Inside your loop for generating predictions
        for _ in range(X):
            # Make the prediction
            prediction = model.predict(future_sequence)
            preds.append(
                scaler.inverse_transform(prediction.reshape(-1, 1)).reshape(
                    prediction.shape
                )[0]
            )
            # Ensure the prediction is a 1D array or a scalar value

            # Concatenate the prediction to future_sequence
            future_sequence = future_sequence.tolist()
            future_sequence[0].pop(0)
            future_sequence[0].append(prediction.tolist()[0])
            future_sequence = np.array(future_sequence)
            # X_train_reshaped = np.concatenate(X_train_reshaped, future_sequence)
            X_train_reshaped = X_train_reshaped + future_sequence
            model.fit(X_train_reshaped, y_train_scaled, epochs=5, batch_size=10)

        return preds, last_month

def lm_convert(last_month):
    if isinstance(last_month, list):
        for i in range(len(last_month)):
            month = str(last_month[i])[5:7]
            year = str(last_month[i])[:4]
            last_month[i] = month + "/" + year
    else:
        month = str(last_month)[5:7]
        year = str(last_month)[:4]
        last_month = month + "/" + year
    return last_month

def getPrediction(choice="income", model="LSTM", months=12, lb=1, ifTrain = 'True'):
    print(choice, model, months, lb, ifTrain)
    
    if isinstance(months, int):
        months = int(months)
    if isinstance(lb, int):
        lb = int(lb)
    #print(months, lb)
    #pd.set_option("display.max_columns", None)
    #pd.set_option("display.max_rows", None)
    ex_categories = [
        "Education",
        "Groceries",
        "Health",
        "Subscriptions",
        "Takeaways",
        "Clothing",
        "Travelling",
        "Other",
    ]
    in_categories = [
        "Salary",
        "Freelancing",
        "Investments",
        "Stocks",
        "Bitcoin",
        "Bank Transfer",
        "Youtube",
        "Other",
    ]
    colors = [
        "#FF5733",
        "#33FF57",
        "#5733FF",
        "#FF33A1",
        "#33A1FF",
        "#A1FF33",
        "#FFD700",
        "#8A2BE2",
    ]
    
    

    try:
        mongo_url = config("MONGO_URL")
        connect(alias="default", host=mongo_url)
        print("DB Connection Successful")

        df_expense = pd.DataFrame(getExpense())
        df_income = pd.DataFrame(getIncomes())
        #df_combined = pd.concat([df_income, df_expense])
        
        if int(months) > 12:
            return -1, -1
        
        last_month = ""
        if ifTrain == "False":
            if choice == "income":
                preds, last_month = build_income_prediction_model(df_income, model, 12, int(lb))
                last_month = lm_convert(last_month)
                with open("income.pickle", 'wb') as file:
                    pickle.dump([preds, last_month], file)
                preds = transpose(preds[:int(months)])
            elif choice == "expense":
                preds, last_month = build_expense_prediction_model(df_expense, model, 12, int(lb))
                last_month = lm_convert(last_month)
                with open("expense.pickle", 'wb') as file:
                    pickle.dump([preds, last_month], file)
                preds = transpose(preds[:int(months)])
            elif choice == "both":
                preds_ex, last_month_e = build_expense_prediction_model(df_expense, model, 12, int(lb))
                preds_e = []
                for i in range(len(preds_ex)):
                    preds_e.append(sum(preds_ex[i]))
                preds_in, last_month_i = build_income_prediction_model(df_income, model, 12, int(lb))
                preds_i = []
                for i in range(len(preds_in)):
                    preds_i.append(sum(preds_in[i]))
                preds = [preds_e[:int(months)], preds_i[:int(months)]]
                last_month = [last_month_e, last_month_i]
                last_month = lm_convert(last_month)
            
        else:
            if choice == "income":
                with open("income.pickle", 'rb') as file:
                    loaded_data = pickle.load(file)
                    preds = transpose(loaded_data[0])
                    last_month = loaded_data[1]
            elif choice == "expense":
                with open("expense.pickle", 'rb') as file:
                    loaded_data = pickle.load(file)
                    preds = transpose(loaded_data[0])
                    last_month = loaded_data[1]
            elif choice == "both":
                preds_ex, last_month_e = build_expense_prediction_model(df_expense, model, 12, int(lb))
                preds_e = []
                for i in range(len(preds_ex)):
                    preds_e.append(sum(preds_ex[i]))
                preds_in, last_month_i = build_income_prediction_model(df_income, model, 12, int(lb))
                preds_i = []
                for i in range(len(preds_in)):
                    preds_i.append(sum(preds_in[i]))
                preds = [preds_e[:int(months)], preds_i[:int(months)]]
                last_month = [last_month_e, last_month_i]
                last_month = lm_convert(last_month)
                    
        print(preds, last_month)    
        return preds, last_month

    except ValueError as e:
        return -1, -1
        #return {"prediction": -1}, 200
        #return -1
        #raise e

    except ConnectionFailure as e:
        print("DB Connection Error:", e)
        raise e


#getPrediction("expense", "LSTM", 1, 2)
