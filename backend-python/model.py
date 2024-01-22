import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler, LabelEncoder
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Flatten, Reshape
from keras.optimizers import Adam
from keras.losses import MeanSquaredError
from decouple import config
from mongoengine import connect
from pymongo.errors import ConnectionFailure
from datetime import timedelta
import matplotlib.pyplot as plt

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
    #print(last_month, "++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
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
    #print(monthly_totals_list)
    # Create sequences for training
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
    else:
        raise ValueError(
            "Invalid model type. Supported types are 'LSTM', 'ANN', and 'GAN'."
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
    # Assuming 'incomes_df' is your DataFrame containing incomes
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
        else:
            raise ValueError(
                "Invalid model type. Supported types are 'LSTM', 'ANN', and 'GAN'."
            )

        # Train the model
        # print(X_train_reshaped.shape, y_train_scaled.shape)
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


def getPrediction(choice="income", model="LSTM", months=5, lb=5):
    
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
        
        last_month = ""

        if choice == "income":
            categories = in_categories
            preds, last_month = build_income_prediction_model(df_income, model, int(months), int(lb))
            preds = transpose(preds)
        elif choice == "expense":
            categories = ex_categories
            preds, last_month = build_expense_prediction_model(df_expense, model, int(months), int(lb))
            preds = transpose(preds)
        elif choice == "both":
            preds, last_month_e = build_expense_prediction_model(df_expense, model, int(months), int(lb))
            preds_e = []
            for i in range(len(preds)):
                preds_e.append(sum(preds[i]))
            preds, last_month_i = build_income_prediction_model(df_income, model, int(months), int(lb))
            preds_i = []
            for i in range(len(preds)):
                preds_i.append(sum(preds[i]))
            #print(preds_e, preds_i, "+++++++++++++++++++++++++++++++++++++++++")
            preds = [preds_e, preds_i]
            #print(last_month_e, last_month_i)
            last_month = [last_month_e, last_month_i]#last_month_e if last_month_e < last_month_i else last_month_i

        """plt.style.use('dark_background')
        for i, sublist in enumerate(preds):
            label = categories[i]
            # Smooth the line
            x_smooth, y_smooth = smooth_line(range(len(sublist)), sublist)
            plt.plot(x_smooth, y_smooth, label=label, color=colors[i])
            # Display values for each point
            for j, value in enumerate(sublist):
                plt.text(j, value, f'{value:.2f}', ha='center', va='bottom', color=colors[i])
        # Add legend
        plt.legend()
        # Show the plot
        plt.show()"""
        
        if isinstance(last_month, list):
            for i in range(len(last_month)):
                month = str(last_month[i])[5:7]
                year = str(last_month[i])[:4]
                last_month[i] = month + "/" + year
        else:
            month = str(last_month)[5:7]
            year = str(last_month)[:4]
            last_month = month + "/" + year 
        
        print(preds, last_month)
        return preds, last_month
        #return {"prediction": preds}, 200
        #return preds

    except ValueError as e:
        return -1, -1
        #return {"prediction": -1}, 200
        #return -1
        #raise e

    except ConnectionFailure as e:
        print("DB Connection Error:", e)
        raise e


#getPrediction("expense", "LSTM", 1, 2)
