import yfinance as yf
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
from sklearn.metrics import precision_score


CONFIDENCE_TRESHHOLD = 0.6

def predict(train,test,predictors,model):
    model.fit(train[predictors], train["Target"])
    preds = model.predict_proba(test[predictors])[:,1]
    preds[preds > CONFIDENCE_TRESHHOLD] =1
    preds[preds <= CONFIDENCE_TRESHHOLD] =0
    preds = pd.Series(preds,index=test.index,name = "Predictions")
    combined = pd.concat([test["Target"],preds],axis=1)
    return combined

def backtest(data,model, predictors, start =1250, step =25):
    all_predictions =[]
    for i in range(start,data.shape[0],step):
        train =data.iloc[0:i].copy()
        test =data.iloc[i:(i+step)].copy()
        predictions = predict(train,test,predictors,model)
        all_predictions.append(predictions)

    return pd.concat(all_predictions)


def retrieve_data(ticker):
    stock_history = yf.Ticker(ticker)
    stock_history = stock_history.history(period= "max")
    del stock_history["Dividends"]
    del stock_history["Stock Splits"]
    stock_history["Tomorrow"]= stock_history["Close"].shift(-1)
    stock_history["Target"] = (stock_history["Tomorrow"] > stock_history["Close"]).astype(int)
    stock_history = stock_history.loc["1990-01-01":].copy()
    return stock_history

def Get_Model():
    return RandomForestClassifier(n_estimators=200, min_samples_split=50,random_state=1)


def process_data(model,stock_history):
    horizons = [2,5,60,250,1000]
    new_predictors = []

    for horizon in horizons:
        rolling_avg = stock_history.rolling(horizon).mean()
        ratio_column = f"Close_Ration_{horizon}"
        stock_history[ratio_column] = stock_history["Close"]/rolling_avg["Close"]
        trend_column = f"Trend_{horizon}"
        stock_history[trend_column] = stock_history.shift(1).rolling(horizon).sum()["Target"]
        new_predictors += [ratio_column,trend_column]

    predictions = backtest(stock_history,model,new_predictors)
    stock_history = stock_history.dropna(subset = stock_history.columns[stock_history.columns != "Tomorrow"])

    print(stock_history)
    accuracy = precision_score(predictions["Target"], predictions["Predictions"])
    print("Accuracy Rating:", accuracy)

    prediction_tomorrow = predictions.iloc[-1]["Predictions"]
    if prediction_tomorrow == 1:
        print("Prediction for tomorrow: Stock will rise.")
    else:
        print("Prediction for tomorrow: Stock will not rise.")



tickers_input = input("Enter tickers separated by commas: ")
tickers = tickers_input.split(',')
model = Get_Model()

for ticker in tickers:
    try:
        stock_data = retrieve_data(ticker.strip())  # Remove leading/trailing whitespaces
        process_data(model, stock_data)
    except ValueError:
        print(f"Error: Ticker '{ticker}' is invalid. Skipping...")
