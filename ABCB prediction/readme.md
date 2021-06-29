# Stock Price Prediction of ABCB.US Using Ensemble Learning of LSTM and Random Forest

# Theoretical Framework:

Many stock forecasting problems remain in the measurement of daily calculation. In this project, the speculation of
stock price is accurate to the level of hours. At the same time, this project uses ensemble learning and LSTM to improve
the stability of the prediction results

# Research Methodology:

Ensemble learning of LSTM and Random Forest

## Dataset:

Dataset is taken from [stooq](https://stooq.com/db/h/). This dataset consists of Open, High, Low and Closing Prices of
Ameris Bancorp. stocks from 2020/7/16 to 2021/5/12 for each day's 16pm to 22pm. - total 1452 rows.

## Price indicator:

Widely indicators could be used for prediction. In this project, choose below three:

- RSI
- Williams %R
- Moving Average Convergence Divergence
- OHLC (For test only)

## Data Pre-processing:

#### Training:

80% of ABCB.US hourly stock price for training.

### Testing:

left 20%

#### OHLC

After converting the dataset into OHLC average, it becomes one column data. This has been converted into two column time
series data, 1st column consisting stock price of time t, and second column is stock price of time (t+1). All values
have been normalized between 0 and 1.

## Models:

Two sequential LSTM layers have been stacked together and one dense layer is used to build the RNN model using Keras
deep learning library. Since this is a regression task, 'linear' activation has been used in final layer.

To improve the prediction of stock price, Random forest model will be in ensemble learning process

## Version:
Python 3.8 and latest versions of all libraries including deep learning library Keras and Tensorflow.

## Preliminary finding:

## Conclusion:

# References:

# Pre
- Intro
    - problem statement
    - Clearly and distinguishable aims and objectives,

- Literature review
    - sufficient background and technical contents,
  
- research methodology,
    - 4 phases
    
- Conclusion
    - Gantt Chart,
    - Conclusion by reinstating main points,
    - limitations and future work
    