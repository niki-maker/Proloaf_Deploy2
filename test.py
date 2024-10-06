import sys
import os
local_proloaf_path = 'Proloaf Working/src'
sys.path.insert(0, local_proloaf_path)

# Check that proloaf directory exists
print(os.listdir(local_proloaf_path))

# Now, attempt to import the local version
import proloaf
print(proloaf.__file__)


from functools import partial
from pathlib import Path
import datetime
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

from proloaf import modelhandler as mh
from proloaf.confighandler import read_config
from proloaf.tensorloader import TimeSeriesData
from proloaf import datahandler as dh
from enum import Enum

# Path to the proloaf configuration
config_path = 'targets/opsd/config.json'
model_path = 'oracles/Niki/opsd_recurrent.pkl'
data_path = 'data/modified_opsd.csv'
config = read_config(config_path=config_path)




model = mh.ModelHandler.load_model(model_path)

def extend_df(df: pd.DataFrame, add_steps: int):
  freq = df.index.freq if df.index.freq else df.index.inferred_freq
  add_index = pd.date_range(df.index[-1], periods=add_steps + 1, freq=freq, inclusive="right", name=df.index.name)
  add_df = pd.DataFrame(columns=df.columns, index=add_index)
  df = pd.concat((df, add_df))
  return df

def predict(df: pd.DataFrame):
  config["history_horizon"] = len(df)
  ts_data = TimeSeriesData(
      df,
      preparation_steps=[
          partial(dh.set_to_hours, freq="1h"),  # Adjust freq to forecast frequency
          partial(
              dh.fill_if_missing, periodicity=config.get("periodicity", 24)
          ),  # Set in the model config
          partial(
              extend_df, add_steps=24
          ),  # Extend dataframe by 24 steps for forecast
          dh.add_cyclical_features,
          dh.add_onehot_features,
          partial(dh.add_missing_features, all_columns=[*config["encoder_features"], *config["decoder_features"]]),
          model.scalers.transform,
          dh.check_continuity,
      ],
      **config,
  )
  ts_data.to_tensor()
  # data includes targets (NaN) discarded as they are not inputs
  data_tensors = [tens.unsqueeze(0) for tens in ts_data[0]][:-1]

  print(data_tensors)

  prediction = model.predict(*data_tensors)

  pred_df = pd.DataFrame(
      prediction[0, :, :, 0].detach().numpy(),
      index=ts_data.data.index[-24:],
      columns=config["target_id"],  # The last 24 timesteps are the forecast horizon
  )
  if model.scalers is not None:
      for col in pred_df.columns:
          pred_df[col] = model.scalers.manual_inverse_transform(pred_df[[col]], scale_as=col)
  return pred_df

def main():
  timeseries = pd.read_csv(data_path, sep=",", index_col="Time", parse_dates=True)  # Adjust sep and index_col

  # Define the start date and time
  start_time = pd.Timestamp('2019-01-01 00:00:00')

  # Extract data starting from the start_time and including the next 23 rows
  specific_data = timeseries[(timeseries.index >= start_time) & (timeseries.index < start_time + pd.Timedelta(hours=24))]

  prediction = predict(specific_data)
  prediction.to_csv("prediciton2.csv")

  # Calculate scaled residuals
  actual = specific_data[config["target_id"]]
  scaled_residuals = (actual - prediction) / (actual.max() - actual.min())

  # Plot actual vs predicted
  plt.figure(figsize=(12, 6))
  plt.plot(actual.index, actual, label='Actual Load')
  plt.plot(prediction.index, prediction, label='Predicted Load')
  plt.ylabel('Scaled Load')
  plt.xlabel('Time')
  plt.title('Actual vs Predicted Load (Scaled)')
  plt.legend()
  plt.show()

  # Plot scaled residuals
  plt.figure(figsize=(12, 6))
  plt.plot(scaled_residuals.index, scaled_residuals, color='red', label='Scaled Residuals')
  plt.ylabel('Scaled Residuals')
  plt.xlabel('Time')
  plt.title('Scaled Residuals')
  plt.legend()
  plt.show()

  # Calculate scaled residuals
  actual = specific_data[config["target_id"]]
  scaled_residuals = (actual - prediction) / (actual.max() - actual.min())
  scaled_residuals1 = (actual - prediction)

  # Calculate MSE and RMSE
  mse = np.mean((actual - prediction) ** 2)
  rmse = np.sqrt(mse)

  print("MSE:", mse)
  print("RMSE:", rmse)

  # Create boxplot for scaled residuals
  plt.figure(figsize=(12, 6))
  plt.boxplot(scaled_residuals1)
  plt.title("Boxplot of Scaled Residuals")
  plt.ylabel("Scaled Residuals")
  plt.show()

if __name__ == "__main__":
  # Add the local proloaf directory to sys.path
  # This should now point to your local directory
    main()