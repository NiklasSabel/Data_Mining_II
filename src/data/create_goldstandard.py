import datetime
import pandas as pd

def _get_target(row, df_ord, start_period):
  purchase = df_ord[(df_ord["userID"] == row["userID"]) & (df_ord["itemID"] == row["itemID"])]
  if len(purchase) == 0:
    return 0
  else:
    return int((purchase.iloc[0]["date"] - start_period).days / 7) + 1

def compute_target(df_sub, df_ord, start_period):
  df_ord = df_ord[(df_ord["date"] >= start_period) & (df_ord["date"] < start_period + datetime.timedelta(days=28))]
  df_gold = df_sub.copy()
  df_gold["prediction"] = df_gold.apply(lambda row: _get_target(row, df_ord, start_period), axis=1)
  return df_gold

if __name__=="__main__":
  df_ord_all = pd.read_csv("../../data/orders.csv", sep="|", parse_dates=["date"])
  df_sub_dec = pd.read_csv("../../data/submission_dec.csv", sep="|")
  df_sub_jan = pd.read_csv("../../data/submission_jan.csv", sep="|")

  df_gold_dec = compute_target(df_sub_dec, df_ord_all, datetime.datetime(2020, 12, 1))
  df_gold_dec.to_csv("../../data/gold_dec.csv", sep="|", index=False)

  df_gold_jan = compute_target(df_sub_jan, df_ord_all, datetime.datetime(2021, 1, 1))
  df_gold_jan.to_csv("../../data/gold_jan.csv", sep="|", index=False)

  df_gold_dec