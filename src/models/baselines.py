import pandas as pd
import datetime
from eval_model import get_score
import sys
sys.path.append("..")
from data.create_goldstandard import compute_target

def always_1(df):
  df_base = df.copy()
  df_base["prediction"] = 1
  return df_base

def prev_month(df_sub, df_ord, start_period):
  df_base = compute_target(df_sub, df_ord, start_period)
  return df_base

if __name__=="__main__":
  df_ord_all = pd.read_csv("../../data/orders.csv", sep="|", parse_dates=["date"])
  df_sub_dec = pd.read_csv("../../data/submission_dec.csv", sep="|")
  df_sub_jan = pd.read_csv("../../data/submission_jan.csv", sep="|")
  
  df_gold_dec = pd.read_csv("../../data/gold_dec.csv", sep="|")
  df_gold_jan = pd.read_csv("../../data/gold_jan.csv", sep="|")

  df_base_1_dec = always_1(df_sub_dec)
  df_base_1_jan = always_1(df_sub_jan)
  
  df_base_2_dec = prev_month(df_sub_dec, df_ord_all, datetime.datetime(2020, 11, 1))
  df_base_2_jan = prev_month(df_sub_jan, df_ord_all, datetime.datetime(2020, 12, 1))

  print("Always predict '1':")
  print("December: %f"%get_score(df_base_1_dec, df_gold_dec))
  print("January: %f"%get_score(df_base_1_jan, df_gold_jan))

  print("\nPredict based on previous month:")
  print("December: %f"%get_score(df_base_2_dec, df_gold_dec))
  print("January: %f"%get_score(df_base_2_jan, df_gold_jan))