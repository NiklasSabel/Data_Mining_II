import pandas as pd

def count_points(pred, gold):
    df = pd.merge(pred, gold, on=["userID", "itemID"], suffixes=("_pred", "_gold"))
    df["points"] = df.apply(_compute_points_for_row, axis=1)
    return df["points"].sum()

def _compute_points_for_row(row):
    y_pred, y_gold = row.prediction_pred, row.prediction_gold
    if y_pred == y_gold:
        # one point if "no order" (0) is predicted correctly; three points if order week is predicted correctly
        return 1 if y_pred == 0 else 3
    # one point if order is predicted correctly (but not the correct week), otherwise zero points
    return 1 if (y_pred > 0 and y_gold > 0) else 0

def get_score(pred, gold):
    points = count_points(pred, gold)
    max_points = count_points(gold, gold)
    score = points / max_points
    return score