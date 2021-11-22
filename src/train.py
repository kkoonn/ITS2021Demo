import os
import tempfile
from datetime import *

from catboost import CatBoostRegressor
from catboost import Pool
import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import TimeSeriesSplit
from bshishov_forecasting_metrics import mape as fm_mape
from bshishov_forecasting_metrics import maape as fm_maape

import mlflow
import mlflow.catboost

import settings
from utils import *

# experimentの設定
print('experimentの設定')
mlflow.set_experiment('Bus-Delay-Prediction')

# データセットを一時的に保存するディレクトリを作成
temp_dir = tempfile.TemporaryDirectory()

########################################
# データ取得
########################################
print('データ取得')
dataset = pd.read_csv('data/all_route_with_weather.csv')
dataset.dropna(inplace=True)
dataset['weather'] = dataset['weather'].astype(int)
dataset.sort_values('SBSDepartureTime', inplace=True)
dataset.reset_index(inplace=True, drop=True)
# テンプディレクトリに記録
dataset.to_csv(os.path.join(
    temp_dir.name, 'all_route_with_weather.csv'), index=False)

########################################
# 前処理
########################################
print('前処理')
dataset.pop('DelayAtArrival')
dataset.pop('DelayAtDeparture')
dataset.pop('FBSArrivalTime')
dataset.pop('FBSDepartureTime')
dataset.pop('SBSOnTime')
dataset.pop('FBSOnTime')
dataset.pop('seq_no')
dataset.pop('date')
dataset.pop('operation_id')
# dataset.pop('weather')
dataset.pop('amount_of_rain')
# dataset.pop('drive_t_TillUniversityEnt')

y = dataset.pop('drive_t')
X = dataset
# テンプディレクトリに記録
y.to_csv(os.path.join(temp_dir.name, 'y.csv'), index=False)
X.to_csv(os.path.join(temp_dir.name, 'X.csv'), index=False)

########################################
# 訓練と評価
########################################
# 学習/評価データのインデックスを指定して，catboostでの学習・評価を行う


def objectiveByIndex(train_index_list, test_index_list, message='None'):
  with mlflow.start_run(run_name='month validation'):
    # カテゴリのカラムのみを抽出
    categorical_features_indices = ['ASL_ID', 'via', 'SBSCode', 'SBSDepartureTime',
                                    'FBSCode', 'year', 'month', 'day', 'hour', 'weather']

    # パラメータの指定
    params = {
      'iterations': 300,
      # 'depth' : 8,
      # 'learning_rate' : 0.23277813303374406,
      # 'random_strength' : 18,
      # 'bagging_temperature' : 0.022789890209972054,
      # 'od_type': 'Iter',
      # 'od_wait' : 16,
      'loss_function': 'MAE',
      'verbose': False,
      'random_state': 0,
      #    'task_type': 'GPU'
    }
    model = CatBoostRegressor(**params)

    # 保存していたデータセットを記録する
    mlflow.log_artifacts(temp_dir.name, artifact_path='dataset')
    # パラメータ記録
    mlflow.log_params(params)
    mlflow.log_param('msg', message)

    # 学習データ抽出
    print('train')
    #print(train_index_list)
    for i, index_tuple in enumerate(train_index_list):
      index1, index2 = index_tuple[0], index_tuple[1] + 1
      dt1, dt2 = X['SBSDepartureTime'][index1], X['SBSDepartureTime'][index2-1]
      print(dt1 + '~' + dt2)
      if i == 0:
        X_train = X.iloc[index1: index2, :]
        y_train = y.iloc[index1: index2, ]
      else:
        X_train = pd.concat([X_train, X.iloc[index1: index2, :]])
        y_train = pd.concat([y_train, y.iloc[index1: index2, ]])

    # 評価データ抽出
    print('test')
    #print(test_index_list)
    for i, index_tuple in enumerate(test_index_list):
      index1, index2 = index_tuple[0], index_tuple[1] + 1
      dt1, dt2 = X['SBSDepartureTime'][index1], X['SBSDepartureTime'][index2-1]
      print(dt1 + '~' + dt2)
      if i == 0:
        X_test = X.iloc[index1: index2, :]
        y_test = y.iloc[index1: index2, ]
      else:
        X_test = pd.concat([X_test, X.iloc[index1: index2, :]])
        y_test = pd.concat([y_test, y.iloc[index1: index2, ]])

    # 学習/テストデータ記録
    with tempfile.TemporaryDirectory() as tmp:
      np.savetxt(os.path.join(tmp, "X_train.csv"), X_train, delimiter=',', fmt='%s')
      np.savetxt(os.path.join(tmp, "X_test.csv"), X_test, delimiter=',', fmt='%s')
      np.savetxt(os.path.join(tmp, "y_train.csv"), y_train, delimiter=',')
      np.savetxt(os.path.join(tmp, "y_test.csv"), y_test, delimiter=',')
      mlflow.log_artifacts(tmp, artifact_path='train_test_data')

    # データセットの作成。Poolで説明変数、目的変数、
    # カラムのデータ型を指定できる
    train_pool = Pool(X_train, y_train, cat_features=categorical_features_indices)
    test_pool = Pool(X_test, y_test, cat_features=categorical_features_indices)

    # 学習
    model.fit(train_pool)
    # モデルをローカル環境に保存
    mlflow.catboost.save_model(model, 'model' + message)
    # モデルをMLflowに保存
    mlflow.catboost.log_model(model, 'model' + message)

    preds = model.predict(test_pool)
    # 精度の計算
    mae = mean_absolute_error(y_test, preds)
    rmse = np.sqrt(mean_squared_error(y_test, preds))
    mape = fm_mape(y_test, preds)
    maape = fm_maape(y_test, preds)
    # 精度の記録
    mlflow.log_metric('mae', mae)
    mlflow.log_metric('rmse', mae)
    mlflow.log_metric('mape', mape)
    mlflow.log_metric('maape', maape)

    print("MAE = {}, RMSE = {}, MAPE = {}, MAAPE = {}".format(mae, rmse, mape, maape))

    # Feature Importance
    feature_importances = model.get_feature_importance(train_pool)
    feature_names = X_train.columns
    for score, name in sorted(zip(feature_importances, feature_names), reverse=True):
      print('{}: {}'.format(name, score))
      mlflow.log_metric(name, score)

  return 0


# 文字列からDatetime型に変換
# catboostではDatetime型に対応していないため，訓練前に元に戻す必要がある
dataset['SBSDepartureTime'] = pd.to_datetime(dataset['SBSDepartureTime'])

# 年月に対応するデータフレームのインデックスを探す
# プローブデータが存在している年月
year_month_dict = {
  2019: [4, 5, 6, 7, 8, 9, 10, 11, 12],
  2020: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
  2021: [1, 2, 3, 4, 5, 6]
}
# インデックスリスト
index_list = []
for year_month in year_month_dict.items():
  print(year_month)
  #print(index_list)
  year = year_month[0]
  for month in year_month[1]:
    dt1 = datetime(year, month, 1)
    dt2 = datetime(year+(month//12), month % 12+1, 1)
    #print(dt1, dt2)
    index = getIndexByRangeDate(dataset, dt1, dt2)
    index_list.append(index)

# Datetime型から文字列に変換する
dataset['SBSDepartureTime'] = dataset['SBSDepartureTime'].astype(str)

print('訓練・評価開始')
# 訓練データ: 2019/04/01~2019/12/31
# 評価データ: 2019/01/01~2020/03/31
objectiveByIndex(index_list[0:9], index_list[9:12], message='A')
# 訓練データ: 2020/04/01~2020/12/31
# 評価データ: 2021/01/01~2021/03/31
objectiveByIndex(index_list[12:21], index_list[21:24], message='B')
# 訓練データ: 2021/04/01~2021/05/31
# 評価データ: 2021/06/01~2021/06/31
objectiveByIndex(index_list[24:26], index_list[26:27], message='C')
# 訓練データ: 2019/04/01~2019/12/31
# 評価データ: 2021/04/01~2021/06/31
objectiveByIndex(index_list[0:9], index_list[24:27], message='D')