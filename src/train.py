import os
import tempfile

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
dataset.to_csv(os.path.join(temp_dir.name, 'all_route_with_weather.csv'), index=False)

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
def objective():
  # カテゴリのカラムのみを抽出
  categorical_features_indices = ['ASL_ID', 'via', 'SBSCode', 'SBSDepartureTime',
                                  'FBSCode', 'year', 'month', 'day' ,'hour', 'weather']
  
  # パラメータの指定
  params = {
    'iterations' : 300,                         
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
  # パラメータ記録
  
  
  model = CatBoostRegressor(**params)
  # 5分割交差検証
  folds = TimeSeriesSplit(n_splits=5)
  
  for train_index, test_index in folds.split(X):
    # 実験記録開始
    with mlflow.start_run(run_name='Five-part validation'):
      # 保存していたデータセットを記録する
      mlflow.log_artifacts(temp_dir.name, artifact_path='dataset')
      
      # パラメータ記録
      mlflow.log_params(params)      
      
      X_train, X_test = X.iloc[train_index[0]:train_index[-1], :], X.iloc[test_index[0]:test_index[-1], :]
      y_train, y_test = y.iloc[train_index[0]:train_index[-1]], y.iloc[test_index[0]:test_index[-1]]
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
      test_pool  = Pool(X_test, y_test, cat_features=categorical_features_indices)
      
      # 学習
      print('学習')
      model.fit(train_pool)
      # モデルの記録
      mlflow.catboost.log_model(model, 'model')
      # 予測
      preds = model.predict(test_pool)
      # 精度の計算
      print('評価')
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

print('訓練・評価開始')
objective()