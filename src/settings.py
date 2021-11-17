import os

import mlflow

########################################
## 初期設定
########################################
# mlflow trackingサーバのURLを指定
mlflow.set_tracking_uri('http://host.docker.internal:5000')

# オブジェクトストレージへの接続情報を指定
os.environ['MLFLOW_S3_ENDPOINT_URL'] = 'http://host.docker.internal:9000'
os.environ['AWS_ACCESS_KEY_ID'] = 'minio-access-key'
os.environ['AWS_SECRET_ACCESS_KEY'] = 'minio-secret-key'