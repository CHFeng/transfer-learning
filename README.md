# 遷移式訓練
使用keras提供的預訓練模組進行遷移式訓練，將要訓練的資料分為指定種類的資料夾放置在/data之下。
* /data/train放置訓練的資料集
* /data/val放置驗證的資料集

執行train.py
```
python train.py
```
執行訓練完畢後，會產出.h5的模型檔案與模型訓練過程的acc&loss圖檔。

# DenseNet201訓練結果
train_loss: 0.2125 - train_accuracy: 0.9302 - val_loss: 0.4015 - val_accuracy: 0.8616

# Xception訓練結果
train_loss: 0.2561 - train_accuracy: 0.9067 - val_loss: 0.2372 - val_accuracy: 0.9241