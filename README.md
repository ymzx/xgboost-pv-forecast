# xgboost-pv-forecast
光伏发电预测
# xbboost_stack介绍
xgboost_stack.py中封装XgboostStack类，该类实现对数据加载、模型训练、模型保存、模型加载、模型推理、预测和真值可视化曲线、量化评估等功能 
# 使用示例
```python
  train = True # 训练or推理
  start_time = '2022-09-5 00:00:00'  # 预测起始时间
  end_time = '2022-09-15 23:45:00'  # 预测起始时间
  xgb_stack = XgboostStack(start_time=start_time, end_time=end_time, neighbour=4)
  xgb_stack.build_train_data()
  if train: xgb_stack.train()
  xgb_stack.load_model()
  xgb_stack.infer()
  xgb_stack.metric()
  xgb_stack.plot()
```


