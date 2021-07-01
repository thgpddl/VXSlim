# VXSlim
A lightweight face expression recognition network

使用步骤：
1. 数据集准备
从百度云(链接：https://pan.baidu.com/s/1d9iElDNwVr0MiIYyN9pZUA 提取码：h2ok)下载数据集并解压到datasets文件夹中，目录结构如下：
```
--datasets
----fer2013plus
------test
------train
```
2. 安装重要的库
```
tensorflow==2.3.0
matplotlib
```
3. 开始训练
使用train.py文件开始训练
4. TensorFlowLite文件夹中提供了Keras模型转tflite模型的脚本文件
