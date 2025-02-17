# 说明

## 环境

dgl==2.0.0
torch==2.2.1

## 数据

### 数据下载

* 1、English数据集下载地址：https://github.com/RingBDStack/KPGNN/tree/main/datasets/Twitter
* 2、下载到 data 目录下

### 数据准备

* 1、cd 到 data 目录下
* 2、执行 python generate_initial_features.py
* 3、执行 python construct_graph.py

## 运行

### 运行MGPC

1、先设置 `is_incremental` 为 `False` 运行main.py为pre-train
```shell
python .\mgpc\main.py
```

2、再次设置 `is_incremental` 为 `True` 运行main.py为fine-tune过程
```shell
python .\mgpc\main.py
``` 

### 运行QSGNN

1、先设置 `is_incremental` 为 `False` 运行main.py为pre-train
```shell
python .\qsgnn\main.py
```

2、再次设置 `is_incremental` 为 `True` 运行main.py为continue_train过程
```shell
python .\qsgnn\main.py
``` 

## 结果

打出的日志中包括 `This epoch took xxx mins` 查看时间