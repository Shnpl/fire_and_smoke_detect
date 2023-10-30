# 2023_10_29
* Try to use yolo directly on orginal images given in the dataset.
# 2023_10_30
* 是单独训练两个模型,重新把数据集分开
* Yolo v1 直接开始训练,lr=1e-4, 半个epoch之后跑飞,loss->nan
* 调整到1e-5,继续训练,可以稳定训练,15个epoch之后可视化,部分可以识别,框定还算准确,但是没有进行定量(mAP等指标)的评估.检查了原有代码,pipeline应该是没有问题
* 让这个模型训练到64个epoch,然后进行评估
* 多卡的情况下出现卡死,目前还不知道原因,使用单卡训练中
* 今日任务进度超前于预期
