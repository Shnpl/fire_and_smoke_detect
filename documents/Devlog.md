# 2023_10_29
* Try to use yolo directly on orginal images given in the dataset.
# 2023_10_30
* 是单独训练两个模型,重新把数据集分开
* Yolo v1 直接开始训练,lr=1e-4, 半个epoch之后跑飞,loss->nan
* 调整到1e-5,继续训练,可以稳定训练,15个epoch之后可视化,部分可以识别,框定还算准确,但是没有进行定量(mAP等指标)的评估.检查了原有代码,pipeline应该是没有问题
* 让这个模型训练到64个epoch,然后进行评估
* 多卡的情况下出现卡死,目前还不知道原因,使用单卡训练中
* 今日任务进度超前于预期
# 2023_11_11
* 分类进行:
    * 输入图片
    * 直方图->
        * 三通道差值小于Th -> 灰度图
        * 绿色 完全主导-> 绿色IR
        * 输入IR Discrimination Net 
            * RGB图
            * IR图
* 灰度图/IR图 均灰度化后输入 灰度Yolo
* RGB图输入 RGB Yolo
* 三个模型分别训练,IR Discrimination Net 应该是一个相对简单的模型,可以先训练,看看效果
-> IR Discriminator 正确率在1.0,在测试集上完全没有错误，应该可用