# ts2020
2020腾讯广告算法大赛


data目录存放训练数据

在主目录执行,例如 python src/1.data-prepare.py

首先执行1.data-prepare.py 生成数据

下一步执行2.train-baseline-age.py 调参

python 2.train-baseline-age.py [embedingdim] [seqlen] [mode]
这三个参数分别是 嵌入的大小 序列长度 embedingba处理方式

用age调就可以，一般情况下起始准确率 20% 上升到50%以上

gender在90%以上，可以用相同参数

可以加大训练次数和调整优化器参数，包括学习率计划策略