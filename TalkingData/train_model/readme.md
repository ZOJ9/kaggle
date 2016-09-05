    本次比赛主要用到了lr(liblinear和sklearn_lr)，gbdt（xgboost），nnet(keras_mlp)3个模型，思路如下：


    1）单个模型训练提交，发现liblinear不如sklearn_lr表现好（sklearn_lr的优化方法用的是l-bfgs）；
       xgboost中的gblinear比dbtree表现好，但都比lr好；mlp是表现最好的一个单模型，但是需要做大量的
       调参工作，同时特征需要采样，以免过拟合。


    2）采用stacking的方式，即就是把lr，gbdt，mlp训练出来的结果组合起来作为训练数据，再次进行训练，
       但是有很严重的过拟合现象，最后的结果也不尽如人意。


    3）类似blending，将数据分为10分，也就是做一个10折交叉验证，平均每一组结果，将平均的结果作为一个
       模型结果，调节不同的seed（10组)，最后将这10组结果平均，这种方法取到了最好的结果。
