1. pro_all_feature_01.py 将所有的特征文件关联到一起，最后根据所有的特征建立稀疏矩阵；
2. pro_all_feature_02.py 先分别处理每个特征文件，建立稀疏矩阵，最后关联所有的稀疏矩阵；
3. split_isno_events.py 根据是否有匹配的event_id将文件分为两部分，pro_device_events.py和
   pro_device_no_events.py分别对两部分进行稀疏化特征处理。
