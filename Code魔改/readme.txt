本程序是于Matlab实现的，基于稀疏表示的人脸图像超分辨率重构课题。

其中文件组成如下：
Image文件夹是所有程序需要的图像信息。
	其中Origin文件夹是所有原始HR数据图像，其内分为训练集、验证集的图像内容信息。
	HR_Dictionary文件夹为HR字典内容
	HR_Patch文件夹为HR字典分块以后的图像块生成位置。
	LR_Dictionary文件夹为HR图像转化为LR图像的生成位置。
	LR_Patch文件夹为LR图像分块以后的图像块生成位置。
	LR_FeaturePatch文件夹为LR图像提取特征后分块的位置。
	TestingHRPatch_Path文件夹为验证集HR图像分块后的位置。
	TestingLRPatch_Path文件夹为验证集LR图像分块后的位置。
	Input&Output文件夹为输入以及重构输出的图像存放位置。


SuperResolution文件夹是所有程序源码。
	MainFunction.m文件是程序运行入口，其中对所有全局变量进行声明以及赋值。其中要负责【对图像读取输出的路径修正】
	ProgramFramwork文件夹是课题程序中所有大程序框架的程序源码。
		TrainingData_LocalCopy训练数据库的读取——训练人脸数目35,  测试人脸数目45
		TestingImageReading 测试数据读取【未使用】
		DictionaryDownsampleing获得的字典对进行LR下采样以及下采样补齐（图片不会总%块大小=0）
		DictionaryPatchGenerating输入图像与字典的分块补齐与分块以及验证集预处理
		TotalImage_SR_Recovery 碎片合并
		OutputTesting实验结果并比对
	Algorithnm文件夹是课题所需所有算法文件。内含有ADMM、KSVD、KSVD调用、OMP算法以及单patch稀疏重构模型求解的代码。
	Utils文件夹内含一些工具类，例如PSNR值与SSIM的性能评价、矩阵补0补全、图像的下采样以及图像的特征提取程序。
	Yang文件夹包含Yang论文中使用的L1/2正则优化稀疏优化算法，以及全局反向修正算法
	GOG文件夹含有GOG特征提取算法包【未使用】
	
	
本程序在不加入字典训练过程，进行稀疏重构的时间仅需要2分钟即可。
本程序除去Yang、GOG、OMP、KSVD外，皆由宁波大学15级黄亚鹏所撰写，欢迎讨论学习，共同进步。