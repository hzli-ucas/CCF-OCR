# 背景介绍
本项目是CCF竞赛[[1]](#ref-1)的参赛项目，拟解决身份证复印件识别的问题，在复赛测试集上达到95%的识别准确率。
竞赛提供的数据集为生成数据，与真实样本存在差异。我们的方案针对竞赛数据设计实现，故依赖许多强假设。
应用场景发生变化时，需要对方案进行相应调整。

# 数据特点
图像背景干净无噪声，每幅图中包含一张身份证的正面和反面。
身份证的大小、形状始终不变，为长约445像素、宽约280像素的矩形，左上角标有“仅供BDCI比赛使用”字样。
身份证上叠加的半透明水印会遮挡文字或边缘，水印方向与身份证的方向始终一致。图像存在不同程度的模糊。
![avatar](https://github.com/hzli-ucas/CCF-OCR/blob/master/images/00df9505b7e647d8b936fd4bf939afdd.jpg)

<div id='ref-1'>
  
  [1] 2019年CCF大数据与计算智能大赛（CCF Big Data & Computing Intelligence Contest，简称CCF BDCI），[“基于OCR的身份证要素提取”赛题](https://www.datafountain.cn/competitions/346)
  
</div>
