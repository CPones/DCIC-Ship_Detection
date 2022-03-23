# DCIC Ships Detection
## 一、项目背景
海上船舶目标检测对于领海安全、渔业资源管理和海上运输与救援具有重要意义，但在天气和海浪等不可控的自然因素影响下，依靠派遣海警船或基于可见光数据进行船舶目标监测等手段往往难以有效开展。卫星合成孔径雷达(SAR)是一种全天时、全天候、多维度获取信息的主动式微波成像雷达，为海洋上多尺度的船舶检测提供了强有力的数据保障和技术支持，在**遥感图像**船舶检测领域占有重要地位。由于SAR的成像原理与光学相机存在很大的差别，如何利用SAR数据特性设计出一套具有针对性的船舶检测方法是一大难点。本赛题鼓励选手通过数据算法寻找这个难题的新颖解法，进一步推动海上船舶智能检测的发展。
## 二、项目任务
快速精准的检测出船舶的垂直边框是船舶智能检测的基本需求。本项目以训练数据集中船舶和相应垂直边框信息为学习依据，要求对测试数据集中的船舶进行检测（图a），求解出船舶对应垂直边框（图b）。需要考虑SAR图像和船舶目标的特性，如背景强散射杂波的不均匀性，目标的不完整性、十字旁瓣模糊和临近目标干扰等，设计科学适用的算法模型进行船舶的智能检测。
![](https://ai-studio-static-online.cdn.bcebos.com/bbb26718571547019de46c49e4602a11eeface39ddf943399e8f21570012de46)
## 三、数据集介绍
下载链接：[https://aistudio.baidu.com/aistudio/datasetdetail/134192](https://aistudio.baidu.com/aistudio/datasetdetail/134192)

| 数据集划分 | 图片数量（张） |
| :--------: | :--------: |
| train     | 20504 |
| valid     | 1000  |
| test      | 18112 |

目录结构
```
DATASET_VOC/
├── annotations
│   ├── xxx1.xml #voc数据的标注文件
│   ├── xxx2.xml #voc数据的标注文件
│   ├── xxx3.xml #voc数据的标注文件
│   |   ...
├── images
│   ├── xxx1.jpg
│   ├── xxx2.jpg
│   ├── xxx3.jpg
│   |   ...
├── label_list.txt (必须提供，且文件名称必须是label_list.txt )
├── train.txt       (训练数据集文件列表, ./images/xxx1.jpg ./annotations/xxx1.xml)
└── valid.txt       (测试数据集文件列表)
TEST/
├── xxx1.jpg
├── xxx2.jpg
├── xxx3.jpg
```

