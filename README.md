# CrossModalFlow

Pytorch implementation of *Promoting Single-Modal Optical Flow Network for Diverse Cross-modal Flow Estimation* (AAAI 2022)
## Usage
Download the pre-trained model, and put it in the 'pre_trained' folder.

*[baidu yun](https://pan.baidu.com/s/17wM6IaJPp-Loxj0N7IzsQg) access code: sztg*

You can check 'run_model.py' for detailed usage.

## Evaluation
Prepare the corresponding dataset and modify the path in 'evaluate_dataset.py'.

*[RGBNIR-Stereo](http://platformpgh.cs.cmu.edu/tzhi/RGBNIRStereoRelease/rgbnir_stereo/)*

*[TriModalHuman](https://www.kaggle.com/aalborguniversity/trimodal-people-segmentation)*

*[KITTI](http://www.cvlibs.net/datasets/kitti/eval_stereo_flow.php?benchmark=flow)*

*[Cross-KITTI](https://pan.baidu.com/s/17wM6IaJPp-Loxj0N7IzsQg) access code: sztg*

Then run evaluate_for_crossKitti.py / evaluate_for_rgbnir_stereo.py / evaluate_for_trimodalhuman.py.

## Training
Prepare the dataset and modify the path in 'dataset.py'.

[YoutubeVOS](https://youtube-vos.org/dataset/)

Run 'train.py' to train.


# 中文说明
## 使用：
下载预训练模型，并放入pre_trained文件夹：
 *[百度云](https://pan.baidu.com/s/17wM6IaJPp-Loxj0N7IzsQg) 提取码 sztg*

参考run_model.py的用法


## 评估：
准备相应的数据集，修改evaluate_dataset.py中的路径

*[RGBNIR-Stereo](http://platformpgh.cs.cmu.edu/tzhi/RGBNIRStereoRelease/rgbnir_stereo/)*

*[TriModalHuman](https://www.kaggle.com/aalborguniversity/trimodal-people-segmentation)*

*[KITTI](http://www.cvlibs.net/datasets/kitti/eval_stereo_flow.php?benchmark=flow)*

*[Cross-KITTI](https://pan.baidu.com/s/17wM6IaJPp-Loxj0N7IzsQg) 提取码 sztg*

运行evaluate_for_crossKitti.py / evaluate_for_rgbnir_stereo.py / evaluate_for_trimodalhuman.py


## 训练：
准备数据集，并修改dataset.py中的数据集路径

*[YoutubeVOS](https://youtube-vos.org/dataset/)*

运行train.py进行训练

# Acknowledgments
Parts of the code are borrowed from: 
[FlowNet2](https://github.com/NVIDIA/flownet2-pytorch) 
[PWC-Net](https://github.com/NVlabs/PWC-Net)
[RAFT](https://github.com/princeton-vl/RAFT)

