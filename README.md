# PRM

[Position-Aware Recalibration Module: Learning From Feature Semantics and
Feature Position](https://www.ijcai.org/Proceedings/2020/0111.pdf), by [Xu Ma](https://13952522076.github.io/), [Song Fu](https://www.cse.unt.edu/~song/), is a novel plug-in module for improve CNN model capability with minimal computational overheads, and further improve the performances of other high-level visual tasks, like detection.


![PRM_module](figs/structure.pdf)


## Getting Start
### Installation

 __1. Download repo__
 
```Bash
git clone https://github.com/13952522076/PRM.git
cd PRM
```

__2. Requirements__

- Python3.6
- PyTorch 1.3+
- CUDA 10+
- GCC 5.0+
```Bash
pip install -r requirements.txt
```
__3. Install DALI and Apex （For ImageNet Training）__

DALI Installation:
```Bash
cd ~
# For CUDA10
pip install --extra-index-url https://developer.download.nvidia.com/compute/redist nvidia-dali-cuda100
# or
# For CUDA11
pip install --extra-index-url https://developer.download.nvidia.com/compute/redist nvidia-dali-cuda110
```
For more details, please see [Nvidia DALI installation](https://docs.nvidia.com/deeplearning/dali/user-guide/docs/installation.html).


Apex Installation:
```Bash
cd ~
git clone https://github.com/NVIDIA/apex
cd apex
pip install -v --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" ./
```
For more details, please see [Apex](https://github.com/NVIDIA/apex) or [Apex Full API documentation](https://nvidia.github.io/apex/).


<!--__Prepare ImageNet dataset__-->

<!--```Bash-->
<!--cd ~-->
<!--cd Efficient_ImageNet_Classification-->
<!--mkdir data-->
<!--cd data-->
<!--# Replace PATH_TO_ImageNet to your ImageNet dataset path-->
<!--ln -s PATH_TO_ImageNet imagenet-->
<!--```-->

## Training & Testing ImageNet
```Bash
# change the parameters accordingly if necessary
# e.g, If you have 4 GPUs, set the nproc_per_node to 4. If you want to train with 32FP, remove ----fp16.
python3 -m torch.distributed.launch --nproc_per_node=8 imagenet.py -a prm_resnet50 --fp16 --b 32
```


## Reference
If you find this work useful in your research, you can cite the corresponding papers listed below:

    @inproceedings{ma2020position,
      title={Position-Aware Recalibration Module: Learning From Feature Semantics and Feature Position},
      author={Ma, Xu and Fu, Song},
      booktitle={Proceedings of the Twenty-Ninth International Joint Conference on Artificial Intelligence},
      year={2020}
    }






