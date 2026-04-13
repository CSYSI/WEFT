# (AAAI,2026) Small but Mighty: Dynamic Wavelet Expert-Guided Fine-Tuning of Large-Scale Models for Optical Remote Sensing Object Segmentation

Yanguang Sun, Chao Wang, Jian Yang, Lei Luo<br />

Our work has been accepted for **AAAI 2026**. The relevant code has been open sourced, please carefully refer to the environment. txt for configuring the environment.

If you are interested in our work, please do not hesitate to contact us at **Sunyg@njust.edu.cn via email**.


<img width="1335" height="368" alt="image" src="https://github.com/user-attachments/assets/3140e10c-fe8b-45bb-9475-5f15e83dcd5a" />
<img width="1334" height="874" alt="image" src="https://github.com/user-attachments/assets/ef6ba5e2-e905-4370-8989-a835747e14d1" />

# Segmentation results

We provide the segmentation results of the proposed WEFT model under in Optical Remote Sensing Images.

WEFT_AAAI26_ORSIs [(https://pan.baidu.com/s/1ewSbkjKOQsusGDlpx2fk9A), PIN:t7xg] 


# Expend Applications

We provide the segmentation results of our WEFT method under in camouflage, natural and medical scenarios.

WEFT_AAAI26_COD/SOD/PS [(https://pan.baidu.com/s/1I8UFBVeBKFdhuSiAU8zg1A), PIN:wtxk] 


# Training

To train WEFT model on ORSSD on a single node with 2 gpus run:

```shell

bash dist_train.sh configs/COS/WEFT_RSSOD_ORSSD.py 2 --seed 2024

```

# Testing

The visual segmentation results can be obtained through image_demo.py



# Citation

If you use DPU-Former in your research or wish to refer to the baseline results, please use the following BibTeX entry.

```
@article{WEFT,
  title={Small but Mighty: Dynamic Wavelet Expert-Guided Fine-Tuning of Large-Scale Models for Optical Remote Sensing Object Segmentation},
  author={Sun, Yanguang and Wang, Chao and Yang, Jian and Luo, Lei},
  journal={arXiv preprint arXiv:2601.09108},
  year={2026}
}
```

```
@article{WEFT,
  title={Small but Mighty: Dynamic Wavelet Expert-Guided Fine-Tuning of Large-Scale Models for Optical Remote Sensing Object Segmentation},
  author={Sun, Yanguang and Wang, Chao and Yang, Jian and Luo, Lei},
  journal={AAAI Conference on Artificial Intelligence (AAAI)},
  year={2026}
}
```
