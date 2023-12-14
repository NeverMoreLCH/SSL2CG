# SSL2CG
Implementation for the CVPR 2023 paper "[Exploring the Effect of Primitives for Compositional Generalization in Vision-and-Language](https://openaccess.thecvf.com/content/CVPR2023/papers/Li_Exploring_the_Effect_of_Primitives_for_Compositional_Generalization_in_Vision-and-Language_CVPR_2023_paper.pdf)"

## Prerequisites

### Environment
```bash
pip install -r requirements.txt
```

### Data Preparation

For Charades dataset, we use the same data preparation process as [MS-2D-TAN](https://github.com/microsoft/VideoX/tree/master/MS-2D-TAN), follows [here](https://github.com/microsoft/VideoX/tree/master/MS-2D-TAN#download-datasets) for details.

For Charades-CG dataset, download https://github.com/YYJMJC/Compositional-Temporal-Grounding/tree/main/Charades-CG and put them under `data/Charades-CG`.



## Training

```bash
python train.py --cfg experiments/charades/config.yaml
```

## Evaluation

We provide [a ckpt](https://drive.google.com/file/d/1or5YbBOuEi0Kcy3VUrsUcfkc9Tdmx4Iw/view?usp=sharing) with results similar to our paper, and compare them in the table below.

| Split          | Test-Trivial                   | Novel-Composition             | Novel-Word                    |
| -------------- | ------------------------------ | ----------------------------- | ----------------------------- |
| **Metric**     | **R1@0.5 \| R1@0.7 \| mIoU**   | **R1@0.5 \| R1@0.7 \| mIoU**  | **R1@0.5 \| R1@0.7 \| mIoU**  |
| paper version  | 58.14   \|   37.98   \|  50.58 | 46.54  \|   25.10   \|  40.00 | 50.36  \|   28.78   \|  43.15 |
| github version | 59.34   \|   38.53   \|  51.48 | 45.96  \|   24.67   \|  40.01 | 50.07  \|   29.21   \|  42.79 |


You can download the checkpoint and put it under `ckpt' folder, then run
```bash
python evaluate.py --cfg experiments/charades/config.yaml --load ckpt/MS-2D-TAN_iter20461.pt
```

## Citation

If any part of our paper and code is helpful to your work, please generously cite with:

```
@inproceedings{li2023exploring,
  title={Exploring the Effect of Primitives for Compositional Generalization in Vision-and-Language},
  author={Li, Chuanhao and Li, Zhen and Jing, Chenchen and Jia, Yunde and Wu, Yuwei},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={19092--19101},
  year={2023}
}
```

## Acknowledgments

- Thanks for the great [MS-2D-TAN](https://github.com/microsoft/VideoX/tree/master/MS-2D-TAN).
- We use the [Charades-CG dataset](https://github.com/YYJMJC/Compositional-Temporal-Grounding/tree/main/Charades-CG), thanks for their work.

