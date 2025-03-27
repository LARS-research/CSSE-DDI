# Customized Subgraph Selection and Encoding for Drug-drug Interaction Prediction

<p align="left">
<a href="https://neurips.cc/virtual/2024/poster/94377"><img src="https://img.shields.io/badge/NeurIPS%202024-Poster-brightgreen.svg" alt="neurips paper">
</p>

---

## Requirements

```sheel
torch==1.13.0
dgl-cu111==0.6.1
optuna==3.2.0
```

## Run

### Unpack Dataset
```shell
unzip datasets.zip
```

### Supernet Training
```shell
python run.py --encoder searchgcn --score_func mlp --combine_type concat --n_layer 3 --epoch 400 \
--batch 512 --seed 0 --search_mode joint_search --search_algorithm spos_train_supernet_ps2 --input_type allgraph \
--loss_type ce --dataset drugbank --ss_search_algorithm snas
```
### Sub-Supernet Training
```shell
python run.py --encoder searchgcn --score_func mlp --combine_type concat --n_layer 3 --epoch 400 \
--batch 512 --seed 0 --search_mode joint_search --search_algorithm spos_train_supernet_ps2 --input_type allgraph \
--loss_type ce --dataset drugbank --exp_note spfs --few_shot_op rotate --weight_sharing --ss_search_algorithm snas

python run.py --encoder searchgcn --score_func mlp --combine_type concat --n_layer 3 --epoch 400 \
--batch 512 --seed 0 --search_mode joint_search --search_algorithm spos_train_supernet_ps2 --input_type allgraph \
--loss_type ce --dataset drugbank --exp_note spfs --few_shot_op ccorr --weight_sharing --ss_search_algorithm snas

python run.py --encoder searchgcn --score_func mlp --combine_type concat --n_layer 3 --epoch 400 \
--batch 512 --seed 0 --search_mode joint_search --search_algorithm spos_train_supernet_ps2 --input_type allgraph \
--loss_type ce --dataset drugbank  --exp_note spfs --few_shot_op mult --weight_sharing --ss_search_algorithm snas

python run.py --encoder searchgcn --score_func mlp --combine_type concat --n_layer 3 --epoch 400 \
--batch 512 --seed 0 --search_mode joint_search --search_algorithm spos_train_supernet_ps2 --input_type allgraph \
--loss_type ce --dataset drugbank  --exp_note spfs --few_shot_op sub --weight_sharing --ss_search_algorithm snas
```
### Subgraph Selection and Encoding Function Searching
```shell
python run.py --encoder searchgcn --score_func mlp --combine_type concat --n_layer 3 --epoch 400 \
--batch 512 --seed 0 --search_mode joint_search --search_algorithm spos_arch_search_ps2 --input_type allgraph \
--loss_type ce --dataset drugbank  --exp_note spfs --weight_sharing --ss_search_algorithm snas --arch_search_mode ng
```

### Fine-tune the Searched Model
```shell
python run.py --encoder searchgcn --score_func mlp --combine_type concat --n_layer 3 --epoch 400 \
--batch 512 --seed 0 --train_mode spos_tune --search_mode joint_search --input_type allgraph \
--loss_type ce --dataset drugbank  --exp_note spfs --weight_sharing --ss_search_algorithm snas
```

## Citation

Readers are welcomed to follow our work. Please kindly cite our paper:

```bibtex
@inproceedings{du2024customized,
    title={Customized Subgraph Selection and Encoding for Drug-drug Interaction Prediction},
    author={Du, Haotong and Yao, Quanming and Zhang, Juzheng and Liu, Yang and Wang, Zhen},
    booktitle={The Thirty-eighth Annual Conference on Neural Information Processing Systems},
    year={2024}
}
```

## Contact
If you have any questions, feel free to contact me at [duhaotong@mail.nwpu.edu.cn](mailto:duhaotong@mail.nwpu.edu.cn).

## Acknowledgement

The codes of this paper are partially based on the codes of [SEAL_dgl](https://github.com/Smilexuhc/SEAL_dgl), [PS2](https://github.com/qiaoyu-tan/PS2), and [Interstellar](https://github.com/LARS-research/Interstellar). We thank the authors of above work.
