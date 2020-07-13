# README

## Environment

```
conda env create -f intentTorch110.yml
```

## Example to Run 

1. SEG unknown intent detection

```
cd spin_outlier_ws
seg_75.sh SMP Caption 0 0
```

2. SEG RecapsNet for zero shot learning

```
cd spin_outlier_ws
seg_zero_shot.sh SMP Caption 0
```

3. ReCapsNet
   - Please refer to `code/model_torch.CapsAll` and `code/do_spin_outlier`

## Note

- Please refer to `code/config` and `code/main_*` for configuration.
- The output results can be found in `code/saved_models` 

-----



```
@inproceedings{yan-etal-2020-unknown,
    title = "Unknown Intent Detection Using {G}aussian Mixture Model with an Application to Zero-shot Intent Classification",
    author = "Yan, Guangfeng  and
      Fan, Lu  and
      Li, Qimai  and
      Liu, Han  and
      Zhang, Xiaotong  and
      Wu, Xiao-Ming  and
      Lam, Albert Y.S.",
    booktitle = "Proceedings of the 58th Annual Meeting of the Association for Computational Linguistics",
    month = jul,
    year = "2020",
    address = "Online",
    publisher = "Association for Computational Linguistics",
    url = "https://www.aclweb.org/anthology/2020.acl-main.99",
    pages = "1050--1060",
}

@inproceedings{liu-etal-2019-reconstructing,
    title = "Reconstructing Capsule Networks for Zero-shot Intent Classification",
    author = "Liu, Han  and
      Zhang, Xiaotong  and
      Fan, Lu  and
      Fu, Xuandi  and
      Li, Qimai  and
      Wu, Xiao-Ming  and
      Lam, Albert Y.S.",
    booktitle = "Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing and the 9th International Joint Conference on Natural Language Processing (EMNLP-IJCNLP)",
    month = nov,
    year = "2019",
    address = "Hong Kong, China",
    publisher = "Association for Computational Linguistics",
    url = "https://www.aclweb.org/anthology/D19-1486",
    doi = "10.18653/v1/D19-1486",
    pages = "4799--4809",
}
```

