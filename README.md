# Which Examples Should be Multiply Annotated? Active Learning When Annotators May Disagree

By: [Connor Baumler](https://ctbaumler.github.io/) `<baumler@umd.edu>`, Anna Sotnikova, and Hal Daumé III

```
@inproceedings{baumler-etal-2023-examples,
    title = "Which Examples Should be Multiply Annotated? Active Learning When Annotators May Disagree",
    author = "Baumler, Connor  and
      Sotnikova, Anna  and
      Daum{\'e} III, Hal",
    booktitle = "Findings of the Association for Computational Linguistics: ACL 2023",
    month = jul,
    year = "2023",
    address = "Toronto, Canada",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2023.findings-acl.658",
    doi = "10.18653/v1/2023.findings-acl.658",
    pages = "10352--10371",
}
```


## Usage

For example, to finetune on MHS repect using DAAL (query by the absolute difference of human and model entropy, train an entropy predictor, query for one label at a time) with a entropy predictor budget of 100 comments and a counter of 0 to be added to output files (if doing multiple runs):

```
python finetune_al.py -l respect -e initial_budget -a single_any -m abs_ent_diff -bc 100  -c 0
```
