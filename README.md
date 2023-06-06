# Which Examples Should be Multiply Annotated? Active Learning When Annotators May Disagree

By: [Connor Baumler](https://ctbaumler.github.io/) `<baumler@umd.edu`, Anna Sotnikova, and Hal Daumé III

Bibtex for ACL version will go here when avaible.


## Usage

For example, to finetune on MHS repect using DAAL (query by the absolute difference of human and model entropy, train an entropy predictor, query for one label at a time) with a entropy predictor budget of 100 comments and a counter of 0 to be added to output files (if doing multiple runs):

```
python finetune_al.py -l respect -e initial_budget -a single_any -m abs_ent_diff -bc 100  -c 0
```
