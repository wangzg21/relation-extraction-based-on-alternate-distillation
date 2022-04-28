# CNN-relation-extraction
## Environment Requirements
* python 3.6
* pytorch 1.3

## Data
* [SemEval2010 Task8]：https://github.com/wangzg21/relation-extraction-based-on-alternate-distillation/tree/main/datasets
* [wiki80]：https://github.com/wangzg21/relation-extraction-based-on-alternate-distillation/tree/main/datasets
* [Embedding - Turian et
al.(2010)](http://metaoptimize.s3.amazonaws.com/hlbl-embeddings-ACL2010/hlbl-embeddings-scaled.EMBEDDING_SIZE=50.txt.gz) \[[paper](https://www.aclweb.org/anthology/P10-1040.pdf)\]


## Usage
Download the embedding and decompress it into the `embedding` folder.
Run the following the commands to start the program.
```shell
python run.py
```
More details can be seen by `python run.py -h`.

3. You can use the official scorer to check the final predicted result.
```shell
perl semeval2010_task8_scorer-v1.2.pl proposed_answer.txt predicted_result.txt >> result.txt
```

## Result
The result of my version and that in paper are present as follows:
| paper | my version |
| :------: | :self-distillation: | :external distillation: | :alternate distillation: |
| 0.7868 | 0.8041| 0.8021| 0.8047|

The training log can be seen in `train.log` and the official evaluation results is available in `result.txt`.