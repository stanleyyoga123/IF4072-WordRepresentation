# IF4072-WordRepresentation

# To Run
```
python main.py <args>
```
Followed by the following arguments
| Args | Description |
| --- | ----------- |
| -b | Train BERT Model |
| -ib | Train IndoBERT Model |
| -w2v | Train Word2Vec Model |
| -ft | Train Fasttext Model |
| -w | Window Size |
| -t | 1: Skip Gram, else: CBOW |
| -vspace | Train VectorSpace model |
| -lgbm | Train LGBM Model |
| -xgb | Train XGB Model |
| -svm | Train SVM Model |
| -tfidf | Use TFIDF Vectorization method |
| -bow | Use BoW Vectorization method |
| -n | Model name if you use w2v or ft, but pretrained model name (bert-base-uncased, etc) if you use bert |
| -bs | Batch Size |
| -msl | Max Sequence Length |
| -lr | Learning rate used |
| -e | Number epochs used |

## Example
### Vector Space
```
python main.py -vspace -tfidf -lgbm
```
Using TFIDF vectorization and LGBM model

### Word Embedding
```
python main.py -w2v -n testing -lr 0.1 -bs 64 -e 2 -t 0 -w 5 -msl 128
```
Using "testing" model, 0.1 learning rate, 64 batch size, 2 epochs, CBOW, 128 max sequence length

### Word Embedding with Context
```
python main.py -ib -msl 512 -bs 4 -lr "3e-6" -e 3
```
Using indobert model, 512 max sequence length, 4 batch size, 3e-6 learning rate, 3 epochs

