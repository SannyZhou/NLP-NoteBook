# NLP
## Distributed Representation
### Neural Language Model
[A Neural Probabilistic Language Model](http://www.jmlr.org/papers/volume3/bengio03a/bengio03a.pdf)
[cdsn](https://blog.csdn.net/u014568072/article/details/78557837)
### Pretrained Word Embedding
#### Word2Vec
1. `from gensim.models import Word2Vec` <br>
2. [Distributed Representations of Words and Phrases and Their Compositionality](http://papers.nips.cc/paper/5021-distributed-representations-of-words-and-phrases-and-their-compositionality.pdf)

#### GloVe
1. [Source Code](https://github.com/stanfordnlp/GloVe)
2. [GloVe: Global Vectors for Word Representation](https://nlp.stanford.edu/pubs/glove.pdf)

### Pretrained Language Model
#### ELMo
1. Bidirectional language model
2. residual connection
3. character-level embedding
[paper]Deep contextualized word representations(https://arxiv.org/pdf/1802.05365.pdf)

#### GPT
[paper]

#### Bert
[paper]

#### GPT2
[paper]

#### XLNet
[paper]

#### RoBerta
1. Paper: [RoBERTa: A Robustly Optimized BERT Pretraining Approach](https://arxiv.org/pdf/1907.11692.pdf)<br>
2. toolkit:[FAIRSEQ: A Fast, Extensible Toolkit for Sequence Modeling](https://arxiv.org/pdf/1904.01038.pdf)
3. [github](https://github.com/pytorch/fairseq)



## Classification
### Tasks
1. Sentimental Classification
2. Text Classification
3. Textual entailment
4. Paraphrase Identification (detection)

### Convolution Neural Network in NLP
N-gram feature of tokens (word/character)<br>
Paper: [Convolutional Neural Networks for Sentence Classification](https://arxiv.org/pdf/1408.5882.pdf)

### Recurrent Neural Network in NLP
Long-term dependency feature in Text <br>
Paper: [Recurrent Convolutional Neural Networks for Text Classification](https://www.aaai.org/ocs/index.php/AAAI/AAAI15/paper/view/9745/9552)

### CNN + RNN
N-gram feature of tokens + Contextual feature <br>
Paper:[A C-LSTM Neural Network for Text Classification](https://arxiv.org/pdf/1511.08630.pdf)

### ELMo (BiRNN) => Pretrained Feature
[paper]

### Transformer


## Sequence2Sequence (Encoder-Decoder)
### Tasks
1. Machine Translation
[Statistical MT]
[Neural MT]
2. Text Summarization
3.  Grammar Error Correction
4. Question & Answer ( Q&A)

### Models
#### Seq2Seq Based on RNN
#### Attention
#### Tensor2Tensor Based on Transformer
#### GPT (pretrained + finetune)
#### Bert (pretrained + finetune)


## NLP Fundemental Tasks
### Chinese Word Segmentation
### Part-of-Speech Tagging
### Parsing
1. Dependency Parsing
2. Constituency Parsing

### Semantic Role Labeling
### Named Entity Recognition
### Paraphrase Identification
### .etc
