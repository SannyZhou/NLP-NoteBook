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
Paper: [Deep contextualized word representations](https://arxiv.org/pdf/1802.05365.pdf)<br>

1. Bidirectional language model
2. residual connection
3. character-level embedding


#### GPT
[Improving Language Understanding
by Generative Pre-Training](https://s3-us-west-2.amazonaws.com/openai-assets/research-covers/language-unsupervised/language_understanding_paper.pdf)
[github](https://github.com/huggingface/pytorch-openai-transformer-lm)
1. left to right
2. transformer encoder for finetuning, multi-layer decoder for language model
3. BPE(byte pair encoding) token

#### Bert
[BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding
](https://arxiv.org/pdf/1810.04805.pdf)
[github](https://github.com/google-research/bert)
1. bidirectional LM (mask)
2. word-level token
3. LM Multi-task with Next Sentence Predicition

#### GPT2
[Language Models are Unsupervised Multitask Learners](https://d4mucfpksywv.cloudfront.net/better-language-models/language-models.pdf)
[github](https://github.com/openai/gpt-2)
1. left-to-right
2. larger and deeper than gpt
3. BPE token
4. few modification in Transformer (position of Layer-Normalization .etc)

#### XLNet
[XLNet: Generalized Autoregressive Pretraining for Language Understanding
](https://arxiv.org/abs/1906.08237)
1. Bidirectional LM by Random Permutation instead of Mask (better for text generation task, consistent in train and test)
2. Two different Attention
3. Delete NSE

#### RoBerta
[RoBERTa: A Robustly Optimized BERT Pretraining Approach](https://arxiv.org/pdf/1907.11692.pdf)<br>
toolkit:[FAIRSEQ: A Fast, Extensible Toolkit for Sequence Modeling](https://arxiv.org/pdf/1904.01038.pdf)
[github](https://github.com/pytorch/fairseq)
1. Bidirectional LM with different mask in various epoch (more robust than origin mask method of bert)
2. Experiment on NSE with various kind of input and objectives. And prove that removing NSE loss could slightly improve performance of downstream tasks.


#### KnowBert
[Knowledge Enhanced Contextual Word Representations](https://arxiv.org/pdf/1909.04164.pdf)
===
TODO

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
[Deep contextualized word representations](https://arxiv.org/abs/1802.05365)

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
