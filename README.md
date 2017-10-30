# LabelEmb
This is an implementation of the paper [Label Embedding Network: Learning Label Representation for Soft Training of Deep Networks](https://xxxx).  
Label Embedding Network can learn label representation (label embedding) during the training process of deep networks. With the proposed method, the label embedding is adaptively and automatically learned through back propagation. The original one-hot represented loss function is converted into a new loss function with soft distributions, such that the originally unrelated labels have continuous interactions with each other during the training process. As a result, the trained model can achieve substantially higher accuracy and with faster convergence speed. Experimental results based on competitive tasks demonstrate the effectiveness of the proposed method, and the learned label embedding is reasonable and interpretable. The proposed method achieves comparable or even better results than the state-of-the-art systems.  
  
The contributions of this work are as follows:  
**Learning label embedding and compressed embedding**: We propose the Label Embedding Network that can learn label representation for soft training of deep networks. Furthermore, some large-scale tasks have a massive number of labels, and a naive version of label embedding network will suffer from intractable memory cost problem. We propose a solution to automatically learn compressed label embedding, such that the memory cost is substantially reduced.  

**Interpretable and reusable**: The learned label embeddings are reasonable and interpretable, such that we can find meaningful similarities among the labels. The proposed method can learn interpretable label embeddings on both image processing tasks and natural language processing tasks. In addition, the learned label embeddings can be directly adapted for training a new model with improved accuracy and convergence speed.  

**General-purpose solution and competitive results**: The proposed method can be widely applied to various models, including CNN, ResNet, and Seq-to-Seq models. We conducted experiments on computer vision tasks including CIFAR-100, CIFAR-10, and MNIST, and on natural language processing tasks including LCSTS text summarization task and IWSLT2015 machine translation task.
Results suggest that the proposed method achieves significantly better accuracy than the existing methods (CNN, ResNet, and Seq-to-Seq). We achieve results comparable or even better than the state-of-the-art systems on those tasks.   
## DataSet
CIFAR100: [Download](https://www.cs.toronto.edu/~kriz/cifar.html)  
CIFAR10: [Download](https://www.cs.toronto.edu/~kriz/cifar.html)  
MNIST: [Download](http://yann.lecun.com/exdb/mnist/)  
LCSTS: [Download](http://icrc.hitsz.edu.cn/Article/show/139.html)  
IWSLT2015: [Download](https://nlp.stanford.edu/projects/nmt/data/iwslt15.en-vi/)
## Environment and Dependency
- Ubuntu 16.04
- Python 3.5
- Tensorflow 1.3
- Pytorch 0.2.0
## Training for Computer Vision
You can change the training mode by setting the parameter "mode", as the `mode=baseline`
means the baseline models(CNN, ResNet-8 or ResNet-18) and the `mode=emb` means our proposed
label embedding network. There are also some other super parameters, see the codes for more
details.  
### CIFAR-100
`python3 cifar100.py --mode=baseline`  
`python3 cifar100.py --mode=emb`  
The outputs will be in `./100_results`  
### CIFAR-10
`python3 cifar10.py --mode=baseline`  
`python3 cifar10.py --mode=emb`  
The outputs will be in `./10_results`  
### MNIST
`python3 cnn.py --mode=baseline`  
`python3 cnn.py --mode=emb`  
The outputs will be in `./cnn_results`  
`python3 mlp.py --mode=baseline`  
`python3 mlp.py --mode=emb`  
The outputs will be in `./mlp_results`  
## Training for Natural Language Processing
###LCSTS
```bash
python3 preprocess.py -train_src TRAIN_SRC_DATA -train_tgt TRAIN_TGT_DATA
					  -test_src TEST_SRC_DATA -test_tgt TEST_TGT_DATA
					  -valid_src VALID_SRC_DATA -valid_tgt VALID_TGT_DATA
					  -save_data data/lcsts/lcsts.low.share.train.pt
					  -lower -share
```
```bash
python3 train.py -gpus 0 -config lcsts.yaml -unk -score emb -loss emb -log label_embedding
```
```bash
python3 predict.py -gpus 0 -config lcsts.yaml -unk -score emb -restore data/lcsts/label_embedding/best_rouge_checkpoint.pt
```
###IWSLT2015
```bash
python3 preprocess.py -train_src TRAIN_SRC_DATA -train_tgt TRAIN_TGT_DATA
					  -test_src TEST_SRC_DATA -test_tgt TEST_TGT_DATA
					  -valid_src VALID_SRC_DATA -valid_tgt VALID_TGT_DATA
					  -save_data data/iwslt15/iwslt.low.train.pt
					  -lower
```
```bash
python3 train.py -gpus 0 -config iwslt.yaml -unk -score emb -loss emb -log label_embedding
```
```bash
python3 predict.py -gpus 0 -config iwslt.yaml -unk -score emb -restore data/lcsts/label_embedding/best_bleu_checkpoint.pt
```
## Results
### Results of Label Embedding on computer vision:  
![cv_tab.png](https://github.com/lancopku/LabelEmb/blob/master/fig/cv_tab.PNG)  

### Error rate curve for CIFAR-100, CIFAR-10, and MNSIT. 20 times experiments (the light color curves) are conducted for credible results both on the baseline and our proposed model. The average results are shown as deep color curves:  
![cv_fig.png](https://github.com/lancopku/LabelEmb/blob/master/fig/cv_fig.PNG)  

### Heatmaps generated by the label embeddings:  
![cv_heatmap.png](https://github.com/lancopku/LabelEmb/blob/master/fig/cv_heatmap.PNG)


