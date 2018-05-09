# Recurrent Convolutional Neural Network for Text Classification
Tensorflow implementation of "[Recurrent Convolutional Neural Network for Text Classification](https://www.aaai.org/ocs/index.php/AAAI/AAAI15/paper/view/9745)".

![rcnn](https://user-images.githubusercontent.com/15166794/39769535-703d02c8-5327-11e8-99d8-44a060e63e48.PNG)


## Data: Movie Review
* Movie reviews with one sentence per review. Classification involves detecting positive/negative reviews ([Pang and Lee, 2005](#reference)).
* Download "*sentence polarity dataset v1.0*" at the <U>[Official Download Page](http://www.cs.cornell.edu/people/pabo/movie-review-data/)</U>.
* Located in *<U>"data/rt-polaritydata/"</U>* in my repository.
* *rt-polarity.pos* contains 5331 positive snippets.
* *rt-polarity.neg* contains 5331 negative snippets.


## Implementation of Recurrent Structure

![recurrent_structure](https://user-images.githubusercontent.com/15166794/39777565-db89ca68-533e-11e8-8a87-785f98b3cfef.PNG)

* Bidirectional RNN (Bi-RNN) is used to implement the left and right context vectors.
* Each context vector is created by shifting the output of Bi-RNN and concatenating a zero state indicating the start of the context.


## Usage
### Train
* positive data is located in *<U>"data/rt-polaritydata/rt-polarity.pos"*</U>.
* negative data is located in *<U>"data/rt-polaritydata/rt-polarity.neg"*</U>.
* "[GoogleNews-vectors-negative300](https://code.google.com/archive/p/word2vec/)" is used as pre-trained word2vec model.
* Display help message:

	```bash
	$ python train.py --help
	```

* **Train Example:**
	
	```bash
	$ python train.py --cell_type "lstm" \
	--pos_dir "data/rt-polaritydata/rt-polarity.pos" \
	--neg_dir "data/rt-polaritydata/rt-polarity.neg"\
	--word2vec "GoogleNews-vectors-negative300.bin"
	```


### Evalutation
* Movie Review dataset has **no test data**.
* If you want to evaluate, you should make test dataset from train data or do cross validation. However, cross validation is not implemented in my project.
* The bellow example just use full rt-polarity dataset same the train dataset.
* **Evaluation Example:**

	```bash
	$ python eval.py \
	--pos_dir "data/rt-polaritydata/rt-polarity.pos" \
	--neg_dir "data/rt-polaritydata/rt-polarity.neg" \
	--checkpoint_dir "runs/1523902663/checkpoints"
	```


## Result
* Comparision between Recurrent Convolutional Neural Network and Convolutional Neural Network. 
* dennybritz's [cnn-text-classification-tf](https://github.com/dennybritz/cnn-text-classification-tf) is used for compared CNN model.
* Same pre-trained word2vec used for both models.

#### Accuracy for validation set
![accuracy](https://user-images.githubusercontent.com/15166794/39774365-9b8aa27e-5335-11e8-9710-515bc03dccb6.PNG)

#### Loss for validation set
![accuracy](https://user-images.githubusercontent.com/15166794/39774367-9bb2166a-5335-11e8-8d71-f06a61eee88a.PNG)


## Reference
* Recurrent Convolutional Neural Network for Text Classification (AAAI 2015), S Lai et al. [[paper]](https://www.aaai.org/ocs/index.php/AAAI/AAAI15/paper/view/9745)


