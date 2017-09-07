# Features:
* [Generative Adversarial Network][GAN]
* Weight regularization losses and dropout
  - I'm not sure yet whether they help or hurt...
  - at which places should i put dropout
* [Conditioning][cGAN] on data attributes (labels etc.)
  - usually the conditioning vector is just the one-hot label
  - can also be dense vector calculated from several additional data
  - not sure where to best include that? each layer or just append once
* [One-sided label smoothing][improvGan]
* [Feature Matching][improvGan]
  - I hope my code here is correct.
* Default architecture as in [DCGAN][DCGAN]
  - ReLU in Generator (Tanh for final)
  - Leaky ReLU in Discriminator
  - No pooling layers
  - Batch Normalization
  - No fully connected layers
  
# Not Yet Implemented
* Minbatch discrimination
* Historical Averaging
* Virtual Batch Normalization

# Papers
* [Generative Adversarial Networks][GAN]
* [Conditional Generative Adversarial Nets][cGan]
* [Unsupervised Representation Learning with Deep Convolutional Generative Adversarial Networks][DCGAN]
* [Improved Techniques for Training GANs][improvGan]

[GAN]: http://arxiv.org/abs/1406.2661
[cGan]: http://arxiv.org/abs/1411.1784
[DCGAN]: http://arxiv.org/abs/1511.06434
[improvGan]: http://arxiv.org/abs/1606.03498