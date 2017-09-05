# Features:
* Generative Adversarial Network (http://arxiv.org/abs/1406.2661)
* One-sided label smoothing ()
* Weight regularization losses and dropout
  - I'm not sure yet whether they help or hurt...
  - at which places should i put dropout
* Conditioning on data attributes (labels etc.) (http://arxiv.org/abs/1411.1784)
  - usually the conditioning vector is just the one-hot label
  - can also be dense vector calculated from several additional data
  - not sure where to best include that? each layer or just append once

# Not Yet Implemented
* Feature Matching
* Mnbatch discrimination
* Historical Averaging
* Virtual Batch Normalization

# Papers
* http://arxiv.org/abs/1406.2661
* http://arxiv.org/abs/1411.1784

