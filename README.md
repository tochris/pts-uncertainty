# Parameterized Temperature Scaling

This repository is the official implementation of Parameterized Temperature Scaling (PTS) from: 

"Christian Tomani, Daniel Cremers, Florian Buettner, Parameterized Temperature Scaling for Boosting the Expressive Power in Post-Hoc Uncertainty Calibration, ECCV 2022". [[Paper]](https://arxiv.org/abs/2102.12182)


## Calibration

The PTS model can be applied as follows:

### Generate PTS model
```
import pts_calibrator
pts = pts_calibrator.PTS_calibrator(
        epochs = ...,
        lr = ...,
        weight_decay = ...,
        batch_size = ...,
        nlayers = ...,
        n_nodes = ...,
        length_logits = ...,
        top_k_logits = ...)
```

**Arguments for PTS model:**\
`epochs`: number of epochs for model tuning\
`lr`: learning rate for model tuning\
`weight_decay`: lambda for weight decay in loss function\
`batch_size`: batch_size for model tuning\
`n_layers`: number of layers\
`n_nodes`: number of nodes of each hidden layer\
`length_logits`: length of logits vector\
`top_k_logits`: top k logits used for tuning


### Tune PTS model based on validation data

`pts.tune(logits_valid, labels_valid)`

**Arguments:**\
`logits_valid` (tf.tensor or np.array): logits based on validation set of shape (N,length_logits)\
`labels_valid` (tf.tensor or np.array): labels based on validation set of shape (N,length_logits)


### Calibrate logits

`probs = pts.calibrate(logits)`

**Arguments:**\
`logits` (tf.tensor or np.array): logits of shape (N,length_logits)



## Evaluation
### ECE - Expected calibration error (Guo et al)
Required package from Kumar et al.: `pip3 install uncertainty-calibration`

`ece = metrics.ece(probs, labels)`

### Negative Log-Likelihood

`nll = metrics.nll(probs, labels)`

### Brier Score

`brier = metrics.brier(probs, labels)`

## Citation:

If you find this library useful please consider citing our paper:
```
@InProceedings{Tomani_2022_ECCV,
    author    = {Tomani, Christian and Cremers, Daniel and Buettner, Florian},
    title     = {Parameterized temperature scaling for boosting the expressive power in post-hoc uncertainty calibration},
    booktitle = {In European Conference on Computer Vision (ECCV)},
    year      = {2022}
}
```

## References
Chuan Guo, Geoff Pleiss, Yu Sun, Kilian Q, Weinberger. On Calibration of Modern Neural Networks, ICML 2017.\
Ananya Kumar and Percy Liang and Tengyu Ma, Verified Uncertainty Calibration, NeurIPS 2019.
