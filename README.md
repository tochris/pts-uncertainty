# Parameterized Temperature Scaling

This repository is the official implementation of the Parameterized Temperature Scaling (PTS) uncertainty calibration method from: "Christian Tomani, Daniel Cremers, Florian Buettner, Parameterized Temperature Scaling for Boosting the Expressive Power in Post-Hoc Uncertainty Calibration, ECCV 2022". [[Paper]](https://arxiv.org/abs/2102.12182)


## Calibration

The PTS model can be applied as follows:

**Generate PTS model**\
```
import pts_calibrator
pts = pts_calibrator.PTS_calibrator(
        epochs=...,
        lr=...,
        weight_decay=...,
        batch_size=...,
        nlayers=...,
        n_nodes=...,
        length_logits=...,
        top_k_logits=...)
```

**Tune PTS model based on validation data**\

`pts.tune(logits_valid, labels_valid)`

**Calibrate logits**\

`probs = pts.calibrate(logits)`


## Evaluation
### ECE - Expected calibration error (Guo et al)
Required package from Kumar et al. (pip3 install uncertainty-calibration)

`ece = metrics.ece(probs, labels)`

### Negative Log-Likelihood

`nll = metrics.nll(probs, labels)`

### Brier Score

`brier = metrics.brier(probs, labels)`



## References
Chuan Guo, Geoff Pleiss, Yu Sun, Kilian Q, Weinberger. On Calibration of Modern Neural Networks, ICML 2017.
Ananya Kumar and Percy Liang and Tengyu Ma, Verified Uncertainty Calibration, NeurIPS 2019.
