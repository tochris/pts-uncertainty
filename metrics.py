import calibration as cal #pip3 install uncertainty-calibration
from sklearn.metrics import log_loss

def ece(probs, labels):
    """Expected calibration error (Guo et al)"""
    assert np.shape(probs)==np.shape(labels), "shapes of probs and labels need to be equal!"
    labels = np.argmax(labels, 1)
    return cal.get_ece(probs, labels)

def brier(probs, labels):
    assert np.shape(probs)==np.shape(labels), "shapes of probs and labels need to be equal!"
    return np.mean(np.sum((probs - labels)**2, axis=1))

def nll(probs, labels):
    assert np.shape(probs)==np.shape(labels), "shapes of probs and labels need to be equal!"
    return log_loss(labels, probs)

def accuracy(probs, labels):
    assert np.shape(probs)==np.shape(labels), "shapes of probs and labels need to be equal!"
    probs = np.argmax(probs, 1)
    labels = np.argmax(labels, 1)
    return sum(labels == probs) * 1.0 / len(labels)
