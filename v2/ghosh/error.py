from keras import backend as K

def multilabel_error(label, output):
    p_hat = K.exp(output) / K.sum(K.exp(output))
    return -K.mean(K.sum(label * K.log(p_hat)))
