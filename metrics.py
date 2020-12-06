import numpy as np

def HR_score(y_true, y_pred, topk):
    """compute Hit_Ratio@topk score"""
    assert len(y_true) == len(y_pred)
    
    n_positive = sum(y_true == 1)
    assert topk > 0 and n_positive > 0
    
    topk = min(topk, sum(y_true == 1))
    order = np.argsort(y_pred, kind="stable")[-topk:][::-1]
    y_true = y_true[order]

    # return HR@topk
    return (y_true > 0).sum() / n_positive
	
def DCG_score(y_true, y_pred, topk):
    """compute DCG_score@topk"""
    assert len(y_true) == len(y_pred)
    
    n_positive = sum(y_true == 1)
    assert topk > 0 and n_positive > 0

    topk = min(topk, sum(y_true == 1))
    order = np.argsort(y_pred, kind="stable")[-topk:][::-1]
    y_true = y_true[order]

    # linear gain, not seansitive to position
    # gains = y_true
    # exponential gain, seansitive to position
    gains = 2 ** y_true - 1

    # highest rank is 1 so +2 instead of +1
    discounts = np.log2(np.arange(len(y_true)) + 2)

    # return DCG@topk
    return np.sum(gains / discounts)

def nDCG_score(y_true, y_pred, topk):
    """compute nDCG_score@topk"""
    best = DCG_score(y_true, y_true, topk)
    real = DCG_score(y_true, y_pred, topk)
    
    # return nDCG@topk
    return real / best if best != 0 else 0
