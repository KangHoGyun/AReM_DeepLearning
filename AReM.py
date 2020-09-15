# coding: utf-8
try:
    import urllib.request
except ImportError:
    raise ImportError('You should use Python 3.x')

import pickle
import numpy as np
from model import *


def _change_one_hot_label(X):
    T = np.zeros((X.size, 6))
    for idx, row in enumerate(T):
        row[X[idx]] = 1

    return T


def load_AReM(normalize=False, standardze=False, one_hot_label=False):
    """AReM 데이터셋 읽기

    Parameters
    ----------
    normalize : 데이터를 0.0~1.0 사이의 값으로 정규화할지 정한다.
    standardze : 데이터를 평균을 기준으로 어느정도 떨어지게 만들지 정한다.
    one_hot_label :
        one_hot_label이 True면、레이블을 원-핫(one-hot) 배열로 돌려준다.
        one-hot 배열은 예를 들어 [0,0,1,0,0,0,0,0,0,0]처럼 한 원소만 1인 배열이다.

    Returns
    -------
    (트레인 데이터, 트레인 정답),(발리데이션 데이터, 발리데이션 정답)
    """
    #assert (not (normalize & standardze)), "Choose one"
    with open('dataset.pkl', 'rb') as f:
        dataset = pickle.load(f)

    if normalize:
        for key in ('train_', 'val_'):
            dataset[key] = dataset[key].astype(np.float32)
            dataset[key] = (dataset[key] - np.min(dataset[key], axis=0)) / (
                        np.max(dataset[key], axis=0) - np.min(dataset[key], axis=0))

    if standardze:
        for key in ('train_', 'val_'):
            dataset[key] = dataset[key].astype(np.float32)
            dataset[key] -= np.mean(dataset[key], axis=0)
            dataset[key] /= np.std(dataset[key], axis=0)

    if one_hot_label:
        dataset['train_label'] = _change_one_hot_label(dataset['train_label'])
        dataset['val_label'] = _change_one_hot_label(dataset['val_label'])

    return (dataset['train_'], dataset['train_label']), (dataset['val_'], dataset['val_label'])

