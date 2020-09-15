# coding: utf-8
import sys, os
import argparse
import time
import numpy as np
sys.path.append(os.pardir)

from AReM import *
from model import *


class Tester:
    """
    test 해주는 클래스. 수정불가
    ----------
    network : 네트워크
    x_test : 발리데이션 데이터
    t_test : 발리데이션 데이터에 대한 라벨
    mini_batch_size : 미니배치 사이즈
    verbose : 출력여부

    ----------
    """
    def __init__(self, network, x_test, t_test, mini_batch_size=100, verbose=True):
        self.network = network
        self.x_test = x_test
        self.t_test = t_test
        self.batch_size = int(mini_batch_size)
        self.verbose = verbose
        self.train_size = x_test.shape[0]

    def accuracy(self, x, t):
        """
        수정불가
        """
        if t.ndim != 1: t = np.argmax(t, axis=1)

        acc = 0.0
        start_time = time.time()

        for i in range(int(x.shape[0] / self.batch_size)):
            tx = x[i * self.batch_size:(i + 1) * self.batch_size]
            tt = t[i * self.batch_size:(i + 1) * self.batch_size]

            y = self.network.predict(tx)
            y = np.argmax(y, axis=1)
            acc += np.sum(y == tt)

        inference_time = (time.time()-start_time)/x.shape[0]

        return acc / x.shape[0], inference_time


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="사용예)python test.py --sf=myparam.")
    parser.add_argument("--sf", required=False, default="params.pkl", help="save_file_name")
    args = parser.parse_args()

    (_, _), (x_test, t_test) = load_AReM(one_hot_label=False) 

    network = Model()

    tester = Tester(network, x_test, t_test)

    network.load_params(args.sf)


    # 배치사이즈100으로 accuracy test, 다른 배치사이즈로 학습했다면 결과가 달라질 수 있습니다.
    test_acc, inference_time = tester.accuracy(x_test, t_test)

    print("=============== Final Test Accuracy ===============")
    print("test acc:" + str(test_acc) + ", inference_time:" + str(inference_time))
