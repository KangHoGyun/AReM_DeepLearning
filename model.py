# coding: utf-8
# 2020/인공지능/final/B511004/강호균
import sys
import os
from collections import OrderedDict
import pickle
import numpy as np
#sys.path.append(os.pardir)  # 부모 디렉터리의 파일을 가져올 수 있도록 설정


def softmax(x):
    if x.ndim == 2:
        x = x.T
        x = x - np.max(x, axis=0)
        y = np.exp(x) / np.sum(np.exp(x), axis=0)

        return y.T
    x = x - np.max(x)  # 오버플로 대책
    return np.exp(x) / np.sum(np.exp(x))

def cross_entropy_error(y, t):
    if y.ndim == 1:
        t = t.reshape(1, t.size)
        y = y.reshape(1, y.size)

    # 훈련 데이터가 원-핫 벡터라면 정답 레이블의 인덱스로 반환
    if t.size == y.size:
        t = t.argmax(axis=1)

    batch_size = y.shape[0]
    return -np.sum(np.log(y[np.arange(batch_size), t] + 1e-7)) / batch_size


class Relu:
    def __init__(self):
        self.mask = None

    def forward(self, x):
        self.mask = (x <= 0)
        out = x.copy()
        out[self.mask] = 0
        return out

    def backward(self, dout):
        dx = dout.copy()
        dx[self.mask] = 0
        return dx

class CustomActivation:
    def __init__(self):
        pass

    def forward(self, x):
        pass

    def backward(self, dout):
        pass


class Affine:
    def __init__(self, W, b):
        self.W, self.b = W, b  # 각 계층마다 w, b 값을 초기화 해줘야 한다.
        self.x, self.dW, self.db = None, None, None  # 학습할 때 필요함. gradient decent 적용할 때

    def forward(self, x):
        self.x = x
        out = np.dot(x, self.W) + self.b

        return out

    def backward(self, dout):
        dx = np.dot(dout, self.W.T)  # 얘만 전 layer로 보내줘서 return 해주는 것.
        self.dW = np.dot(self.x.T, dout)
        self.db = np.sum(dout, axis=0)

        return dx


class SoftmaxWithLoss:
    def __init__(self):
        self.loss = None
        self.y, self.t = None, None

    def forward(self, x, t):
        self.t = t
        self.y = softmax(x)
        self.loss = cross_entropy_error(self.y, self.t)

        return self.loss

    def backward(self, dout):
        batch_size = self.t.shape[0]
        dx = (self.y - self.t) / batch_size

        return dx


class SGD:
    def __init__(self, lr=0.01):
        self.lr = lr

    def update(self, params, grads):
        for key in params.keys():
            params[key] -= self.lr * grads[key]


class CustomOptimizer:
    pass


class BatchNormalization:
    def __init__(self, gamma, beta, momentum=0.9, running_mean=None, running_var=None):
        self.gamma = gamma
        self.beta = beta
        self.momentum = momentum
        self.input_shape = None

        # 시험할 때 사용할 평균과 분산
        self.running_mean = running_mean
        self.running_var = running_var

        # backward 시에 사용할 중간 데이터
        self.batch_size = None
        self.xc = None
        self.std = None
        self.dgamma = None
        self.dbeta = None

    def forward(self, x, train_flg=True):
        self.input_shape = x.shape
        if x.ndim != 2:
            N, C, H, W = x.shape
            x = x.reshape(N, -1)

        out = self.__forward(x, train_flg)

        return out.reshape(*self.input_shape)

    def __forward(self, x, train_flg): #순전파 계산
        if self.running_mean is None:
            N, D = x.shape
            self.running_mean = np.zeros(D)
            self.running_var = np.zeros(D)

        if train_flg:
            mu = x.mean(axis=0) #평균
            xc = x - mu #편차
            var = np.mean(xc ** 2, axis=0) #분산
            std = np.sqrt(var + 10e-7) #표준편차 0이 안나오게 입실론값 10e-7
            xn = xc / std #표준화된 변수

            self.batch_size = x.shape[0]
            self.xc = xc
            self.xn = xn
            self.std = std
            self.running_mean = self.momentum * self.running_mean + (1 - self.momentum) * mu
            self.running_var = self.momentum * self.running_var + (1 - self.momentum) * var
        else:
            xc = x - self.running_mean
            xn = xc / ((np.sqrt(self.running_var + 10e-7)))

        out = self.gamma * xn + self.beta #gamma 만큼 확대, beta만큼 이동
        return out

    def backward(self, dout):
        if dout.ndim != 2:
            N, C, H, W = dout.shape
            dout = dout.reshape(N, -1)

        dx = self.__backward(dout)

        dx = dx.reshape(*self.input_shape)
        return dx

    def __backward(self, dout): #역전파 계산
        dbeta = dout.sum(axis=0)
        dgamma = np.sum(self.xn * dout, axis=0)
        dxn = self.gamma * dout
        dxc = dxn / self.std
        dstd = -np.sum((dxn * self.xc) / (self.std * self.std), axis=0)
        dvar = 0.5 * dstd / self.std
        dxc += (2.0 / self.batch_size) * self.xc * dvar
        dmu = np.sum(dxc, axis=0)
        dx = dxc - dmu / self.batch_size

        self.dgamma = dgamma
        self.dbeta = dbeta

        return dx

class Model:
    """
    네트워크 모델 입니다.

    """
    def __init__(self, lr=0.01):
        """
        클래스 초기화
        """

        self.params = {}
        self.layers = {}
        self.__init_weight()
        self.__init_layer()
        self.optimizer = SGD(lr)

    def __init_layer(self):
        """
        레이어를 생성하시면 됩니다.
        """
        w1, w2, w3 = self.params['w1'], self.params['w2'], self.params['w3']  # weight값 지정
        b1, b2, b3 = self.params['b1'], self.params['b2'], self.params['b3']  # bias값 지정

        #self.layers = {}
        self.layers['layer1'] = Affine(w1, b1) #Hidden Layer1의 Affine 계층
        self.layers['layer1_Relu'] = Relu() #Hidden Layer1의 Relu 계층
        self.layers['layer1_Batch'] = BatchNormalization(1, 0) #Hidden Layer1의 BatchNomalization 계층
        self.layers['layer2'] = Affine(w2, b2) #Hidden Layer2의 Affine 계층
        self.layers['layer2_Relu'] = Relu() #Hidden Layer2의 Relu 계층
        self.layers['layer2_Batch'] = BatchNormalization(1, 0) #Hidden Layer2의 BatchNomalization 계층
        self.layers['layer3'] = Affine(w3, b3) #Hidden Layer3의 Affine 계층
        self.last_layer = SoftmaxWithLoss() # 마지막 계층으로 loss값과 predict 값을 구한다.

    def __init_weight(self):
        """
        레이어에 탑재 될 파라미터들을 초기화 하시면 됩니다.
        """
        self.input_size, self.output_size = 6, 6
        self.hidden_size = 150
        self.params['w1'] = np.random.randn(self.input_size, self.hidden_size) * np.exp(2 / self.hidden_size) #He 초깃값 이용
        self.params['w2'] = np.random.randn(self.hidden_size, self.hidden_size) * np.exp(2 / self.hidden_size) #He 초깃값 이용
        self.params['w3'] = np.random.randn(self.hidden_size, self.output_size) * np.exp(2 / self.hidden_size) #He 초깃값 이용
        self.params['b1'] = np.zeros(self.hidden_size)
        self.params['b2'] = np.zeros(self.hidden_size)
        self.params['b3'] = np.zeros(self.output_size)

    def update(self, x, t):
        """
        train 데이터와 레이블을 사용해서 그라디언트를 구한 뒤
         옵티마이저 클래스를 사용해서 네트워크 파라미터를 업데이트 해주는 함수입니다.

        :param x: train_data
        :param t: test_data
        """
        grads = self.gradient(x, t)
        self.optimizer.update(self.params, grads) #SGD 이용

    def predict(self, x):
        """
        데이터를 입력받아 정답을 예측하는 함수입니다.

        :param x: data
        :return: predicted answer
        """

        for layer in self.layers.values():
            x = layer.forward(x)
        return x

    def loss(self, x, t):
        """
        데이터와 레이블을 입력받아 로스를 구하는 함수입니다.
        :param x: data
        :param t: data_label
        :return: loss
        """
        y = self.predict(x)
        return self.last_layer.forward(y, t)

    def gradient(self, x, t):
        """
        train 데이터와 레이블을 사용해서 그라디언트를 구하는 함수입니다.
        첫번째로 받은데이터를 forward propagation 시키고,
        두번째로 back propagation 시켜 grads에 미분값을 리턴합니다.
        :param x: data
        :param t: data_label
        :return: grads
        """

        # forward
        self.loss(x, t)

        # backward
        # 순전파의 반대 방향으로 backward를 이용하여 미분값을 구하고 업데이트한다.
        dout = 1
        back_y = self.last_layer.backward(dout) #softmax_loss 계층
        back_3 = self.layers['layer3'].backward(back_y) #Affine3 계층
        back_Relu2 = self.layers['layer2_Relu'].backward(back_3) #Relu2 계층
        back_Batch2 = self.layers['layer2_Batch'].backward(back_Relu2) #BatchNorm2 계층
        back_2 = self.layers['layer2'].backward(back_Batch2) #Affine2 계층
        back_Relu1 = self.layers['layer1_Relu'].backward(back_2) #Relu1 계층
        back_Batch1 = self.layers['layer1_Batch'].backward(back_Relu1) #BatchNorm1 계층
        back_1 = self.layers['layer1'].backward(back_Batch1) #Affine1 계층

        grads = {} #gradient 값들 업데이트
        grads['w1'] = self.layers['layer1'].dW
        grads['b1'] = self.layers['layer1'].db
        grads['w2'] = self.layers['layer2'].dW
        grads['b2'] = self.layers['layer2'].db
        grads['w3'] = self.layers['layer3'].dW
        grads['b3'] = self.layers['layer3'].db
        return grads

    def save_params(self, file_name="params.pkl"):
        """
        네트워크 파라미터를 피클 파일로 저장하는 함수입니다.

        :param file_name: 파라미터를 저장할 파일 이름입니다. 기본값은 "params.pkl" 입니다.
        """
        params = {}
        layers = {}
        for key, val in self.params.items():
            params[key] = val
        for key, val in self.layers.items():
            layers[key] = val
        with open(file_name, 'wb') as f:
            pickle.dump(params, f)
            pickle.dump(layers, f)

    def load_params(self, file_name="params.pkl"):
        """
        저장된 파라미터를 읽어와 네트워크에 탑재하는 함수입니다.

        :param file_name: 파라미터를 로드할 파일 이름입니다. 기본값은 "params.pkl" 입니다.
        """
        with open(file_name, 'rb') as f:
            params = pickle.load(f)
            layers = pickle.load(f)
        for key, val in params.items():
            self.params[key] = val
        for key, val in layers.items():
            self.layers[key] = val
        pass



