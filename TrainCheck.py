import sys
import os
from collections import OrderedDict
import pickle
import numpy as np
sys.path.append(os.pardir)  # 부모 디렉터리의 파일을 가져올 수 있도록 설정
from AReM import *
from model import *

with open('params.pkl', 'rb') as f:
    data = pickle.load(f)

print(data)