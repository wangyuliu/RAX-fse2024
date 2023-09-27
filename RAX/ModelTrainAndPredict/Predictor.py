"""
这个Predictor首先调用SourceScanner模块，扫描出工程的特征向量
然后利用RF模型，给出预测结果
"""
import os
import sys

import numpy as np
import pandas as pd
from joblib import load
from sklearn.preprocessing import StandardScaler

from SourceScanner.ScannerMain import Scanner


class Predictor:

    def __init__(self, project_path, project_name=None):
        # os.chdir("/Users/jimto/PycharmProjects/RV-Estimator/ModelTrainAndPredict/")
        # 加载模型 rf_model_ak.joblib-tool of du   rf_model_ak.joblib-RAX
        MODEL_PATH = "./data/models/rf_model_ak.joblib"
        self.model = load(MODEL_PATH)
        # new标准化器
        self.scaler = StandardScaler()
        # 加载扫描器
        self.scanner = Scanner(project_path, project_name)

    def standardize(self, vector):
        X = self.load_data(vector)
        # 对特征数据进行正则化
        X_scaled = self.scaler.fit_transform(X)

        new_vec = X_scaled[-1]
        return new_vec

    def load_data(self, new_vec):
        # 读取数据集 ./data/dataset/train-test-merged.csv---dataset of tool Du ./data/dataset/train-test-merged-new.csv---dataset of RAX
        data = pd.read_csv('./data/dataset/train-test-merged.csv',encoding='gbk')
        # 切分特征和标签
        X = data.drop(['label', 'repo_name'], axis=1)

        # 拼接新向量到尾部
        new_df = pd.DataFrame([new_vec], columns=X.columns)
        X = pd.concat([X, new_df], ignore_index=True)

        return X

    def predict(self):
        # 先扫描
        # print("=" * 20, "Scan Codes", "=" * 20)
        # vector = self.scanner.scan()
        # vector=[pcl-master,0,9,0,0,75,729,296]
        # 整理成list
        # # # glfw-master,0,0,0,0,9,3,15,10688低
        # # # pcl-master,0,9,0,0,77,729,296,95312中
        # # # gstreamer-main,3448,1446,0,1,164,1091,251,289775高
        # glfw, 0, 0, 0, 0, 9, 3, 15,��
        # pcl, 0, 9, 0, 0, 77, 729, 296,��        # nss 57996,1480,0,0,51,4275,155,86173
        # opencv 36687,7424,3,11,172,12688,670,239925
        # sqlite0,13,66,0,14,46,122,62920
        # vlc-master,4430,299,36,0,28,428,829,114944
        # 176,2962,74,0,9,20,136 r
        # 2968,1648,89,0,144,272,613
        # 0, 23, 0, 5, 69, 96, 1505
        # 0,52,52,0,16,27,108
        # qt 4085, 3533, 0, 11, 385, 3445, 3158, 383142
        # rt thread 448999,36690,59,4,37,594,444,754237
        # ucos 34110,180,0,0,0,0,0,3894
        # buildroot 0,0,122,0,2,0,38,5169
        # chibios 18229,799,0,1,1,10,17,22008
        # tiny c 1933,1078,14,6,237,73,43,14839
        # freeRTOS-kernel 36123,36541,0,0,3,3,0,5710
        # zephyr 9971,2302,0,6,224,70,248,197308
        # contiki 4478,1537,0,0,29,28,67,48835
        # u-boot 23437,6283,0,7,94,62,210,231102
        # xen 9894,3255,56,334,408,340,337,93342
        # nuttx 36983,2653,0,9,54,91,340,204446
        # qemu 15141,7445,58,5,398,974,665,243747
        # libpng 235,72,56,0,3,88,11,9895
        # redis 0,21,47,0,42,80,178,41629
        # openblas 1489836,26964,0,9,102,7675,16,449854
        # libtiff 0,0,0,0,3,6,49,15752
        #  fftw 0,155,0,0,26,439,4,6037
        # libcurl 0,2,4,2,8,5,194,31194
        # openmpi 0,272,53,0,83,340,203,
        # wireshark 0,11,1,0,4,181,547,374457
        # minix 276883,5364,830,1092,635,5393,7716,980373
        # core boot 10548,1729,1,16,80,407,147,117057
        # rtems 51534,10604,0,2,60,61,1239,120725
        # gtk+ 0,106,0,4,4,740,19,94624
        # grub 15659,1338,91,0,190,145,97,41662
        # zhong 122531	1	8	0	11	3	224	39705
        # eigen 0,175,0,3,39,3657,51,42350
        # boost-math 0,14,0,0,32,152,19,32519
        # gsl 0,5,62,0,8,7,39,30268
        # moosefs 0,0,53,0,0,0,365,17032
        # vtk 0,290,1,4,120,3005,957,532559
        # flann 0,0,0,3,2,1,4,2856 高维空间快速搜索库
        # ITK 跨平台工具包 n维科学图像处理290,315,6,17,121,3848,466,215393
        # superlu 0,0,0,0,0,0,3,11943
        # petsc 0,2996,0,0,34,1046,47,133979
        # gmpy 0,0,0,0,0,0,0,4562
        # cgal 0,40,0,0,29,1091,1473,184148
        # trilinos 0,71,36,12,99,1105,3851,652472
        # kokkos 0,32,0,6,14,330,113,24369
        # halide5,42,0,3,31,743,1435,52801
        # libav  64962,6911,37,0,116,190,230,90634
        # hip 0,5,0,0,0,0,61,6834
        # flac 0,14,14,0,57,3295,17,13811
        # arrayfire 0,4,0,6,2,6,57,21922
        # portmidi 0,0,0,0,0,0,2,1850
        # openbsd-src 139043,31261,1430,2219,958,37968,7816,2515048
        # xgboost 0,1,0,2,3,10,53,14311
        # TNT 0,0,0,0,0,0,0,699
        # fuchsia 303241,1891,0,27,541,278,4487,744079
        # libsndfile 0,3,3,0,3,15,157,11323
        # libsimdpp 0,25,0,8,36,2620,0,9111
        # freebsd 552140,21810,1066,763,2076,26125,12473,2540938
        # libusb 0,0,0,0,0,0,42,4754
        # threadX 269648,3149,0,0,4,0,2,17967
        # bullet3 0,289,0,17,75,4999,307,100143
        # ogre 0,51,0,0,12,1059,103,61909
        # openal-soft 0,7,0,3,10,253,87,11915
        # caffe 0,0,0,0,2,0,48,9188
        # armnn 15,8,0,0,9,133,65,29387
        # apache mesos 0,1,4,0,5,2,1952,47975
        # kicad 4299,2620,96,5,441,365,745,185598
        # openvdb 0,0,0,4,10,44,206,44542
        # openexr 0,423,2,0,59,700,30,16536
        # assimp 3619,843,0,0,35,258,270,59642
        # oneTBB 0,59,0,0,53,121,60,21634
        # nanomsg 0,0,0,0,0,0,11,3246
        # tesseract 0,1,1,0,13,284,41,28587
        # cryengine 0,300,2,4,26,3103,1004,504983
        # wxWidgets 0,12,57,0,31,17,577,143656
        # armadillo 0,9,0,0,4,1,59,23744
        # abseil 0,76,0,8,76,164,42,21935
        # libzmq 0,61,0,0,6,17,57,9381
        # libgit2 0,9,0,0,11,11,73,42841
        # kde-plasma 0,5877,0,0,0,0,16,3024
        # mongo 8667,36215,544,33,1011,4798,2285,968394
        # postgresql 81,212,48,3,24,193,552,154736
        # sdl-mixer 0,47,0,0,21,511,7,7502
        # liquidDSP 12,4,50,0,1,548,41,15533
        # poco 0,57,0,0,28,79,535,78793
        # tinyXML2 0,0,0,0,1,0,5,2423
        # gRPC 0,5,0,0,29,2449,136,141807
        # SFML 0,23,0,0,12,207,129,14173
        # clanlib 0,6,0,0,15,921,104,32676
        # irrlicht 355,38,49,0,4,1,31,44959
        # portaudio 0,436,87,0,6,1,33,8524
        # MRPT 0,0,0,10,32,306,355,80843
        # fluidsynth 0,2,0,0,0,0,10,6389
        # drake 0,0,1,2,5,101,169,54086
        # ardupilot 318,170,0,15,3,23,319,95303
        # 在此处拼接带预测的移植复杂度向量，与数据集进行标准化处理得到预测结果
        lengths = [0,0,1,2,5,101,169]
        # 将列表转换成numpy向量
        vector = np.array(lengths)
        # print("=" * 20, "Standardiza", "=" * 20)
        # 做标准化（加载数据、添加到最后一行、标准化、抽出最后一行）
        new_vector = self.standardize(vector)

        # print("=" * 20, "Predict", "=" * 20)
        # 预测
        res = self.model.predict([new_vector])[0]
        return res

if __name__ == '__main__':
    project_path = "E:/kuokuokuo/"
    project_name = ""
    predictor = Predictor(project_path, project_name)
    result =predictor.predict()

    # project_path, project_name = sys.argv[1], sys.argv[2]
    # predictor = Predictor(project_path, project_name)
    # result = predictor.predict()

    if result == 0:
        print(f"工程{project_name}移植复杂度：低")
    elif result == 1:
        print(f"工程{project_name}移植复杂度：中")
    elif result == 2:
        print(f"工程{project_name}移植复杂度：高")

