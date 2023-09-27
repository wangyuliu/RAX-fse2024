import pandas as pd
from imblearn.over_sampling import RandomOverSampler
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.tree import DecisionTreeClassifier
import random

import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import GridSearchCV


class ModelTrainer:

    def __init__(self):
        self.scaler = StandardScaler()
        self.rf, self.ada, self.svm, self.xgb = None, None, None, None
        self.X_train, self.X_test, self.y_train, self.y_test = None, None, None, None

    # param_list = [1000, 10, 6, 10, 'poly', 1, 5.558, 5, 'DecisionTreeClassifier', 0.01, 1000, 1000, 10, 0.01, 8.684]

    def run(self, rf_nestimator, rf_depth, rf_min_split, rf_min_leaf, svm_kernel, svm_c, svm_gamma, svm_degree,
            ada_base_estimator, ada_learning_rate, ada_n_estimator, xg_n_estimator, xg_depth, xg_learning_rate,
            xg_gamma):
        # 加载数据
        x, y = self.load_data()
        # 数据预处理
        self.pre_process(x, y)

        # 创建模型
        self.create_model(rf_nestimator, rf_depth, rf_min_split, rf_min_leaf, svm_kernel, svm_c, svm_gamma, svm_degree,
                          ada_base_estimator, ada_learning_rate, ada_n_estimator, xg_n_estimator, xg_depth,
                          xg_learning_rate,
                          xg_gamma)
        # 训练模型
        self.train()
        # 测试模型
        self.test()

    def load_data(self):
        # 读取数据集
        data = pd.read_csv('E:/kuokuokuo/论文/ModelTrainAndPredict/data/dataset/train-test-merged.csv')
        # 将标签转换为数字类型
        label_map = {'低': 0, '中': 1, '高': 2}
        data['label'] = data['label'].map(label_map)
        # 切分特征和标签
        X = data.drop(['label', 'repo_name'], axis=1)
        y = data['label']
        # 对特征数据进行正则化
        # scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X)

        return X_scaled, y

    # 第二个任务在此修改
    def pre_process(self, X, y):
        # 切分训练和测试数据集
        # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
        # 按照考虑两个合并成一个，然后再把test集合分出来
        Xf = np.split(X, [62])
        X_train = Xf[0]
        X_test = Xf[1]
        yf = np.split(y, [62])
        y_train = yf[0]
        y_test = yf[1]
        # 随机过采样方法对数据进行重采样
        ros = RandomOverSampler(random_state=1234)
        X_train, y_train = ros.fit_resample(X_train, y_train)

        self.X_train, self.X_test, self.y_train, self.y_test = X_train, X_test, y_train, y_test

    def create_model(self, rf_nestimator, rf_depth, rf_min_split, rf_min_leaf, svm_kernel, svm_c, svm_gamma, svm_degree,
                     ada_base_estimator, ada_learning_rate, ada_n_estimator, xg_n_estimator, xg_depth, xg_learning_rate,
                     xg_gamma):
        # def run(self, rf_nestimator, rf_depth, rf_min_split, rf_min_leaf, svm_kernel, svm_c, svm_gamma, svm_degree,
        #         ada_base_estimator, ada_learning_rate, ada_n_estimator, xg_n_estimator, xg_depth, xg_learning_rate,
        #         xg_gamma):
        """
        随机森林：
        n_estimators：树的数量。增加树的数量可以提高模型的准确性，但是会增加计算时间和内存消耗。
        max_depth：树的最大深度。增加树的最大深度可以提高模型的准确性，但是会增加过拟合的风险。
        min_samples_split：最小分裂样本数。增加最小分裂样本数可以减少过拟合的风险，但是可能会降低模型的准确性。
        min_samples_leaf：最小叶子节点样本数。增加最小叶子节点样本数可以减少过拟合的风险，但是可能会降低模型的准确性。
        SVM：
        kernel：核函数。不同的核函数有不同的影响，如线性核、多项式核、径向基核等。选择合适的核函数可以提高模型的准确性。
        C：惩罚系数。增加惩罚系数可以减少过拟合的风险，但是可能会降低模型的准确性。
        gamma：核函数系数。增加gamma可以提高模型的准确性，但是可能会增加过拟合的风险。
        degree：多项式核函数的次数。增加多项式核函数的次数可以提高模型的准确性，但是可能会增加过拟合的风险。
        Adaboost：
        base_estimator：基分类器。不同的基分类器有不同的影响，如决策树、神经网络等。选择合适的基分类器可以提高模型的准确性。
        learning_rate：学习率。减小学习率可以提高模型的稳定性，但是可能会降低模型的准确性。
        n_estimators：迭代次数。增加迭代次数可以提高模型的准确性，但是会增加计算时间和内存消耗。
        Xgboost：
        n_estimators：树的数量。增加树的数量可以提高模型的准确性，但是会增加计算时间和内存消耗。
        max_depth：树的最大深度。增加树的最大深度可以提高模型的准确性，但是会增加过拟合的风险。
        learning_rate：学习率。减小学习率可以提高模型的稳定性，但是可能会降低模型的准确性。
        gamma：正则化系数。增加正则化系数可以减少过拟合的风险，但是可能会降低模型的准确性。
        需要注意的是，不同的参数对模型性能的影响可能会相互作用，因此需要综合考虑调整参数的影响，并使用交叉验证等方法来评估模型性能。

        随机森林：
        n_estimators：10~1000
        max_depth：1~50
        min_samples_split：2~50
        min_samples_leaf：1~50
        SVM：
        kernel：linear、poly、rbf、sigmoid
        C：0.1~100
        gamma：0.001~10
        degree：1~10
        Adaboost：
        base_estimator：决策树、神经网络等
        learning_rate：0.01~1
        n_estimators：10~1000
        Xgboost：
        n_estimators：10~1000
        max_depth：1~50
        learning_rate：0.01~1
        gamma：0.001~10
        """

        # 使用随机森林、adaboost、SVM、XGBoost四种机器学习模型进行三分类训练
        rf = RandomForestClassifier(n_estimators=rf_nestimator, max_depth=rf_depth, min_samples_split=rf_min_split,
                                    min_samples_leaf=rf_min_leaf)

        ada = AdaBoostClassifier(DecisionTreeClassifier(max_depth=10, min_samples_split=22, min_samples_leaf=6,
                                                        min_weight_fraction_leaf=0.36),
                                 algorithm="SAMME", n_estimators=ada_n_estimator, learning_rate=ada_learning_rate)
        # adaboost的主要修改是增加了DecisionTreeClassifier(max_depth=10,min_samples_split=22, min_samples_leaf=6,min_weight_fraction_leaf=0.36)
        # max_depth = 10, min_samples_split = 22, min_samples_leaf = 3, min_weight_fraction_leaf = 0.36

        svm = SVC(kernel=svm_kernel, C=svm_c, gamma=svm_gamma, degree=svm_degree)

        # 在增加了两个参数 但没有什么变化
        xgb = XGBClassifier(n_estimators=xg_n_estimator, max_depth=xg_depth, learning_rate=xg_learning_rate,
                            gamma=xg_gamma, min_child_weight=3, reg_lambda=0.45)

        self.rf, self.ada, self.svm, self.xgb = rf, ada, svm, xgb

    def train(self):
        # 训练模型
        self.rf.fit(self.X_train, self.y_train)
        self.ada.fit(self.X_train, self.y_train)
        self.svm.fit(self.X_train, self.y_train)
        self.xgb.fit(self.X_train, self.y_train)

        # 对各机器学习模型在测试集上进行十折交叉验证，并统计预测准确率、F1等指标
        rf_scores = cross_val_score(self.rf, self.X_test, self.y_test, cv=10)
        ada_scores = cross_val_score(self.ada, self.X_test, self.y_test, cv=10)
        svm_scores = cross_val_score(self.svm, self.X_test, self.y_test, cv=10)
        xgb_scores = cross_val_score(self.xgb, self.X_test, self.y_test, cv=10)

        print('随机森林准确率：', rf_scores.mean())
        print('AdaBoost准确率：', ada_scores.mean())
        print('SVM准确率：', svm_scores.mean())
        print('XGBoost准确率：', xgb_scores.mean())

    def test(self):
        # 对测试数据集进行预测，并计算准确率、F1等指标
        rf_pred = self.rf.predict(self.X_test)
        ada_pred = self.ada.predict(self.X_test)
        svm_pred = self.svm.predict(self.X_test)
        xgb_pred = self.xgb.predict(self.X_test)

        # 在没有训练的测试集里计算macro指标
        rf_acc = accuracy_score(self.y_test, rf_pred)
        ada_acc = accuracy_score(self.y_test, ada_pred)
        svm_acc = accuracy_score(self.y_test, svm_pred)
        xgb_acc = accuracy_score(self.y_test, xgb_pred)

        rf_f1 = f1_score(self.y_test, rf_pred, average='macro')
        ada_f1 = f1_score(self.y_test, ada_pred, average='macro')
        svm_f1 = f1_score(self.y_test, svm_pred, average='macro')
        xgb_f1 = f1_score(self.y_test, xgb_pred, average='macro')

        rf_precision = precision_score(self.y_test, rf_pred, average='macro')
        ada_precision = precision_score(self.y_test, ada_pred, average='macro')
        svm_precision = precision_score(self.y_test, svm_pred, average='macro')
        xgb_precision = precision_score(self.y_test, xgb_pred, average='macro')

        rf_recall = recall_score(self.y_test, rf_pred, average='macro')
        ada_recall = recall_score(self.y_test, ada_pred, average='macro')
        svm_recall = recall_score(self.y_test, svm_pred, average='macro')
        xgb_recall = recall_score(self.y_test, xgb_pred, average='macro')

        print(self.X_test.shape)

        print("rf, ada, svm, xgb")
        print("acc:", rf_acc, ada_acc, svm_acc, xgb_acc)
        print("f1-macro:", rf_f1, ada_f1, svm_f1, xgb_f1)
        print("precision-macro:", rf_precision, ada_precision, svm_precision, xgb_precision)
        print("recall-macro:", rf_recall, ada_recall, svm_recall, xgb_recall)

        print("*" * 20)
        print(list(self.y_test))
        print(list(rf_pred))
        print(list(ada_pred))
        print(list(svm_pred))
        print(list(xgb_pred))


if __name__ == '__main__':
    param_list = [1000, 10, 6, 10, 'poly', 6, 5.8, 7.3, 'DecisionTreeClassifier', 0.01, 1000, 1000, 5, 0.3, 8.5]
    trainer = ModelTrainer()
    # 1000, 10, 6, 10,'poly',6,5.8, 7.3, 'DecisionTreeClassifier', 0.01, 1000, 1000, 5, 0.3, 0.5

    trainer.run(*param_list)
