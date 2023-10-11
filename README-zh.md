# RAX  


## RAX介绍

RAX是一种面向RISC-V的移植复杂度评估工具，提供自动化工具以快速确定软件的架构移植难度，为开发者推荐适配其开发能力和现有资源的软件，对于大规模移植适配工作具有实际意义。


一、RAX工具概览及要求
----------

支持面向C、C++语言为主的软件源码包的riscv架构移植复杂度评估，为使用者展示三个等级的移植复杂难度

* 低移植复杂度代表软件包括：sqlite、gzip、patch、kmod
* 中移植复杂度代表软件包括：openssl、freertos、nuttx
* 高移植复杂度代表软件包括：opencv、gcc、glibc、freebsd

RAX包含两个模块，SourceScanner模块以及ModelTrainAndPredict模块
```
1. SourceScanner：
    - Arch_code扫描功能
    - 软件系统圈复杂度扫描功能
2. ModelTrainAndPredict：
    - 训练
    - 预测-使用随机森林机器学习模型
```

其中软件系统圈复杂度扫描功能适用于其他语言编写的软件，包括如下：
```
-  C/C++ (works with C++14)
-  Java
-  C# (C Sharp)
-  JavaScript (With ES6 and JSX)
-  TypeScript
-  Objective-C
-  Swift
-  Python
-  Ruby
-  TTCN-3
-  PHP
-  Scala
-  GDScript
-  Golang
-  Lua
-  Rust
-  Fortran
-  Kotlin
-  Solidity
-  Erlang
```

其中Arch_code扫描功能获取如下架构绑定代码的潜在工作量：
```
1.   条件编译语句	
     -  ^#\s*(ifdef|ifndef|if|elif|if defined)\b.*\n(?:.*\n)*?#\s*endif\b.*\n?
2.  内联汇编	
     -  [__]*?asm[__]*?\s*[__]*[volatile]*[__]*\s*[(|{]
3.  全汇编	
     -  ^.*\.(s|S|asm|ASM)$
4.  Intrinsic	
     -  \b_mm_\w+\b|\b__m\d{2,4}\b|\b_mm\d{2,4}_\w+\b|\b_m_\w+\b|\b_tile_\w+\b
5.  Builtin	
     -  \b([builtin_names])\b
6.  系统调用	
     -  \b([x86_syscall_names])\s*\(
7.  构建脚本	
     -  \b(x86_[0-9]{0,2}|i[2345678]86|amd64|x86-[0-9]{0,2}|i\[3456789\]86)
```

二、下载安装
------------

### 环境准备与部署

#### 硬件部署

```bash
系统硬件部署环境方面，系统对计算机硬件要求较低，建议如下硬件配置
      CPU        4 Cores
      内存         8 GB
      磁盘        500 GB
```
#### 软件部署


|  软件   | 最低版本号  |
|  ----  | ----  |
| python  | 3.7.0 |
| imbalanced-learn  | 0.11.0 |
| joblib  | 1.2.0 |
| numpy  | 1.21.6 |
| pandas  | 1.2.3 |
| pip  | 19.0.3 |
| scikit-learn  | 1.0.2 |
| scipy  | 1.7.3 |
| xgboost  | 1.6.2 |


#### 或使用
```bash
pip3 install pandas scipy xgboost scikit-learn joblib numpy imbalanced-learn
```

#### 下载源码
```bash
git clone https://github.com/wangyuliu/RAX-2024.git
```

#### 下载RAX后，通过设置您所需要评估的软件地址参数，既可以在pycharm等平台上顺利使用此工具



三、快速使用指南
------------

### 1、移植复杂度向量扫描模块

#### 可以选择直接获取扫描移植复杂度向量，或是分别获取软件系统圈复杂度扫描结果和架构绑定代码Arch_code扫描结果

待评估软件地址配置：

首先确定待评估软件的本地地址例如D:\porting\projectone，随后在/RAX/SourceScanner/misson.py中设置地址，此方法能够设置架构绑定代码Arch_code的扫描目标

```shell
root_dir = "D:/porting/"  #目录

sub_dirs = ['projectone']  #具体软件
```

为/RAX/SourceScanner/lizard.py设置参数，例如D:\porting\projectone；如果您使用pycharm，在pycharm中通过RUN-Edit Configurations中，选定Script path为../RAX/SourceScanner/lizard.py，选定Parameters为D:\porting\projectone。此方法设置圈复杂度模块的扫描目标

```shell
Script path ： ../RAX/SourceScanner/lizard.py

Parameters  ： D:\porting\projectone
```

在/RAX/SourceScanner/settings.py设置参数，设置移植复杂度向量结果地址

```shell
INTRINSIC_PATH = "./data/intrinsics.json"
BUILTIN_PATH = "./data/x86-builtins-names.txt"
X86_MACROS = "./data/x86-macros.txt"
SYSCALL_PATH = "./data/x86-syscall.txt"
SAVE_PATH_1 = "./data/result/1.csv"  # 保留数值类型，做成向量
SAVE_PATH_2 = "./data/result/2.csv"  # 保留文本类型，做成文本
```

同时注意在lizard.py当中设置圈复杂度的结果地址

可以在settings.py文件中设置的地址中获取运行结果，即移植复杂度向量，同时展示圈复杂度扫描功能的运行中间结果，分别展示出NLOC、CCN、token、PARAM、length等参数，location部分前缀为函数级别的圈复杂度结果，后缀为函数所在的具体文件，最终展示出系统级别的圈复杂度统计结果

```shell
================================================
      NLOC    CCN   token  PARAM  length  location  
      17      4    123      3      32 StripContentBetweenTags@53-84@D:\tool\stage5\abseil-cpp-master\create_lts.py
      33      4    153      1      43 main@87-129@D:\tool\stage5\abseil-cpp-master\create_lts.py
     ......
1654 file analyzed.
==============================================================
NLOC  Avg.NLOC  AvgCCN  Avg.token  function_cnt    file
--------------------------------------------------------------
 CC ----10000
```

```shell
D:\python-setup\python.exe ".\RAX-2024-main\RAX\SourceScanner\lizard.py" D:\porting\projectone
[扫描工程汇编]
[扫描构建脚本]
[扫描工程Builtin]
[扫描条件编译语句]
[扫描工程的Intrinsic]
[扫描工程的Syscall]
```



### 2、训练模块

#### 可以选择增删改数据的方式来自定义数据集，重新进行训练，也可以选择直接使用RAX提供的训练模型。已有的模型位于.\RAX\ModelTrainAndPredict\data\models\rf_model_bs.joblib，其中rf_model_ak.joblib为对比工具的训练模型

RAX的数据集位置为./RAX/ModelTrainAndPredict/data/dataset/train-test-merged-new.csv，其中train-test-merged.csv为旧版本工具的数据集，也可以在github仓库中获取RAX的新增数据集new dataset.csv，进行所需要的训练。训练在目录下的ModelTrainAndPredict.py中完成，RAX最终选择随机森林机器学习模型完成分类工作。

```bash
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
```


### 3、预测软件包移植复杂度

通过执行模型生成工具，或使用现有模型来进行预测，同时注意选择相应的数据集来源。


```
class Predictor:

    def __init__(self, project_path, project_name=None):
        # os.chdir("/Users/jimto/PycharmProjects/RV-Estimator/ModelTrainAndPredict/")
        # 加载模型 rf_model_ak.joblib-tool of du   rf_model_ak.joblib-RAX
        MODEL_PATH = "./data/models/rf_model_bs.joblib"
    ......

    def load_data(self, new_vec):
        # 读取数据集 ./data/dataset/train-test-merged.csv---dataset of tool Du 
        #./data/dataset/train-test-merged-new.csv---dataset of RAX
        data = pd.read_csv('./data/dataset/train-test-merged-new.csv',encoding='gbk')
    ......

    def predict(self):
        # 在此处拼接带预测的移植复杂度向量，与数据集进行标准化处理得到预测结果
        lengths = [0,0,0,0,0,0,0,0]
```



四、如何贡献
----------
我们非常欢迎新贡献者加入到项目中来，也非常高兴能为新加入贡献者提供指导和帮助。