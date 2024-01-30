# RAX  


## RAX Introduction

RAX is a porting complexity assessment tool for RISC-V. It provides automated tools to quickly determine the difficulty of software architecture porting, and recommends software to developers that adapts to their development capabilities and existing resources. For large-scale porting adaptation work has practical significance.RAX video demo at https://youtu.be/g_e4VhG4kkM.

1、Overview and Requirements of RAX Tools
----------

Support riscv architecture porting complexity assessment for software source code packages primarily targeting C and C++languages, demonstrating three levels of porting complexity to users

* Low porting complexity represents software：sqlite、gzip、patch、kmod
* Medium porting complexity represents software：openssl、freertos、nuttx
* High porting complexity represents software：opencv、gcc、glibc、freebsd

RAX consists of two modules, the SourceScanner module and the ModelTrainAndPredict module
```
1. SourceScanner：
    - Arch_code scanning function
    - System level Cyclomatic Complexity scanning function
2. ModelTrainAndPredict：
    - Training
    - Prediction-Random Forest Machine Learning Model
```

The software system cyclomatic complexity scanning function is applicable to software written in other languages, including the following:
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

The Arch_code scanning function obtains the potential workload of the following architecture binding code:
```
1.  Conditional Compilation and Architecture Macro
     -  ^#\s*(ifdef|ifndef|if|elif|if defined)\b.*\n(?:.*\n)*?#\s*endif\b.*\n?
2.  Inline Assembly	
     -  [__]*?asm[__]*?\s*[__]*[volatile]*[__]*\s*[(|{]
3.  Assembly	
     -  ^.*\.(s|S|asm|ASM)$
4.  Intrinsic	
     -  \b_mm_\w+\b|\b__m\d{2,4}\b|\b_mm\d{2,4}_\w+\b|\b_m_\w+\b|\b_tile_\w+\b
5.  Builtin	
     -  \b([builtin_names])\b
6.  System Call	
     -  \b([x86_syscall_names])\s*\(
7.  Build Scripts	
     -  \b(x86_[0-9]{0,2}|i[2345678]86|amd64|x86-[0-9]{0,2}|i\[3456789\]86)
```

2、Downloading
------------

### Environmental preparation and deployment

#### hardware disposition

```bash
In terms of system hardware deployment environment, the system has low requirements for computer hardware. It is recommended to configure the following hardware configuration
      CPU        4 Cores
      Memory         8 GB
      Disk        500 GB
```
#### Deployment


|  Software   | Minimum version |
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


#### Or
```bash
pip3 install pandas scipy xgboost scikit-learn joblib numpy imbalanced-learn
```

#### Download source code
```bash
git clone https://github.com/wangyuliu/RAX-2024.git
```

#### After downloading RAX, by setting the software address parameters you need to evaluate, you can use this tool smoothly on platforms such as PyCharm



3、QSG
------------

### 1、Porting complexity vector scanning module

#### You can choose to directly obtain the scanning and porting complexity vector, or separately obtain the software system cyclomatic complexity results and architecture binding code Arch_ Code result

Software address configuration:

Firstly, determine the local address of the software to be evaluated, such as D: \ porting \ projectone. Then, set the address in /RAX/SourceScanner/misson.py, which can set the architecture binding code Arch_ Scan target for code

```shell
root_dir = "D:/porting/"  #directory

sub_dirs = ['projectone']  #software package name
```

Set parameters for /RAX/Source Scanner/lizard. py, such as D: \ porting \ projectone; If you are using pycharm, in pycharm, select the Script path as / RAX/Source Scanner/lizard. py through the RUN Edit Configuration, select Parameters as D: \ porting \ projectone. This method sets the scanning target of the cyclomatic complexity module

```shell
Script path ： ../RAX/SourceScanner/lizard.py

Parameters  ： D:\porting\projectone
```

Set parameters in /RAX/SourceScanner/settings. py ,set the address of the porting complexity vector result

```shell
INTRINSIC_PATH = "./data/intrinsics.json"
BUILTIN_PATH = "./data/x86-builtins-names.txt"
X86_MACROS = "./data/x86-macros.txt"
SYSCALL_PATH = "./data/x86-syscall.txt"
SAVE_PATH_1 = "./data/result/1.csv"  # # Keep the numeric type and make it a vector 
SAVE_PATH_2 = "./data/result/2.csv"  # # Keep the text type and make it text
```

Also, pay attention to setting the result address for cyclomatic complexity scanning in lizard.py line:836

The running results can be obtained from the address set in the settings. py file, which is the porting complexity vector. At the same time, the intermediate results of the cyclomatic complexity scanning function can be displayed, showing parameters such as NLOC, CCN, token, PARAM, length, etc. The prefix is the cyclomatic complexity result at the function level, and the suffix is the specific file where the function is located. Finally, the system level cyclomatic complexity statistics results are displayed

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
[scanning Assembly]
[scanning BuildScrpit]
[scanning Builtin]
[scanning Conditional Compilation statements]
[scanning Intrinsic]
[scanning Syscall]
```



### 2、Train

#### You can choose to customize the dataset by adding, deleting, and modifying data, and retrain it, or you can choose to directly use the training model provided by RAX. The existing model is located at \RAX \ ModelTrainAndPredict \ data \ models \ rf_ Model_ bs.joblib, where rf_ Model_ ak.joblib is a training model for comparison tools

The location of the RAX dataset is/ RAX/ModelTrainAndPredict/data/dataset/train-test-merged-new.csv, where train-test-merged.csv is the dataset from the old version of the tool. You can also obtain the new dataset in RAX from the GitHub repository and perform the required training. The training was completed in ModelTrainAndPredict. py under the directory, and RAX ultimately chose a random forest machine learning model to complete the classification work.

```bash
        Random Forest：
        n_estimators：The number of trees. Increasing the number of trees can improve the accuracy of the model, but it will increase computation time and memory consumption.
        max_depth：The maximum depth of the tree. Increasing the maximum depth of the tree can improve the accuracy of the model, but it will increase the risk of overfitting.
        min_samples_split：The minimum number of split samples. Increasing the minimum split sample size can reduce the risk of overfitting, but it may decrease the accuracy of the model.
        min_samples_leaf：The minimum number of leaf node samples. Increasing the minimum number of leaf node samples can reduce the risk of overfitting, but it may decrease the accuracy of the model.
        SVM：
        kernel：Kernel function. Different kernel functions have different effects, such as linear kernels, polynomial kernels, radial basis kernels, etc. Choosing the appropriate kernel function can improve the accuracy of the model.
        C： Penalty coefficient. Increasing the penalty coefficient can reduce the risk of overfitting, but it may also decrease the accuracy of the model.
        gamma：Kernel function coefficients. Increasing gamma can improve the accuracy of the model, but it may increase the risk of overfitting.
        degree：The degree of a polynomial kernel function. Increasing the degree of polynomial kernel function can improve the accuracy of the model, but it may increase the risk of overfitting.
        Adaboost：
        base_estimator：Base classifier. Different base classifiers have different impacts, such as decision trees, neural networks, etc. Choosing a suitable base classifier can improve the accuracy of the model.
        learning_rate：Learning rate. Reducing the learning rate can improve the stability of the model, but it may decrease the accuracy of the model.
        n_estimators： The number of iterations. Increasing the number of iterations can improve the accuracy of the model, but it will increase computation time and memory consumption.
        Xgboost：
        n_estimators：The number of trees. Increasing the number of trees can improve the accuracy of the model, but it will increase computation time and memory consumption.
        max_depth：The maximum depth of the tree. Increasing the maximum depth of the tree can improve the accuracy of the model, but it will increase the risk of overfitting.
        learning_rate：Learning rate. Reducing the learning rate can improve the stability of the model, but it may decrease the accuracy of the model.
        gamma：Regularization coefficient. Increasing the regularization coefficient can reduce the risk of overfitting, but it may decrease the accuracy of the model.
        It should be noted that the impact of different parameters on model performance may interact with each other, so it is necessary to comprehensively consider the impact of adjusting parameters and use methods such as cross validation to evaluate model performance.
```


### 3、Predicting the complexity of software package porting

By executing model loading tools or using existing models for prediction, while paying attention to selecting the corresponding dataset source.


```
class Predictor:

    def __init__(self, project_path, project_name=None):
        # os.chdir("/Users/jimto/PycharmProjects/RV-Estimator/ModelTrainAndPredict/")
        # loading model rf_model_ak.joblib-other tool   rf_model_ak.joblib-RAX
        MODEL_PATH = "./data/models/rf_model_bs.joblib"
    ......

    def load_data(self, new_vec):
        # reading dataset ./data/dataset/train-test-merged.csv---dataset of tool Du 
        #./data/dataset/train-test-merged-new.csv---dataset of RAX
        data = pd.read_csv('./data/dataset/train-test-merged-new.csv',encoding='gbk')
    ......

    def predict(self):
        # Splicing the predicted porting complexity vector here and normalizing it with the dataset to obtain the prediction result
        lengths = [0,0,0,0,0,0,0,0]
```



4、How to contribute
----------
We warmly welcome new contributors to join the project and are delighted to provide guidance and assistance to them.
