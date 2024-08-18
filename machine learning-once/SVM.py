import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.metrics import make_scorer, accuracy_score
from sklearn.svm import SVC
# 定义SVM参数网格
param_grid_svm = {
    'C': [0.1, 1, 10, 100],  # 误差项的惩罚参数
    'gamma': ['scale', 'auto'],  # 核函数的系数
    'kernel': ['linear', 'rbf', 'poly', 'sigmoid'],  # 核函数类型
    # 可以添加其他SVM参数，但请注意不是所有参数都适合网格搜索
}

# 定义评分函数，这里使用准确度
scoring = make_scorer(accuracy_score)
# 读取Excel文件
D = pd.read_excel('CD.xlsx',header=None,skiprows=[0])
F=pd.read_excel('CD-label.xlsx')
# 创建一个 LabelEncoder 对象
le = LabelEncoder()

# 对 target 进行编码，将其转换为数值型变量
target_encoded = le.fit_transform(F)
# 将特征名称转换为字符串类型
D.columns = [str(col) for col in D.columns]
# 初始化列表来存储准确率和预测结果
accuracies = []
# 初始化存储预测结果和概率的列表

predictions = []

probabilities = []
# 初始化列表来存储SVM的结果

svm_results = []

test_svm_results = []


# 循环遍历每行
for i in range(D.shape[0]):
    D1 = D.copy()

    F1 = F.copy()
    # 计算要去掉的行的索引
    start_row, end_row = i, i + 1
    testD1 = D1.iloc[i]
    testD1_array = testD1.values
    testD1= testD1_array.reshape(1, -1)
    testF1 = F1.iloc[i]

    # 去掉一行数据
    reduced_D1 = np.delete(D1, i, axis=0)
    reduced_F1 = np.delete(F1, i, axis=0)
    # 创建PCA对象并拟合去除当前行后的数据

    pcaD1 = PCA(n_components=10)

    pca_reduced_D1 = pcaD1.fit_transform(reduced_D1)

    # 对测试集数据应用相同的PCA变换
    pca_testD1 = pcaD1.transform(testD1)

    # 使用GridSearchCV进行参数优化和交叉验证（SVM）
    svm_grid_search = GridSearchCV(SVC(), param_grid_svm, scoring=scoring, cv=10)
    svm_grid_search.fit(pca_reduced_D1, reduced_F1)

    # 获取最佳参数
    best_params_svm = svm_grid_search.best_params_

    # 创建SVM分类器，并启用概率估计

    svm = SVC(**best_params_svm, probability=True)
    svm.fit(pca_reduced_D1, reduced_F1)

    # 存储结果
    svm_results.append(pca_reduced_D1)
    test_svm_results.append(pca_testD1)
    # 对当前行进行分类预测
    prediction = svm.predict(pca_testD1.reshape(1, -1))  # 确保是二维的

    probability = svm.predict_proba(pca_testD1.reshape(1, -1))  # 确保是二维的

    # 确保probability是二维的
    if len(probability.shape) != 2:
        raise ValueError("Probability array is not 2-dimensional. Shape: ", probability.shape)

        # 将预测结果和概率添加到列表中
    predictions.append(prediction[0])  # 因为predict返回的是二维数组，即使只有一个预测，我们也要取第一个元素
    probabilities.append(probability[0])  # 同样，我们只取第一个元素，因为probability现在应该是二维的

    # ...
    # 循环结束后，将列表转换为DataFrame

    # 将预测结果转换为DataFrame
    predictions_df = pd.DataFrame(predictions, columns=['Prediction'])

    # 概率已经是二维的，所以可以直接转换为DataFrame
    probabilities_df = pd.DataFrame(probabilities)

    # 合并预测结果和概率的DataFrame
    results_df = pd.concat([predictions_df, probabilities_df], axis=1)

    # 如果需要，添加行索引（通常，DataFrame会自动为每行分配一个索引）
    # results_df['Row_Index'] = range(results_df.shape[0])  # 这行通常不是必要的，除非你有特定的索引需求

    # 保存为Excel文件
    results_df.to_excel('CD-SVM.xlsx', index=False)









