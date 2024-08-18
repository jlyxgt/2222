from sklearn.model_selection import LeaveOneOut
import numpy as np
from sklearn.decomposition import PCA
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.metrics import make_scorer, accuracy_score

# 定义您想要搜索的参数网格

param_grid = {

    'n_neighbors': range(1, 10),  # 邻居数量

    'weights': ['uniform', 'distance'],  # 权重类型

    'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute'],  # 算法类型

    'leaf_size': range(10, 50, 10),  # 叶大小

    'p': [1, 2],  # 距离度量的幂 (对于闵可夫斯基距离)

    # 还可以添加其他参数，如metric, metric_params等，但请注意不是所有参数都适合网格搜索

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
# 初始化列表来存储每次迭代的结果

pca_results = []

test_pca_results = []
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
    # 存储结果

    pca_results.append(pca_reduced_D1)

    test_pca_results.append(pca_testD1)

    # 使用GridSearchCV进行参数优化和交叉验证

    knn_grid_search = GridSearchCV(KNeighborsClassifier(), param_grid, scoring=scoring, cv=10)# 假设使用5折交叉验证

    knn_grid_search.fit(pca_reduced_D1, reduced_F1)

    # 获取最佳参数

    best_params = knn_grid_search.best_params_

    # 使用最佳参数创建KNN分类器

    knn = KNeighborsClassifier(**best_params)

    knn.fit(pca_reduced_D1, reduced_F1)
    # 对当前行进行分类预测
    prediction = knn.predict(pca_testD1.reshape(1, -1))  # 确保是二维的

    probability = knn.predict_proba(pca_testD1.reshape(1, -1))  # 确保是二维的

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
    results_df.to_excel('CD-KNN.xlsx', index=False)









