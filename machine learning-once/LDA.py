from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import LeaveOneOut
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.neighbors import KNeighborsClassifier
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
# 初始化交叉验证对象
loo = LeaveOneOut()
knn = KNeighborsClassifier(n_neighbors=10)  # 假设使用10个邻居
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
    # 创建LDA对象并拟合训练集数据
    lda = LDA(n_components=1)  # 设置您想要的组件数
    lda.fit(pca_reduced_D1, reduced_F1)  # 使用特征和标签来拟合LDA模型


    # 对当前行进行分类预测
    prediction = lda.predict(pca_testD1.reshape(1, -1))  # 确保是二维的

    probability = lda.predict_proba(pca_testD1.reshape(1, -1))  # 确保是二维的

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
    results_df.to_excel('CD-LDA.xlsx', index=False)









