import pandas as pd
import numpy as np
from itertools import combinations
from sklearn.metrics import accuracy_score, classification_report

import pandas as pd
import numpy as np
from itertools import combinations
from sklearn.metrics import accuracy_score, classification_report

# Excel文件路径
file_path = 'G4.xlsx'  # 替换为你的Excel文件路径

# 读取Excel文件，跳过第一行的文字说明
df = pd.read_excel(file_path, skiprows=0)

# 假设第一列是标签列（文字标签），其他列是分类器的预测结果
labels = df.iloc[:, 0]

predictions = df.iloc[:, 1:].values


# 计算每个分类器的准确度
classifier_accuracies = []
for i in range(predictions.shape[1]):
    clf_labels = predictions[:, i]
    clf_accuracy = accuracy_score(labels, clf_labels)
    classifier_accuracies.append(clf_accuracy)

max_accuracy = 0
best_combination = []
best_combination_votes = []
best_combination_metrics = {}

# 遍历从1到所有分类器数量的所有组合数量
for num_classifiers in range(1, predictions.shape[1] + 1):
    all_combinations = list(combinations(range(predictions.shape[1]), num_classifiers))

    for combination in all_combinations:
        selected_predictions = predictions[:, list(combination)]
        selected_weights = np.array(classifier_accuracies)[list(combination)]

        combined_predictions = []
        for preds in selected_predictions:
            label_scores = {}
            for label, weight in zip(preds, selected_weights):
                label_scores[label] = label_scores.get(label, 0) + weight
            combined_predictions.append(max(label_scores, key=label_scores.get))

        accuracy = accuracy_score(labels, combined_predictions)

        if accuracy > max_accuracy:
            max_accuracy = accuracy
            best_combination = list(combination)
            print(best_combination)
            best_combination_votes = combined_predictions
            # 如果你想要知道最佳组合具体是哪些分类器，你可以这样做：
            best_combination_classifiers = [df.columns[i+1] for i in best_combination]
            print(f"Best Combination Classifiers: {best_combination_classifiers}")


            # 计算每个类别的准确度、召回率、F1分数和精确率
            # 注意：target_names 应该是一个列表，包含所有唯一标签的名称
            unique_labels = np.unique(labels)
            report = classification_report(labels, combined_predictions, target_names=unique_labels, output_dict=True)
            best_combination_metrics = report

        # 将最佳组合的性能指标转换为DataFrame
results_df = pd.DataFrame(best_combination_metrics).T.reset_index()
results_df.columns = ['Class', 'Precision', 'Recall', 'F1-score', 'Support']

# 导出到Excel文件
output_file_path = 'G4-汇总.xlsx'
results_df.to_excel(output_file_path, index=False)

