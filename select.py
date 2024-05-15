import pandas as pd
from sklearn.linear_model import LassoCV, Lasso
from sklearn.feature_selection import SelectFromModel
from sklearn.preprocessing import StandardScaler


df = pd.read_excel('output_processed.xlsx')
# 分离特征和标签
X = df.iloc[:, 1:].values
y = df.iloc[:, 0].values
feature_names = df.columns[1:]

# 特征标准化
scaler = StandardScaler().fit(X)
X_scaled = scaler.transform(X)

alpha = 0.01  # 需要根据你的数据调整这个值
lasso = Lasso(alpha=alpha, max_iter=10000)
lasso.fit(X_scaled, y)
model = SelectFromModel(lasso, prefit=True)
X_selected = model.transform(X_scaled)
selected_features = feature_names[model.get_support()]

# 打印被选择的特征名
print("Selected features:", selected_features)

# 如果你想将这些特征名保存到文件
selected_features.to_series().to_csv('selected_features.csv', index=False)