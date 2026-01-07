import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn. preprocessing import StandardScaler
from sklearn. pipeline import Pipeline
from sklearn.model_selection import cross_val_score
import lightgbm as lgb

def build_classifiers():
    """
    构建多个分类器进行对比
    """
    classifiers = {
        'KNN': KNeighborsClassifier(n_neighbors=7, weights='distance', metric='manhattan'),
        
        'SVM_RBF': Pipeline([
            ('scaler', StandardScaler()),
            ('svm', SVC(kernel='rbf', C=10, gamma='scale'))
        ]),
        
        'RandomForest': RandomForestClassifier(
            n_estimators=200,
            max_depth=15,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42
        ),
        
        # ✅ 替换为 LightGBM：更快、更强、支持类别特征、自动处理缺失值
        'LightGBM': lgb.LGBMClassifier(
            n_estimators=200,        # 树的数量（可减少到 100 加速）
            max_depth=8,             # 控制过拟合（LightGBM 的 depth 不是 exact）
            num_leaves=63,           # ≈ 2^max_depth，但更灵活
            learning_rate=0.1,
            subsample=0.8,           # 行采样
            colsample_bytree=0.8,    # 列采样
            random_state=42,
            n_jobs=-1,               # ← 关键！启用所有 CPU 核心
            verbose=-1               # 静默模式
        )
    }
    
    return classifiers


def train_and_evaluate(classifiers, X_train, y_train, X_test, y_test):
    """
    训练并评估所有分类器
    """
    results = {}
    
    for name, clf in classifiers. items():
        print(f"Training {name}...")
        clf.fit(X_train, y_train. ravel())
        
        # 预测
        y_pred = clf.predict(X_test)
        accuracy = np.mean(y_pred == y_test. ravel()) * 100
        
        results[name] = accuracy
        print(f"{name} Accuracy: {accuracy:.2f}%")
    
    return results