import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.linear_model import LogisticRegression

def split_data(df):
    X = df.drop(columns=['churned'])
    y = df['churned']
    # تقسيم البيانات 80/20 مع الحفاظ على التوزيع
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    return X_train, X_test, y_train, y_test

def calculate_metrics(y_true, y_pred):
    # حساب المقاييس المطلوبة: Accuracy, Precision, Recall, F1
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred)),
        "recall": float(recall_score(y_true, y_pred)),
        "f1": float(f1_score(y_true, y_pred))
    }

def run_cross_validation(X_train, y_train):
    # استخدام LogisticRegression مع موازنة الأوزان
    model = LogisticRegression(random_state=42, max_iter=1000, class_weight="balanced")
    scores = cross_val_score(model, X_train, y_train, cv=5, scoring="accuracy")
    return {
        "scores": scores,
        "mean": float(np.mean(scores)),
        "std": float(np.std(scores))
    }