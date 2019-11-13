import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report, \
    confusion_matrix, roc_curve, auc
from sklearn.metrics import roc_curve, roc_auc_score
import matplotlib.pyplot as plt

"""
This code contains functions for model validation.
It's not an integral part of the pipeline.

"""


def auc_metric(labal, prediction):
    fpr, tpr, thresholds = roc_curve(labal, prediction)
    return auc(fpr, tpr)


def classification_metrics(clf, x_train, x_test, y_train, y_test, thr=0.5, plot_roc=False):
    results = pd.DataFrame()
    y_pred_test = clf.predict(x_test)
    y_pred_train = clf.predict(x_train)
    y_pred_test_proba = clf.predict_proba(x_test)[:, 1]
    y_pred_train_proba = clf.predict_proba(x_train)[:, 1]
    results.loc['Precision', 'Train'] = precision_score(y_train, y_pred_train)
    results.loc['Recall', 'Train'] = recall_score(y_train, y_pred_train)
    # thr effects only the test precison and recall
    results.loc['Precision', 'Test'] = precision_score(y_test, (y_pred_test_proba > thr) * 1)
    results.loc['Recall', 'Test'] = recall_score(y_test, (y_pred_test_proba > thr) * 1)
    results.loc['AUC', 'Train'] = auc_metric(y_train.tolist(), y_pred_train_proba)
    results.loc['AUC', 'Test'] = auc_metric(y_test.tolist(), y_pred_test_proba)
    conf_df = pd.DataFrame(confusion_matrix(y_test, (y_pred_test_proba > thr) * 1).T,
                           columns=['True_Negative', 'True_Positive'],
                           index=['Predicted_Negative', 'Predicted_Positive'])
    display((conf_df / conf_df.sum().sum()).round(2))
    display(conf_df)
    display(results)

    if plot_roc:
        y_score = clf.decision_function(x_test)
        fpr, tpr, thresholds = roc_curve(y_test, y_score)
        roc_auc = roc_auc_score(y_test, y_score)

        # Plot ROC curve
        plt.figure(figsize=(16, 12))
        plt.plot(fpr, tpr, label='ROC curve (area = %0.3f)' % roc_auc)
        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.0])
        plt.xlabel('False Positive Rate (1 - Specificity)', size=16)
        plt.ylabel('True Positive Rate (Sensitivity)', size=16)
        plt.title('ROC Curve', size=20)
        plt.legend(fontsize=14);
