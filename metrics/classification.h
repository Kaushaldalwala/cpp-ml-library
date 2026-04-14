// header file of the classification metrics

#ifndef CLASSIFICATION_METRICS_H
#define CLASSIFICATION_METRICS_H

#include <vector>
#include <cmath>
#include <algorithm>

namespace metrics {

class ClassificationMetrics {
public:

    static double accuracy_score(const std::vector<int>& y_true,
                                 const std::vector<int>& y_pred) {

        int correct = 0;
        for (int i = 0; i < y_true.size(); i++)
            if (y_true[i] == y_pred[i]) correct++;

        return (double)correct / y_true.size();
    }

    static double precision_score(const std::vector<int>& y_true,
                                  const std::vector<int>& y_pred) {

        int tp = 0, fp = 0;
        for (int i = 0; i < y_true.size(); i++) {
            if (y_pred[i] == 1) {
                if (y_true[i] == 1) tp++;
                else fp++;
            }
        }
        return (tp + fp == 0) ? 0 : (double)tp / (tp + fp);
    }

};

}


#endif