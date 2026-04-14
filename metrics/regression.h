// regression header file
#ifndef REGRESSION_METRICS_H
#define REGRESSION_METRICS_H

#include <vector>
#include <cmath>
#include <numeric>
#include <algorithm>

namespace metrics {

class RegressionMetrics {
public:

    static double mean_absolute_error(const std::vector<double>& y_true,
                                      const std::vector<double>& y_pred) {
        double sum = 0;
        for (int i = 0; i < y_true.size(); i++)
            sum += std::abs(y_true[i] - y_pred[i]);
        return sum / y_true.size();
    }

    static double mean_squared_error(const std::vector<double>& y_true,
                                     const std::vector<double>& y_pred) {
        double sum = 0;
        for (int i = 0; i < y_true.size(); i++)
            sum += pow(y_true[i] - y_pred[i], 2);
        return sum / y_true.size();
    }

    static double r2_score(const std::vector<double>& y_true,
                           const std::vector<double>& y_pred) {

        double mean = std::accumulate(y_true.begin(), y_true.end(), 0.0) / y_true.size();

        double ss_res = 0, ss_tot = 0;

        for (int i = 0; i < y_true.size(); i++) {
            ss_res += pow(y_true[i] - y_pred[i], 2);
            ss_tot += pow(y_true[i] - mean, 2);
        }

        return 1 - (ss_res / ss_tot);
    }

};

}


#endif