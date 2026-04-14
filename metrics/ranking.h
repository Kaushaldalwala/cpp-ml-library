// ranking metrics headder file 

#ifndef RANKING_METRICS_H
#define RANKING_METRICS_H

#include <vector>
#include <algorithm>
#include <cmath>
#include <numeric>
using namespace std;

class RankingMetrics {
public:

    // ================= COVERAGE ERROR =================
    static double coverage_error(const vector<vector<int>>& y_true,
                                 const vector<vector<double>>& y_score) {
        int n = y_true.size();
        double total = 0;

        for (int i = 0; i < n; i++) {
            vector<pair<double,int>> data;
            for (int j = 0; j < y_score[i].size(); j++)
                data.push_back({y_score[i][j], j});

            sort(data.begin(), data.end(), greater<>());

            int max_rank = 0;
            for (int k = 0; k < data.size(); k++) {
                if (y_true[i][data[k].second] == 1)
                    max_rank = max(max_rank, k + 1);
            }
            total += max_rank;
        }
        return total / n;
    }

    // ================= LRAP =================
    static double label_ranking_average_precision_score(
        const vector<vector<int>>& y_true,
        const vector<vector<double>>& y_score) {

        int n = y_true.size();
        double total = 0;

        for (int i = 0; i < n; i++) {
            vector<pair<double,int>> data;
            for (int j = 0; j < y_score[i].size(); j++)
                data.push_back({y_score[i][j], j});

            sort(data.begin(), data.end(), greater<>());

            int relevant = 0;
            double score = 0;

            for (int k = 0; k < data.size(); k++) {
                if (y_true[i][data[k].second] == 1) {
                    relevant++;
                    score += (double)relevant / (k + 1);
                }
            }

            int total_relevant = accumulate(y_true[i].begin(), y_true[i].end(), 0);
            if (total_relevant > 0)
                total += score / total_relevant;
        }

        return total / n;
    }

    // ================= RANKING LOSS =================
    static double label_ranking_loss(const vector<vector<int>>& y_true,
                                     const vector<vector<double>>& y_score) {
        int n = y_true.size();
        double total_loss = 0;

        for (int i = 0; i < n; i++) {
            int loss = 0, total_pairs = 0;

            for (int j = 0; j < y_true[i].size(); j++) {
                for (int k = 0; k < y_true[i].size(); k++) {
                    if (y_true[i][j] == 1 && y_true[i][k] == 0) {
                        total_pairs++;
                        if (y_score[i][j] <= y_score[i][k])
                            loss++;
                    }
                }
            }

            if (total_pairs > 0)
                total_loss += (double)loss / total_pairs;
        }

        return total_loss / n;
    }

    // ================= DCG =================
    static double dcg_score(const vector<int>& y_true,
                            const vector<double>& y_score,
                            int k = -1) {

        vector<pair<double,int>> data;
        for (int i = 0; i < y_score.size(); i++)
            data.push_back({y_score[i], y_true[i]});

        sort(data.begin(), data.end(), greater<>());

        if (k == -1) k = data.size();

        double dcg = 0;
        for (int i = 0; i < k; i++) {
            double rel = data[i].second;
            dcg += rel / log2(i + 2);
        }

        return dcg;
    }

    // ================= NDCG =================
    static double ndcg_score(const vector<int>& y_true,
                             const vector<double>& y_score,
                             int k = -1) {

        double dcg = dcg_score(y_true, y_score, k);

        vector<int> ideal = y_true;
        sort(ideal.begin(), ideal.end(), greater<>());

        vector<double> dummy_score(ideal.size());
        for (int i = 0; i < ideal.size(); i++)
            dummy_score[i] = ideal[i];

        double idcg = dcg_score(ideal, dummy_score, k);

        return (idcg == 0) ? 0 : dcg / idcg;
    }
};

#endif