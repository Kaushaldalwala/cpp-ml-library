// this is the header file of the clustring_unsupervised metrics

#ifndef CLUSTERING_METRICS_H
#define CLUSTERING_METRICS_H

#include <vector>
#include <cmath>
#include <map>
#include <numeric>
#include <limits>
#include <algorithm>

namespace metrics {

class ClusteringMetrics {
private:

    // Euclidean distance
    static double euclidean(const std::vector<double>& a,
                            const std::vector<double>& b) {
        double sum = 0;
        for (int i = 0; i < a.size(); i++)
            sum += (a[i] - b[i]) * (a[i] - b[i]);
        return std::sqrt(sum);
    }

    // Compute centroid of cluster
    static std::vector<double> centroid(const std::vector<std::vector<double>>& X,
                                        const std::vector<int>& labels,
                                        int cluster_id) {
        int count = 0;
        std::vector<double> center(X[0].size(), 0);

        for (int i = 0; i < X.size(); i++) {
            if (labels[i] == cluster_id) {
                for (int j = 0; j < X[i].size(); j++)
                    center[j] += X[i][j];
                count++;
            }
        }

        for (double &val : center)
            val /= count;

        return center;
    }

public:

    // ================= CALINSKI-HARABASZ =================
    static double calinski_harabasz_score(const std::vector<std::vector<double>>& X,
                                          const std::vector<int>& labels) {

        int n = X.size();
        std::map<int, std::vector<int>> clusters;

        for (int i = 0; i < labels.size(); i++)
            clusters[labels[i]].push_back(i);

        int k = clusters.size();

        // global centroid
        std::vector<double> global(X[0].size(), 0);
        for (auto& row : X)
            for (int j = 0; j < row.size(); j++)
                global[j] += row[j];

        for (double& val : global)
            val /= n;

        double between = 0, within = 0;

        for (auto& [cluster_id, indices] : clusters) {
            std::vector<double> c = centroid(X, labels, cluster_id);

            for (int idx : indices)
                within += pow(euclidean(X[idx], c), 2);

            between += indices.size() * pow(euclidean(c, global), 2);
        }

        return (between / (k - 1)) / (within / (n - k));
    }

    // ================= DAVIES-BOULDIN =================
    static double davies_bouldin_score(const std::vector<std::vector<double>>& X,
                                       const std::vector<int>& labels) {

        std::map<int, std::vector<int>> clusters;

        for (int i = 0; i < labels.size(); i++)
            clusters[labels[i]].push_back(i);

        int k = clusters.size();

        std::vector<std::vector<double>> centroids;
        std::vector<double> scatter;

        for (auto& [cid, idxs] : clusters) {
            auto c = centroid(X, labels, cid);
            centroids.push_back(c);

            double s = 0;
            for (int i : idxs)
                s += euclidean(X[i], c);

            scatter.push_back(s / idxs.size());
        }

        double db = 0;

        for (int i = 0; i < k; i++) {
            double max_ratio = 0;
            for (int j = 0; j < k; j++) {
                if (i == j) continue;

                double dist = euclidean(centroids[i], centroids[j]);
                double ratio = (scatter[i] + scatter[j]) / dist;

                max_ratio = std::max(max_ratio, ratio);
            }
            db += max_ratio;
        }

        return db / k;
    }

    // ================= SILHOUETTE (PER SAMPLE) =================
    static std::vector<double> silhouette_samples(
        const std::vector<std::vector<double>>& X,
        const std::vector<int>& labels) {

        int n = X.size();
        std::vector<double> scores(n);

        for (int i = 0; i < n; i++) {

            double a = 0; // intra-cluster
            double b = std::numeric_limits<double>::max();

            int same_count = 0;

            for (int j = 0; j < n; j++) {
                if (i == j) continue;

                double dist = euclidean(X[i], X[j]);

                if (labels[i] == labels[j]) {
                    a += dist;
                    same_count++;
                }
            }

            if (same_count > 0)
                a /= same_count;

            // nearest cluster distance
            std::map<int, std::vector<double>> cluster_dists;

            for (int j = 0; j < n; j++) {
                if (labels[i] != labels[j]) {
                    cluster_dists[labels[j]].push_back(
                        euclidean(X[i], X[j])
                    );
                }
            }

            for (auto& [cid, dists] : cluster_dists) {
                double mean = std::accumulate(dists.begin(), dists.end(), 0.0) / dists.size();
                b = std::min(b, mean);
            }

            scores[i] = (b - a) / std::max(a, b);
        }

        return scores;
    }

    // ================= SILHOUETTE SCORE =================
    static double silhouette_score(
        const std::vector<std::vector<double>>& X,
        const std::vector<int>& labels) {

        auto scores = silhouette_samples(X, labels);

        double sum = std::accumulate(scores.begin(), scores.end(), 0.0);
        return sum / scores.size();
    }

};

}

#endif