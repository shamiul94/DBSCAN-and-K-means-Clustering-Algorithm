import sys

import numpy as np
import matplotlib.pyplot as plt
import copy

sys.setrecursionlimit(10000)


class Clustering:
    def __init__(self, K, input_file_path):
        self.K = K
        self.eps = 0
        self.MINPTS = K
        self.input_path = input_file_path
        self.max_X = -np.inf
        self.max_Y = -np.inf
        self.data_set = []
        self.get_data()
        self.calculate_eps()
        self.visited = [False] * len(self.data_set)
        self.num_of_cluster_by_dbscan = 0
        self.points_respective_cluster = [0 for _ in range(len(self.data_set))]

    def get_data(self):
        with open(self.input_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        for line in lines:
            data = [float(ls) for ls in line.strip().split()]
            self.data_set.append(data)
            self.max_X = max(self.max_X, abs(data[0]))
            self.max_Y = max(self.max_Y, abs(data[1]))
        # print(self.dataset)
        self.data_set.sort(key=lambda curr_data: (curr_data[0], curr_data[1]))
        self.data_set = np.array(self.data_set) / np.array([self.max_X, self.max_Y])

    def get_distance(self, point1, point2):
        return np.sqrt((point1[0] - point2[0]) ** 2 + (point1[1] - point2[1]) ** 2)

    def calculate_eps(self):
        k_th_dist = []
        plot_x = []
        for i in range(len(self.data_set)):
            plot_x.append(i)
            curr_dist = []
            for j in range(len(self.data_set)):
                if i != j:
                    d = self.get_distance(self.data_set[i], self.data_set[j])
                    curr_dist.append(d)
            curr_dist.sort()
            k_th_dist.append(curr_dist[self.K - 1])

        k_th_dist.sort()
        plt.figure(1)
        plt.plot(plot_x, k_th_dist)
        plt.grid()
        plt.show()
        # plt.savefig('Rumman_blob_eps.png')

        print('Enter EPS: ')
        self.eps = float(input())

    def dfs(self, start_point, cluster_idx):
        self.points_respective_cluster[start_point] = cluster_idx
        self.visited[start_point] = True

        for i in range(len(self.data_set)):
            if i != start_point:
                if not self.visited[i] and self.get_distance(self.data_set[start_point], self.data_set[i]) <= self.eps:
                    self.dfs(i, cluster_idx)

    def DBSCAN(self):
        # self.eps = 0.08
        print('Starting DBSCAN')
        core_points = []
        for i in range(len(self.data_set)):
            neighbour_of_curr_point = 0
            for j in range(len(self.data_set)):
                if i != j:
                    d = self.get_distance(self.data_set[i], self.data_set[j])
                    if d <= self.eps:
                        neighbour_of_curr_point += 1

            if neighbour_of_curr_point >= self.MINPTS:
                core_points.append(i)

        np.random.shuffle(core_points)
        for core_point in core_points:
            if not self.visited[core_point]:
                self.num_of_cluster_by_dbscan += 1
                self.dfs(core_point, self.num_of_cluster_by_dbscan)

        print("Total cluster number found by DBSCAN is: ", self.num_of_cluster_by_dbscan)

        color_options = ["#DFFF00", "#FF7F50", "#DE3163", "#9FE2BF", "#40E0D0", "#6495ED", "#CCCCFF"]
        plt.figure(2)
        for i in range(len(self.data_set)):
            if self.points_respective_cluster[i]:
                color = color_options[(self.points_respective_cluster[i] - 1) % len(color_options)]
                plt.scatter(self.data_set[i][0], self.data_set[i][1], color=color)

        plt.show()
        # plt.savefig('Rumman_blob_dbscan.png')


    def k_means_clustering(self, max_iteration_no=1000):
        # self.eps = 0.08
        # self.num_of_cluster_by_dbscan = 2

        print('In K-means clustering')

        interval = len(self.data_set) // self.num_of_cluster_by_dbscan

        centroid_idx = [i * interval for i in range(self.num_of_cluster_by_dbscan)]
        centroid_list = [self.data_set[centroid_idx[i]] for i in range(self.num_of_cluster_by_dbscan)]

        temp_points_respective_cluster_idx = [-1 for _ in range(len(self.data_set))]
        dist = [np.inf for _ in range(len(self.data_set))]

        for itr in range(max_iteration_no):
            print('iteration no: ', itr)
            for i in range(len(self.data_set)):
                dist[i] = np.inf
                for j in range(self.num_of_cluster_by_dbscan):
                    if self.get_distance(self.data_set[i], centroid_list[j]) < dist[i]:
                        dist[i] = self.get_distance(self.data_set[i], centroid_list[j])
                        temp_points_respective_cluster_idx[i] = j

            full_clusters = [[] for _ in range(self.num_of_cluster_by_dbscan)]
            for i in range(len(self.data_set)):
                full_clusters[temp_points_respective_cluster_idx[i]].append(self.data_set[i])

            new_centroid_list = [np.average(np.array(full_clusters[i]), axis=0) for i in
                                 range(self.num_of_cluster_by_dbscan)]

            check = True
            for i in range(self.num_of_cluster_by_dbscan):
                diff_arr = np.abs(centroid_list[i] - new_centroid_list[i])
                if (diff_arr != 0).any():
                    check = False
            if check == True:
                break

            centroid_list = copy.deepcopy(new_centroid_list)

        color_options = ["#DFFF00", "#FF7F50", "#DE3163", "#9FE2BF", "#40E0D0", "#6495ED", "#CCCCFF"]
        plt.figure(3)
        for i in range(len(self.data_set)):
            if temp_points_respective_cluster_idx[i] >= 0:
                color = color_options[temp_points_respective_cluster_idx[i] % len(color_options)]
                plt.scatter(self.data_set[i][0], self.data_set[i][1], color=color)

        for centroid in centroid_list:
            plt.scatter(centroid[0], centroid[1], color='#000000', marker='p', linewidths=5)

        plt.show()
        # plt.savefig('Rumman_blob_kmc.png')



if __name__ == "__main__":
    np.random.seed(38)
    # c = Clustering(4, "./data/blobs.txt")
    # c = Clustering(4, "./data/moons.txt")
    c = Clustering(4, "./data/bisecting.txt")
    # c.calculate_eps()
    c.DBSCAN()
    c.k_means_clustering()

# bisecting - eps: 0.03
# blob - eps: 0.08
# moon - eps: 0.06
