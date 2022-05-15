import random
import csv
import numpy as np
import itertools
import math

def random_centroids(n, data):
    centroids = []
    for x in range(n):
        x = random.randint(1, 200)
        centroids.append(data[x])
    return centroids

def euclidean_distance(x1, x2):
    tmp = 0
    for i in range(len(x1)):
        tmp += np.sum((float(x1[i]) - float(x2[i])) ** 2)
    return np.sqrt(tmp)

def manhatten_distance(x1, x2):
    tmp = 0
    for i in range(len(x1)):
        tmp += (abs(float(x1[i]) - float(x2[i])))
    return tmp

def closest_centroid(sample, centroids, measurement):
    if(measurement == "manhatten"):
        distances = [manhatten_distance(sample, point) for point in centroids]
    if(measurement == "euclidean"):
        distances = [euclidean_distance(sample, point) for point in centroids]
    closest_index = np.argmin(distances)
    return closest_index

def create_clusters(centroids, n, data, measurement):
    clusters = [[] for _ in range(n)]
    for idx, sample in enumerate(data):
        centroid_idx = closest_centroid(sample, centroids, measurement)
        clusters[centroid_idx].append(idx)
    return clusters

def calculate_new_centroids(clusters, n, data):
    centroids = np.zeros((n, 7))
    arr = []
    for cluster_idx, cluster in enumerate(clusters):
        for i in cluster:
            arr.append(data[i])
        cluster_mean = np.mean(arr, axis=0)
        centroids[cluster_idx] = cluster_mean
        arr.clear()
    return centroids

def centroid_changed(old_centroids, current_centroids, n, measurement):
    distances = []
    for i in range(n):
        if (measurement == "manhatten"):
            dist = manhatten_distance(old_centroids[i], current_centroids[i])
        if (measurement == "euclidean"):
            dist = euclidean_distance(old_centroids[i], current_centroids[i])
        distances.append(dist)
    return sum(distances) == 0

def outlier_detect(clusters, centroids, data, n, measurement):
    distances = [{} for _ in range(n)]
    sorted_distances = [{} for _ in range(n)]
    outliers = [{} for _ in range(n)]
    clean_points = [{} for _ in range(n)]
    for i, cluster in enumerate(clusters):
        for idx, sample in enumerate(cluster):
            if (measurement == "manhatten"):
                dist = manhatten_distance(data[sample], centroids[i])
            if (measurement == "euclidean"):
                dist = euclidean_distance(data[sample], centroids[i])

            distances[i][sample] = dist

    for idx, sample in enumerate(distances):
        new_list = sorted(sample.items(), key=lambda x: x[1])
        sorted_distances[idx] = dict(new_list)
    for i in range(n):
        x = math.ceil(len(sorted_distances[i]) * 0.05)
        j = len(sorted_distances[i]) -x
        clean_points[i] = dict(itertools.islice(sorted_distances[i].items(), j))
        outliers[i] = dict(list(sorted_distances[i].items())[j:])
        print(len(clean_points[i]), clean_points[i])
        print(len(outliers[i]), outliers[i])

if __name__ == "__main__":
    file = open("Power_Consumption.csv")
    csvreader = csv.reader(file)
    header = next(csvreader)

    data = []
    for row in csvreader:
        data.append(row[1:])
    for i in range(0, 200):
        for j in range(0, 7):
            data[i][j] = float(data[i][j])
    num = 4
    #method = "euclidean"
    method = "manhatten"
    current_centroids = random_centroids(num, data)
    for _ in range(100):
        clusters = create_clusters(current_centroids, num, data, method)
        old_centroids = current_centroids
        current_centroids = calculate_new_centroids(clusters, num, data)
        if centroid_changed(old_centroids, current_centroids, num, method):
            break
    for i in clusters:
        print(len(i), i)
    outlier_detect(clusters, current_centroids, data, num, method)









