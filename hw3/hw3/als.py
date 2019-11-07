from __future__ import print_function
import sys
from collections import defaultdict
import numpy as np
from numpy.random import rand
from numpy import matrix
from pyspark.sql import SparkSession
import copy
import csv
LAMBDA = 0.01   # regularization
np.random.seed(42)


def rmse(R, ms, us):
    diff = R - ms * us.T
    return np.sqrt(np.sum(np.power(diff, 2)) / (M * U))


def update(i, mat, ratings):
    uu = mat.shape[0]
    ff = mat.shape[1]
#
    XtX = mat.T * mat
    Xty = mat.T * ratings[i, :].T
#
    for j in range(ff):
        XtX[j, j] += LAMBDA * uu
#
    return np.linalg.solve(XtX, Xty)


if __name__ == "__main__":

    if len(sys.argv) == 3:
        file = sys.argv[1]
        output = sys.argv[2]
    else:
        print("your input format is wrong")
        exit(0)

    spark = SparkSession\
        .builder\
        .appName("PythonALS")\
        .getOrCreate()
    
    sc = spark.sparkContext
    ratingTable = spark.read.csv(file, header=True)
    lines = ratingTable.rdd.repartition(2).map(list).map(lambda x:(int(x[0]),int(x[1]),float(x[2])))
    movies = lines.map(lambda x: x[1]).distinct().collect()
    movies = sorted(movies)
    users = lines.map(lambda x: x[0]).distinct().collect()

    dict_mv_id = defaultdict(list)
    i = 0
    for mv in movies:
        dict_mv_id[mv] = i
        i+=1

    ratings = lines.map(lambda x: ((x[0]-1,dict_mv_id[x[1]]),x[2])).collect()
    
    M = len(movies)
    U = len(users)
    F = 5
    ITERATIONS = 5
    partitions = 2
    # map the movie to the right collumn
    

    print("Running ALS with M=%d, U=%d, F=%d, iters=%d, partitions=%d\n" %
          (M, U, F, ITERATIONS, partitions))
    


    r = np.zeros([U,M])
    for rating in ratings:
        r[rating[0][0]][rating[0][1]] = rating[1]
        #print((rating[0][0],rating[0][1]),r[rating[0][0]][rating[0][1]])
    # print(r[0][:11])
    # exit(0)
    Filter = copy.deepcopy(r)
    R = np.matrix(r.T)
    ms = matrix(rand(M, F))
    us = matrix(rand(U, F))

    Rb = sc.broadcast(R)
    msb = sc.broadcast(ms)
    usb = sc.broadcast(us)

    for i in range(ITERATIONS):
        ms = sc.parallelize(range(M), partitions) \
               .map(lambda x: update(x, usb.value, Rb.value)) \
               .collect()
        # collect() returns a list, so array ends up being
        # a 3-d array, we take the first 2 dims for the matrix
        ms = matrix(np.array(ms)[:, :, 0])
        msb = sc.broadcast(ms)

        us = sc.parallelize(range(U), partitions) \
               .map(lambda x: update(x, msb.value, Rb.value.T)) \
               .collect()
        us = matrix(np.array(us)[:, :, 0])
        usb = sc.broadcast(us)

        error = rmse(R, ms, us)
        print("Iteration %d:" % i)
        print("\nRMSE: %5.4f\n" % error)

    us = np.array(us)
    ms = np.array(ms)
    prediction = np.dot(us,ms.T)
    #610*9724
    for i in range(U):
        for j in range(M):
            if Filter[i][j]==0:
                Filter[i][j]=1
            else: Filter[i][j]=0
    # get a filter
    prediction = prediction*Filter
   
    #inverse the table between mvid and colid
    dict_mv_id_inv = defaultdict(list)
    for k,v in dict_mv_id.items():
        dict_mv_id_inv[v] = k

    # collect the result
    result = []
    for i in range(U):
        for j in range(M):
            if prediction[i][j]!=0:
                mv = dict_mv_id_inv[j]
                user = i+1
                result.append([user,mv,prediction[i][j]])

    # write into csv file
    with open(output, "wb") as f:
        writer = csv.writer(f, delimiter=',')
        writer.writerow(["userId","movieId","ratings"])
        for item in result:
                    writer = csv.writer(f, delimiter=',')
                    writer.writerow(item)
    f.close()


    spark.stop()

    