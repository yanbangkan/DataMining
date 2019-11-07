import sys
import os
import itertools
from collections import defaultdict
from pyspark import SparkContext
import copy
'''
the runtime is less than 1min when
THE SUPPORT THRESOLD is 200 and above

'''

def getOneFrequents(baskets):
    local_frequent = []
    countTable = {}
    for basket in baskets:
        for item in basket:
            if item not in countTable.keys():
                countTable[item] = 0
            else: countTable[item] += 1

    for item, item_num in countTable.items():
        if item_num*num_partition >= support_threshold:
            local_frequent.append(item)

    local_frequent = sorted(local_frequent)
    return local_frequent


def getMoreFrequents(baskets, pre_frequents, size):
    countTable = {}
    local_frequent = []
    candidates = []
    if size == 2:
        for candidate in itertools.combinations(pre_frequents, size):
            candidates.append(candidate)
    else:
        for item in itertools.combinations(pre_frequents, 2):
            if len(set(item[0]).intersection(set(item[1]))) == size - 2:
                candidate = tuple(sorted(set(item[0]).union(set(item[1]))))
                # print 'candidate=',candidate
                if candidate not in candidates:
                    temp = itertools.combinations(candidate, size - 1)
                    # to generate (b,c,j) as candidate, have to make sure (b,c),(c,j)(b,j) are previous candidates 
                    if set(temp).issubset(pre_frequents):
                        # print 'appending',candidate
                        candidates.append(candidate)
                else:
                    continue

    #exit(0)
    for candidate in candidates:
        for basket in baskets:
            if set(candidate).issubset(basket):
                countTable.setdefault(candidate, 0)
                countTable[candidate] += 1
    
    for candidate, count in countTable.items():
        # this is support for each partition, not support for the whole dataset
        if count*num_partition >= support_threshold:
            local_frequent.append(candidate)

    return sorted(local_frequent)


def apriori(iterator):
    baskets = []
    for l in iterator:
        baskets.append(l)
    freqitems = []
    size = 1
    
    onefrequentsItem = getOneFrequents(baskets)
    freqitems = copy.deepcopy(onefrequentsItem)
    
    size += 1
    cur_frequents = onefrequentsItem
    flag = 1
    while flag != 0:
        pre_frequents = cur_frequents
        cur_frequents = getMoreFrequents(baskets, pre_frequents, size)

        ###############################
        for item in cur_frequents:
            freqitems.append(item)
        flag = len(cur_frequents) 
        size += 1
    return freqitems


if __name__ == "__main__":

    if len(sys.argv) == 4:
        file = sys.argv[1]
        support_threshold = float(sys.argv[2])
        confident_threshold = float(sys.argv[3])
        num_partition = 2
    else: 
        print "your input format is wrong please see: spark-submit assoc.py ratings.csv <support threshold> <confidence threshold>"
        print "now the program is using support_threshold=200 and confident_threshold=0.5 "
        num_partition = 2
        support_threshold = 300
        confident_threshold = 0.5
        #file = "C:/Users/kyb/Desktop/inf553/HW2/ratings.csv"
        file = "ratings.csv"

    sc = SparkContext.getOrCreate()
    # convert to rdd
    lines = sc.textFile(file).map(lambda line: tuple(line.strip().split(',')[:2]))
    # remove header
    lines = lines.filter(lambda line: line[0]!= 'userId')
    #convert to int type
    lines = lines.map(lambda line: (int(line[0]),int(line[1])))
    # sort data by userid
    # userid to be the bracket
    lines = lines.groupByKey().mapValues(tuple).map(lambda x: x[1])

    lines.repartition(num_partition)
    local_frequent = lines.mapPartitions(apriori)
    local_list = local_frequent.collect()

    frequents = []
    if lines.getNumPartitions() == 1:
        for item in local_list:
            frequents.append(item)
    else:
        for item in local_list:
            if item in frequents:
                continue
            count = 0
            for line in lines.collect():
                if type(item)==type(1):
                    if item in line:
                        count += 1
                else:
                    if set(item).issubset(line):
                        count += 1

            if count >= support_threshold:
                frequents.append(item)
    #print(frequents,len(frequents))
    result_dict = defaultdict(list)
    for item in frequents:
        length = 0
        if type(item) == type(1):
            length = 1
        else:
            length = len(item)
        result_dict[length].append(item)


    for k,v in result_dict.items():
        for i in range(len(v)):
            print v[i]
    # print the result in specific format

    #calculate confidence
    
    result_conf = []
    
    fre_set = frequents
    for frequent_set in fre_set:
        if type(frequent_set)==type(1):
                continue
        frequent_set = list(frequent_set)
        for i in range(len(frequent_set)):

            j = frequent_set[i]
            U = copy.deepcopy(frequent_set)
            U.remove(j)
            #print(U,j)

            a = lines.filter(lambda line: set(U).issubset(set(line)))
            support1 = len(a.collect())
            b = a.filter(lambda line: j in line)
            support2 = len(b.collect())
            confi = float(support2)/support1
            if confi > confident_threshold:
                result_conf.append((confi,tuple(frequent_set),j))
                
    result_conf = sorted(result_conf)
    for i in range(len(result_conf)):
        print"%r , %r confi:%r"%(result_conf[i][1],result_conf[i][2],result_conf[i][0])