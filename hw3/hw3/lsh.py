import sys
import os
import itertools
from collections import defaultdict
from pyspark import SparkContext
import copy
import csv

def jaccard(list1,list2):
    count = 0
    if len(list1) == len(list2):
        for i in range(len(list1)):
            if list1[i]==list2[i]:
                count+=1
        jac = float(count)/len(list1)
    else: jac = count
    return jac

def real_similarity(ith_user, user):
    count = [x for x in info_dict[ith_user] if x in info_dict[user]]
    union = list(set(info_dict[ith_user]).union(set(info_dict[user])))
    #print(ith_user,user)
    #print(len(count))
    return float(len(count))/len(union)

def split_list(l,b):
    bandsize=5
    return l[bandsize*(b-1):bandsize*b]

if __name__ == "__main__":

    if len(sys.argv) == 3:
        file = sys.argv[1]
        output = sys.argv[2]
    else:
        print("your input format is wrong")
        exit(0)
    num_partition = 2
    

    sc = SparkContext.getOrCreate()
    # convert to rdd
    lines = sc.textFile(file).map(lambda line: tuple(line.strip().split(',')[:3]))
    # remove header
    lines = lines.filter(lambda line: line[0]!= 'userId')
    #convert to int type
    lines = lines.map(lambda line: (int(line[0]),int(line[1]),float(line[2])))
    line = lines.map(lambda line: (int(line[0]),(line[1],line[2])))
    line1 = lines.map(lambda line: (int(line[0]),int(line[1])))

    #unique movie id for later use
    unique_movie = lines.map(lambda x: x[1]).distinct().collect()
    unique_movie = sorted(unique_movie)

    # sort data by userid
    # userid to be the bracket
    line = line.groupByKey().mapValues(list)
    line1 = line1.groupByKey().mapValues(list)
    line.repartition(num_partition)
    line1.repartition(num_partition)
    #store ratings
    rating = line.collect()
    ratings = defaultdict(list)
    for r in rating:
        ratings[r[0]] = r[1]


    #store important info
    info_dict = defaultdict(list)
    info = line1.collect()
    for i in range(len(info)):
        info_dict[info[i][0]] = info[i][1]
    # k = [x for x in info_dict[1][0] if x in info_dict[43][0]]
    # print(k)

    lsh_dict = defaultdict(list)
    for i in range(1,51):
        Hash_result = line1.mapValues(lambda mv: [((3*mvid+11*i)%100)+1 for mvid in mv])
        Hash_result = Hash_result.mapValues(lambda list_hash: min(list_hash)).collect()
        N = len(Hash_result)
        for k in range(N):
            lsh_dict[Hash_result[k][0]].append(Hash_result[k][1])
    # lsh_dict: key:userid value: list of hash values of size 50
    users = lsh_dict.keys()

    # jac = defaultdict(list)
    # for ith_user in range(1,len(users)+1):
    #     otheruser = copy.deepcopy(users)
    #     otheruser.remove(users[ith_user-1])
    #     for user in otheruser:
    #         if jaccard(lsh_dict[ith_user],lsh_dict[user])>0:
    #         # at least one band is the same
    #             sim = real_similarity(ith_user,user)
    #             # return how many elements are the same in i-th user and current user
    #             jac[ith_user].append((sim,user))
    #hashing bands
    jac = defaultdict(list)
    for b in range(1,11):
        for ith_user in range(1,len(users)+1):
                vec = split_list(lsh_dict[ith_user],b)
                jac[(b,tuple(vec))].append(ith_user)
    # for k,v in jac.items():
    #     if len(v)>=2 and 567 in v:
    #         print(k)
    #         print(v)
    #         print('*****************************')
    # exit(0)
    #find candidate
    candidates = defaultdict(list)
    for user in users:
        find_pair = defaultdict(list)
        for k,v in jac.items():
            if len(v)>=2 and user in v:
            # print(k)
            # print(v)
            # print('*****************************')
                v1 = copy.deepcopy(v)
                v1.remove(user)
                for element in v1:
                    find_pair[element].append(1)

        candidate = []
        for k,v in find_pair.items():
            candidate.append((len(v),(user,k)))
    
        candidate = sorted(candidate)
        candidates[user] = candidate

    #calculate jaccard for each user and find the top 3 similar user
    sim = defaultdict(list)
   
    for user in users:
        for pair in candidates[user]:
            user = pair[1][0]
            other = pair[1][1]
            jacc = real_similarity(user,other)
            sim[user].append((jacc,other))
    
        #print((user,other), jacc)
        #sim.append((jacc,other))
    # print(sorted(sim[32]))
    # exit(0)
    # output prediction
    #  find top 3 similar user for each user
    similar_users = defaultdict(list)
    for user in users:
        top_sim = sorted(sim[user])
        similar_user = []
        if len(top_sim)==0:
            continue
        if len(top_sim)==1:
            other1 = top_sim[0][1]
            similar_user.append(other)
        if len(top_sim)==2:
            other1 = top_sim[0][1]
            other2 = top_sim[1][1]
            similar_user.append(other1)
            similar_user.append(other2)
        if len(top_sim)>=3:
            other1 = top_sim[-1][1]
            other2 = top_sim[-2][1]
            other3 = top_sim[-3][1]
            similar_user.append(other1)
            similar_user.append(other2)
            similar_user.append(other3)
        similar_users[user] = similar_user
    
    # print("****************")
    # calculate ratings based on top 3 similar user
    # and write csv file
    # with open("predictions.csv","w") as csvfile: 
    #         writer = csv.writer(csvfile)
    #         writer.writerows([["userId","movieId","ratings"]])
    # writer.close()
    # with open("predictions.csv", "wb") as f:
    #         writer = csv.writer(f, delimiter=',')
    #         writer.writerow(["userId","movieId","ratings"])
    final_result = []
    for user in users:
        if len(similar_users[user])==0:
            continue
        for mv in unique_movie:
            if mv not in info_dict[user]:
                similar_users_list = similar_users[user]
                score = 0.0
                i = 0
                for other in similar_users_list:
                    rating = ratings[other]
                    for r in rating:
                        if r[0]==mv:
                            score+=r[1]
                            i+=1
                if score == 0:
                    continue

                predict = float(score)/i
                score = 0.0
                final_result.append([user,mv,predict]) 
        
                #print(user,mv,predict)
                #print("************************************")
    # write into csv file
    with open(output, "wb") as f:
        writer = csv.writer(f, delimiter=',')
        writer.writerow(["userId","movieId","ratings"])
        for item in final_result:
                    writer = csv.writer(f, delimiter=',')
                    writer.writerow(item)
    f.close()
