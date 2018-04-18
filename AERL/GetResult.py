# -*- coding: utf-8 -*-
import numpy as np
import random

def get_rank_matrix(result_matix):
    query_num=len(result_matix[:,0])
    ranks=np.zeros(query_num)
    for i in range(query_num):
        true = result_matix[i, 0]
        list = result_matix[i, :].tolist()
        list.sort(reverse=True)
        rank1 = list.index(true)+1
        list.reverse()
        rank2=len(list)-list.index(true)
        ranks[i]=rank1+(rank2-rank1)*random.random()
    return ranks

def get_result_by_ranks(ranks,rec_k_list):
    '''

    :param ranks: result rank of every query
    :param candidate_num:
    :param rec_k_list: you want compute the recall of befor k list
    :return: the matrix[mean rank,mrr,recall@1,...recall@k]
    '''
    result=np.zeros(len(rec_k_list)+2)
    mean_rank=np.mean(ranks)
    result[0]=mean_rank
    for index,k in enumerate(rec_k_list):
        result[index+2]=sum(ranks<k+1)/float(len(ranks))
    mrr_sum=0.0
    for i in ranks:
        mrr_sum=mrr_sum+1/float(i)
    result[1]=mrr_sum/len(ranks)
    return result

def get_simple(ranks):
    '''

    :param ranks: result rank of every query
    :param candidate_num:
    :param rec_k_list: you want compute the recall of befor k list
    :return: the matrix[mean rank,mrr,recall@1,...recall@k]
    '''
    result=np.zeros(2)
    mean_rank=np.mean(ranks)
    result[0]=mean_rank

    # adding = lambda t: 1/float(t)
    mrr_sum = 0.0
    for i in ranks:
        mrr_sum=mrr_sum+1/float(i)
    # vfunc = np.vectorize(adding)
    # vfunc(ranks)
    # mrr_sum = np.sum(ranks)
    result[1]=mrr_sum/len(ranks)
    return result

