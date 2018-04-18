# -*- coding: utf-8 -*-
import random
import os
import codecs
import pickle
import numpy as np
import feature as fut
import getLSA as lsa
import doc2vec as dv
from gensim.models.doc2vec import Doc2Vec, TaggedDocument


def get_query_all_file(file_name, query_num=None):
    f = open(file_name, 'rb')
    db = pickle.load(f)
    f.close()
    query_list = list()
    if query_num is None:
        for index, db_l in enumerate(db):
            db_0 = None
            db_1 = None
            db_2 = None
            db_3 = None
            if index < len(db) - 2:
                db_c = db[index + 1]
                db_n = db[index + 2]

                if index > 3:
                    db_0 = db[index - 4]
                if index > 2:
                    db_1 = db[index - 3]
                if index > 1:
                    db_2 = db[index - 2]
                if index > 0:
                    db_3 = db[index - 1]
                if is_right(db_l, db_c, db_n):
                    front5 = get_front_N5(db_0, db_1, db_2, db_3, db_c)
                    query_list.append(((db_c[2], db_c[3], db_c[11]), (db_l[2], db_l[3], db_l[11]),
                                       (db_n[2], db_n[3], db_n[11], db_n[1]), front5))

        w = open('/data/rap_data_save/dataset/query_all', 'wb')
        parameters = query_list
        pickle.dump(parameters, w, protocol=pickle.HIGHEST_PROTOCOL)
        w.close()
    else:
        id_list = range(len(db))
        random.shuffle(id_list)
        nowget = 0
        index = 0
        while nowget < query_num:

            current_index = id_list[index]
            index += 1
            if 0 < current_index < len(db) - 1:
                db_0 = None
                db_1 = None
                db_2 = None
                db_3 = None
                db_c = db[current_index]
                db_l = db[current_index - 1]
                db_n = db[current_index + 1]
                if current_index > 4:
                    db_0 = db[current_index - 5]
                if current_index > 3:
                    db_1 = db[current_index - 4]
                if current_index > 2:
                    db_2 = db[current_index - 3]
                if current_index > 1:
                    db_3 = db[current_index - 2]
                if is_right(db_l, db_c, db_n):
                    front5 = get_front_N5(db_0, db_1, db_2, db_3, db_c)
                    query_list.append(((db_c[2], db_c[3], db_c[11]), (db_l[2], db_l[3], db_l[11]),
                                       (db_n[2], db_n[3], db_n[11], db_n[1]), front5))
                    nowget += 1
        w = open('dataset/query_random', 'wb')

        parameters = query_list
        pickle.dump(parameters, w, protocol=pickle.HIGHEST_PROTOCOL)
        w.close()
    return query_list


def get_and_Doc(filename=None):
    f = open(filename, 'rb')
    db = pickle.load(f)
    f.close()
    file_num = '1'
    count = ''
    for index, text_l in enumerate(db):
        if text_l[9] == file_num:
            count = count + text_l[11] + '\n'
        else:

            c = open('DOC/' + file_num, 'w')
            c.write(count)
            count = ''
            c.close()
            file_num = text_l[9]
    c = open('DOC/' + file_num, 'w')
    c.write(count)
    c.close()


def is_right(last, current, ne):
    if last[9] == current[9] and current[9] == ne[9] and not (last[3] == '') and not (current[3] == '') and not (
                ne[3] == ''):
        return True
    else:
        return False


def get_front_N5(db_0, db_1, db_2, db_3, db_c):
    front = ''
    if not (db_0 is None):
        if db_0[9] == db_c[9]:
            front = front + db_0[11] + '\n'
    if not (db_1 is None):
        if db_1[9] == db_c[9]:
            front = front + db_1[11] + '\n'
    if not (db_2 is None):
        if db_2[9] == db_c[9]:
            front = front + db_2[11] + '\n'
    if not (db_3 is None):
        if db_3[9] == db_c[9]:
            front = front + db_3[11] + '\n'
    return front


def get_random_except_id_by_file(file_name, id_except=299, num_candidate=299):  # id_except is not index of file db
    f = open(file_name, 'rb')
    db = pickle.load(f)
    f.close()
    random_index = range(len(db))
    random.shuffle(random_index)
    nowget = 0
    index = 0
    candidate_list = list()
    while nowget < num_candidate:

        ind = random_index[index]
        index += 1
        if not ind == id_except:
            if not db[ind][3] == '':
                candidate_list.append((db[ind][2], db[ind][3], db[ind][11]))
                nowget += 1

    return candidate_list


def get_feature_score_query_candidates(query, candidate, features):
    feature_score = np.zeros((len(candidate), len(features)))
    current_line = query[0]
    previous_line = query[1]

    lsa_candidate = []
    for i, cand in enumerate(candidate):
        leng = len(features)
        if 7 in features:
            leng = (leng - 1)
            lsa_candidate.append(cand[2])

        fea_sc = np.zeros(leng)
        index = 0
        f1 = fut.rhyming(query=current_line[2], rhyme_query=current_line[1], candidate=cand[2], rhyme_can=cand[1])
        f2 = fut.rhyming(query=previous_line[2], rhyme_query=previous_line[1], candidate=cand[2], rhyme_can=cand[1])
        if 1 in features:
            fea_sc[index] = f1.get_endrhyme()
            index += 1
        if 2 in features:
            fea_sc[index] = f2.get_endrhyme()
            index += 1
        if 3 in features:
            fea_sc[index] = f1.get_otherrhyme()
            index += 1
        if 4 in features:
            fea_sc[index] = f1.get_lineLength()
            index += 1
        if 5 in features:
            fea_sc[index] = f1.get_bow()
            index += 1
        if 6 in features:
            fea_sc[index] = f1.get_bow(front5=query[3] + previous_line[2])
            index += 1

        feature_score[i, :index] = fea_sc

    if 5 in features and not len(lsa_candidate) == 0:
        Lsa = lsa.LSA()
        lsa_f = Lsa.get_simli_trains_test(train_textlist=lsa_candidate, line=current_line[2])
        if 10 in features:
            feature_score[:, -2] = lsa_f
        else:
            feature_score[:, -1] = lsa_f
    return feature_score


def get_distribution_query_candidates_doc2vec(query, candidate, frontk=0, docv_du=100, is_phoneme=None, model=None):
    if is_phoneme is None:

        current_line = query[0]
        query_set = current_line[2]
        vec = np.zeros((0, docv_du))
        model = Doc2Vec.load('doc_model_/doc_model_size125')
        for i, cand in enumerate(candidate):
            vec_doc = dv.get_Doc2vec_vector(model, query_set, cand[2], docv_du)
            vec = np.row_stack((vec, vec_doc))
    else:
        current_line = query[0]
        previous_line = query[1]
        if frontk == 0:
            query_set = current_line[1]
        else:
            query_set = previous_line[1]

        vec = np.zeros((0, docv_du))
        model = Doc2Vec.load(model)
        for i, cand in enumerate(candidate):
            vec_doc = dv.get_Doc2vec_vector_phoneme(model, query_set, cand[1], docv_du)
            vec = np.row_stack((vec, vec_doc))
    return vec


def get_RANKSVM_input_file_feature_score_by_file(file_name, num_candidate, query_num=None, query_file=None):
    file_name1 = '/data/rap_data_save/dataset' + file_name
    if query_file is None:

        list_query = get_query_all_file(file_name1, query_num=query_num)
    else:
        f = open('artist_train_validation_test', 'rb')
        list_query = pickle.load(f)
        f.close()
    list_query = list_query[:]
    query_num = len(list_query)
    print('%d querys has done!' % (query_num))

    for i, query in enumerate(list_query):
        candidate = get_random_except_id_by_file(file_name=file_name1, id_except=query[2][3],
                                                 num_candidate=num_candidate - 1)
        candidate.insert(0, (query[2][0], query[2][1], query[2][2]))
        f = open('/data/rap_data_save/query_candidate' + file_name + '/' + str(i), 'wb')
        parameters = (query, candidate)
        pickle.dump(parameters, f, protocol=pickle.HIGHEST_PROTOCOL)
        f.close()
        print('%i querys candidate has done!' % (i))

    return query_num


def computFeature(Rankfilename, querynum=None, filename=None, features=[1, 2, 3, 4, 5, 6, 7], isRandom=None):
    file_open = open(Rankfilename, 'w+')
    feature_score_all = np.zeros((0, len(features)))
    if querynum is not None:

        if isRandom is None:
            querynum_list = range(querynum)
        else:
            querynum_list = np.load('random_index.npy')
            querynum_list = querynum_list[:querynum]
        for i, index in enumerate(querynum_list):
            f = open(filename + '/' + str(index), 'rb')
            (query, candidate) = pickle.load(f)
            f.close()
            feature_score = get_feature_score_query_candidates(query, candidate, features)
            print('%i feature_score has done!' % (index))
            write_feature_to_file(feature_score, i + 1, file_open, features)
            print('%i feature_score has write to %s!' % (i, Rankfilename))
    else:
        print('with wrong querynum')

    file_open.close()
    return feature_score_all


def computFeature_doc2vec(Rankfilename, querynum=None, filename=None, frontk=5,
                          docv_du=100, features=[15], is_phoneme=None, isRandom=None, can_num=300,
                          cut=None, save_cut=None, isreturn=None):
    file_open = open(Rankfilename, 'w+')
    feature_score_all = np.zeros((0, docv_du))

    if querynum is not None:

        if isRandom is None:
            querynum_list = range(querynum)
        else:
            querynum_list = np.load('random_index.npy')
            querynum_list = querynum_list[:querynum]
        if cut is None:
            if isreturn is None:
                for i, index in enumerate(querynum_list):
                    f = open(filename + '/' + str(index), 'rb')
                    (query, candidate) = pickle.load(f)
                    f.close()
                    candidate = candidate[:can_num]
                    feature_score = get_distribution_query_candidates_doc2vec(query, candidate, frontk=frontk,
                                                                              docv_du=docv_du, is_phoneme=is_phoneme)
                    print('%i feature_score has done!' % index)
                    write_feature_to_file(feature_score, i + 1, file_open, features=features,
                                          docv_du=docv_du)
                    print('%i feature_score has write to %s!' % (i, Rankfilename))
            else:
                for i, index in enumerate(querynum_list):
                    f = open(filename + '/' + str(index), 'rb')
                    (query, candidate) = pickle.load(f)
                    f.close()
                    candidate = candidate[:can_num]
                    feature_score = get_distribution_query_candidates_doc2vec(query, candidate, frontk=frontk,
                                                                              docv_du=docv_du, is_phoneme=is_phoneme)
                    print('%i feature_score has done!' % index)
                    feature_score_all = np.vstack((feature_score_all, feature_score))
                return feature_score_all

        else:
            now = 0
            for i, index in enumerate(querynum_list):
                f = open(filename + '/' + str(index), 'rb')
                (query, candidate) = pickle.load(f)
                f.close()
                candidate = candidate[:can_num]
                feature_score = get_distribution_query_candidates_doc2vec(query, candidate, frontk=frontk,
                                                                          docv_du=docv_du, is_phoneme=is_phoneme)
                print('%i feature_score has done!' % index)
                feature_score_all = np.vstack((feature_score_all, feature_score))
                if (i + 1) % cut == 0:
                    np.save(save_cut + '/' + str(now) + '.npy', feature_score_all)

                    feature_score_all = np.zeros((0, docv_du))
                    now += 1
            if not feature_score_all.shape[0] == 0:
                np.save(save_cut + '/' + str(now) + '.npy', feature_score_all)

    else:
        print('you need input a querynum!')

    file_open.close()


def write_feature_to_file(feature_score, query_num, file_open, features, docv_du=None):
    file_open.write('# query ' + str(query_num) + '\t features:' + str(features[:]) + '\n')
    if docv_du is None:
        for i in range(len(feature_score)):
            if i == 0:
                writ = '2 qid:%d ' % (query_num)

            else:
                writ = '1 qid:%d ' % (query_num)

            for k, f in enumerate(features):
                writ = writ + str(f) + ':' + str(feature_score[i][k]) + ' '

            file_open.write(writ + '\n')
    else:
        for i in range(len(feature_score)):
            if i == 0:
                writ = '2 qid:%d ' % (query_num)

            else:
                writ = '1 qid:%d ' % (query_num)

            for k in range(docv_du):
                writ = writ + str(k + 1) + ':' + str(feature_score[i][k]) + ' '

            file_open.write(writ + '\n')


def RankSVM_learn(Rank_learn_filename, model_file, parameter):
    cmd = u'svm_rank_learn  -c %f %s %s' % (parameter, Rank_learn_filename, model_file)
    print(cmd)
    os.system(cmd)


def RankSVM_predict(predict_filename, model_file, output_filename):
    cmd = u'svm_rank_classify %s %s %s' % (predict_filename, model_file, output_filename)
    print(cmd)
    os.system(cmd)


def read_prediction_to_matrix(query_num=1000, candidate_num=2000, predict_file='validation_result/prediction0'):
    f = open(predict_file)
    t = f.read().strip()
    k = t.split('\n')
    mat = [float(x) for x in k]
    matrix = np.array(mat)
    matrix = matrix.reshape(query_num, candidate_num)
    return matrix


def get_rank_materix(result_matix):
    query_num = len(result_matix[:, 0])
    ranks = np.zeros(query_num)
    for i in range(query_num):
        true = result_matix[i, 0]
        list = result_matix[i, :].tolist()
        list.sort(reverse=True)
        rank1 = list.index(true) + 1
        list.reverse()
        rank2 = len(list) - list.index(true)
        ranks[i] = rank1 + (rank2 - rank1) * random.random()
    return ranks


def get_result_by_ranks(ranks, rec_k_list):
    result = np.zeros(len(rec_k_list) + 2)
    mean_rank = np.mean(ranks)
    result[0] = mean_rank
    for index, k in enumerate(rec_k_list):
        result[index + 2] = sum(ranks < k + 1) / float(len(ranks))
    mrr_sum = 0.0
    for i in ranks:
        mrr_sum += 1 / float(i)
    result[1] = mrr_sum / len(ranks)
    return result


def Get_rsult_by_predictfile(query_num=1000, candidate_num=2000, predict_file='', rec_k_list=[1, 5, 30, 150]):
    result_matix = read_prediction_to_matrix(query_num=query_num, candidate_num=candidate_num,
                                             predict_file=predict_file)  # 对预测文件读取整理获取他的结果
    ranks = get_rank_materix(result_matix)  # 获取正确结果的排序位置
    result = get_result_by_ranks(ranks, rec_k_list)
    return result
