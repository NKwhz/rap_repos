import numpy as np
import string
from gensim.models.doc2vec import Doc2Vec,TaggedDocument
delset=string.punctuation
def get_Doc2vec_sim(model, query, candiadate):


    query = query.replace('  ', ' ')
    # candiadate = candiadate.replace('  ', ' ')
    query_list = query.split(' ')

    rhyme1_vec = model.infer_vector(query_list)
    # rhyme2_vec = np.zeros((len(candiadate),len(rhyme1_vec)))
    score=np.zeros(len(candiadate))
    vector=np.zeros((len(candiadate),2*len(rhyme1_vec)))
    for i,can in enumerate(candiadate):
        can = can.replace('  ', ' ')
        candiadate_list = can.split(' ')
        rhyme2_vec=model.infer_vector(candiadate_list)
        cos_dis = np.dot(rhyme1_vec, rhyme2_vec) / (np.linalg.norm(rhyme1_vec) * np.linalg.norm(rhyme2_vec))
        score[i]=cos_dis
    return score
    # similarity_lsi = similarities.Similarity('Similarity-LSI-index', train, num_features=400)
    # return similarity_lsi[test]

    # return cos_dis
def get_Doc2vec_vector(model, query, candiadate,docvec_du=500):
    all = query + ' ' + candiadate
    all = all.replace('  ', ' ')
    all_list = all.split(' ')
    rhyme2_vec = model.infer_vector(all_list)

    return rhyme2_vec
def get_Doc2vec_vector_phoneme(model, query, candiadate,docvec_du=500):
    all = query + '\n' + candiadate
    transtab = str.maketrans('', '', delset)
    all = all.replace('  ', ' ').translate(transtab)
    all_list =list(all)
    rhyme2_vec = model.infer_vector(all_list)
    return rhyme2_vec


if __name__ == '__main__':
    pass