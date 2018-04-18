import phonetics as ph
import string
import numpy as np
import scipy.spatial.distance as dist


class rhyming:
    def __init__(self, query="4-5-6-7 chains on", rhyme_query='fo@ TaUz@nd faIvhVndr@d sIkstisEv@n tSeInz 0n',
                 candidate="iPod, the coolet muthafucka that you prolly ever saw",
                 rhyme_can='aI p0d D@2 ku:l@t mVTa#fVk@ Dat ju: proUli; Ev3 sO:'):
        self.text1 = query
        self.text2 = candidate
        self.word_1 = []  # List of words in lyrics_1
        self.word_2 = []  # List of words in the lyrics_2
        self.vowel_1 = []  # List of vowels of each word in lyrics_1
        self.vowel_2 = []  # List of vowels of each word in lyrics_2
        self.all_vowel_1 = []  # List of vowels in the lyrics_1
        self.all_vowel_2 = []  # List of vowels in the lyrics_2
        self.rhyme_index1 = []
        self.rhyme_index2 = []

        self.word_1 = query.strip().split(' ')
        self.word_2 = candidate.strip().split(' ')
        self.rhyme1 = rhyme_query
        self.rhyme2 = rhyme_can
        self.rhyme_p_1 = rhyme_query.strip().split(' ')  # List of rhyme of each word in lyrics_1
        self.rhyme_p_2 = rhyme_can.strip().split(' ')  # List of rhyme of each word in lyrics_2
        r1 = self.rhyme_p_1
        r2 = self.rhyme_p_2
        (self.vowel_1, self.rhyme_index1, self.all_vowel_1) = self.Get_vowel(r1)
        (self.vowel_2, self.rhyme_index2, self.all_vowel_2) = self.Get_vowel(r2)

    def get_endrhyme(self):
        self.endrhyme = self.EndRhyme()
        return self.endrhyme

    def get_otherrhyme(self):
        self.otherrhyme = self.OtherRhyme()
        return self.otherrhyme

    def get_lineLength(self):
        self.lineLength = self.LineLength()
        return self.lineLength

    def LineLength(self):
        delset = string.punctuation
        transtab = str.maketrans('', '', delset)
        len_l = len(self.text1.translate(transtab).replace(' ', ''))
        len_s = len(self.text2.translate(transtab).replace(' ', ''))
        return 1 - float(abs(len_l - len_s)) / max(len_l, len_s)

    def get_bow(self, front5=None):
        k = self.text1.strip().split()
        if front5 is None:
            m = self.text2.strip().split()
        else:
            m = front5.replace('\n', ' ').split(' ')
        k.extend(m)
        dic = set(k)
        victer = np.zeros((2, len(dic)))
        for i, w in enumerate(dic):
            if w in self.word_1:
                victer[0, i] = 1
            if w in m:
                victer[1, i] = 1
        a = dist.pdist(victer, 'jaccard')
        return a

    def GetRhyme(self, text):
        delset = string.punctuation
        transtab = str.maketrans('', '', delset)
        text = text.translate(transtab)
        return ph.get_phonetic_transcription(text, language='en-g')

    def EndRhyme(self):
        end = 0
        if len(self.all_vowel_1) > 0 and len(self.all_vowel_2) > 0:
            while (self.all_vowel_1[-1 - end] == self.all_vowel_2[-1 - end]):
                end += 1

                if (end >= len(self.all_vowel_1) or end >= len(self.all_vowel_2)):
                    break
        return end

    def OtherRhyme(self):
        score = []

        for i in range(len(self.vowel_1)):
            sc_i = 0
            for j in range(len(self.vowel_2)):
                w_i = i
                w_j = j
                if w_i >= len(self.word_1):
                    w_i = len(self.word_1) - 1
                if w_j >= len(self.word_2):
                    w_j = len(self.word_2) - 1
                sc = self.Get_longest_match_w2w(self.vowel_1[i], self.vowel_2[j], self.word_1[w_i], self.word_2[w_j],
                                                self.rhyme_index1[i], self.rhyme_index2[j])

                if sc > sc_i:
                    sc_i = sc
            score.append(sc_i)
        rls = np.array(score)
        return np.mean(rls)

    def Get_longest_match_w2w(self, word_v_1, word_v_2, word1, word2, w_in1, w_in2):
        end = 0

        if w_in1 > -1 and w_in2 > -1:

            if word1[0:w_in1] != word2[0:w_in2]:

                while word_v_1[-1 - end] == word_v_2[-1 - end]:
                    end += 1
                    if end >= len(word_v_1) or end >= len(word_v_2):
                        break
        return end

    def Get_vowel(self, rhyme_list):
        vowel_list = []
        vowel_last_index = []
        all_vowel = []
        for k in range(len(rhyme_list)):
            vowel = ''
            index = -1
            for j in range(len(rhyme_list[k])):

                c = ph.map_vow(rhyme_list[k][j], 'en-g')
                if ph.is_vow(c, 'en-g'):
                    vowel = vowel + c
                    index = j
                    all_vowel.append(c)
            vowel_list.append(vowel)
            vowel_last_index.append(index)
        return vowel_list, vowel_last_index, all_vowel
