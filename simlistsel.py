import difflib
import numpy as np
import pandas as pd
import copy, time, math
from numpy import array
from prettytable import PrettyTable


# 定義熵值法函數
def cal_weight(x):
    '''熵值法計算變量的權重'''

    # 標准化
    x = x.apply(lambda x: ((x-np.min(x)) / (np.max(x)-np.min(x))))

    # 求k
    rows = x.index.size  # 行
    cols = x.columns.size  # 列
    k = 1.0 / math.log(rows)
    lnf = [[None] * cols for i in range(rows)]

    # 矩陣計算--
    # 信息熵
    # p=array(p)
    x = array(x)
    lnf = [[None] * cols for i in range(rows)]
    lnf = array(lnf)
    for i in range(0, rows):
        for j in range(0, cols):
            if x[i][j] == 0:
                lnfij = 0.0
            else:
                p = x[i][j] / x.sum(axis=0)[j]
                lnfij = math.log(p) * p * (-k)
            lnf[i][j] = lnfij
    lnf = pd.DataFrame(lnf)
    E = lnf

    # 計算冗余度
    d = 1 - E.sum(axis=0)
    # 計算各指標的權重
    w = [[None] * 1 for i in range(cols)]
    for j in range(0, cols):
        wj = d[j] / sum(d)
        w[j] = wj
        # 計算各樣本的綜合得分,用最原始的數據

    # w = pd.DataFrame(w)
    return w


def self_argument(self):
    c = self.c
    one_u = c.list["onepass_user"]
    one_s = c.list["onepass_score"]
    one_v = c.list["onepass_vocNmmb"]
    try:
        two_u = c.listwo["twopass_user"]
        two_s = c.listwo["twopass_score"]
        two_v = c.listwo["twopass_vocNmmb"]
    except:
        two_u = ["nan"]*len(one_u)
        two_s = [0]*len(one_u)
        two_v = [0]*len(one_u)

    try:
        three_p = c.listthree["pandas_list_three"]
        three_w = c.listthree["weight"]
    except:
        three_p = [[0]]*len(one_u)
        three_w = [0]*len(one_u)

    try:
        three_p = c.listthree["pandas_list_three"]
        three_w = c.listthree["weight"]
    except:
        three_p = [[0]]*len(one_u)
        three_w = [0]*len(one_u)
    return (one_u, one_s, one_v, two_u, two_s, two_v, three_p, three_w)


class HiBOUSelect(object):
    def __init__(self, sentece):
        self.c = self
        self.example_sentece = sentece

    @property
    def onepasssummary(self):
        return self._onepasssummary

    @property
    def twopasssummary(self):
        return self._twopasssummary

    @property
    def threepasssummary(self):
        return self._threepasssummary

    @onepasssummary.setter
    def onepasssummary(self, val):
        self.list = val

    @twopasssummary.setter
    def twopasssummary(self, val):
        self.listwo = val

    @threepasssummary.setter
    def threepasssummary(self, val):
        self.listthree = val

    def filter_layer(self, list_, threshold=0.5):
        list_onepass_user = []
        list_times = []
        list_score = []
        example_sentece = self.example_sentece

        for user in list_:
            slice_txt = [example_sentece[sent:sent+len(user)] for sent in range(len(example_sentece)-len(user)+1)]
            similarity = difflib.get_close_matches(user, slice_txt, cutoff=threshold)
            score = [difflib.SequenceMatcher(None, user, i).ratio() for i in similarity]

            if len(score) > 0:
                list_score.append(np.mean(score))

            # 取出有根據自有關聯的清單名稱
            if len(similarity) > 0:
                list_onepass_user.append(user)

            if len(similarity) > 0:
                list_times.append(len(similarity))

        c = self.c
        c.onepasssummary = {
            "onepass_user": list_onepass_user,
            "onepass_score": list_score,
            "onepass_vocNmmb": list_times,
        }
        # print(c.list)
        return list_onepass_user

    def sc_layer(self, list, copy_sentece):
        list_sample = []
        list_sample_score = []
        for i in list:
            s = 0
            for ii in copy_sentece:
                if ii in i:
                    s += 1
            # 必須注意只有1的長度0/1=0
            if len(copy_sentece) == 1:
                allen = 1/len(copy_sentece)
            else:
                allen = (len(copy_sentece)-1)/len(copy_sentece)

            if s/len(copy_sentece) >= allen:
                list_sample.append(i)
                list_sample_score.append(s/len(copy_sentece))

        c = self.c
        c.onepasssummary = {
            "onepass_user": list_sample,
            "onepass_score": list_sample_score,
            "onepass_vocNmmb": [0]*len(list_sample),
        }
        return list_sample

    def modify_layer(self, list_onepass_user):
        list_fix = []
        # list_onepass_user = list_onepass_user2[0]

        for user in list_onepass_user:
            for delete in (set(list_onepass_user[0]) & set(list_onepass_user[1])):
                user = user.replace(delete, "", 1)
            list_fix.append(user)
        return [list_fix, list_onepass_user]

    def streng_layer(self, list_fix2, times=1, threshold=0.1):
        list_mod = list_fix2[0]  ## 兩清單相同文字刪除
        list_onepass_user = list_fix2[1]
        list_twopass_user = []
        list_score = []
        list_result = []
        example_sentece = self.example_sentece
        for user in list_mod:
            # 因為刪除相同字串，可能導致清單便空值，需要除錯機制
            try:
                # 疊字最後一個字，加強辨識分數
                slice_txt = [example_sentece[sent:sent+len(user)]+(example_sentece[sent:sent+len(user)][-1])*times for sent in range(len(example_sentece)-len(user)+1)]
                similarity = difflib.get_close_matches(user+user[-1], slice_txt, cutoff=threshold)
                score = [difflib.SequenceMatcher(None, user, i).ratio() for i in similarity]
                if len(score) > 0:
                    list_score.append(np.mean(score))
                else:
                    list_score.append(0)

                # 取出有根據自有關聯的清單名稱
                if len(similarity) > 0:
                    list_twopass_user.append(user)
                else:
                    list_twopass_user.append("")

                if len(similarity) > 0:
                    list_result.append(len(similarity))
                else:
                    list_result.append(0)
            except:
                list_result.append(0)

        c = self.c
        c.twopasssummary = {
            "twopass_user": list_mod,
            "twopass_score": list_score,
            "twopass_vocNmmb": list_result,
        }
        # print(c.listwo)
        return [list_result, list_onepass_user, list_score]

    def reward_layer(self, list_result2):
        list_table = []
        example_sentece = self.example_sentece
        one_u, one_s, one_v, two_u, two_s, two_v, _, _ = self_argument(self)

        len_one_u = list(map(lambda x: len(x), one_u))
        len_two_u = list(map(lambda x: len(x), two_u))
        list_three_filter = []
        for i in two_u:
            copy_sentece = copy.copy(example_sentece)
            copy_i = copy.copy(i)
            for ii in i:
                if ii in copy_sentece:
                    copy_sentece = copy_sentece.replace(ii, "", 1)
                    before_i = copy.copy(copy_i)
                    copy_i = copy_i.replace(ii, "", 1)
                    if len(copy_i) == 0:
                        copy_i = before_i
            list_three_filter.append(copy_i)

        list_threepass_user = []
        list_three_times = []
        list_three_score = []
        for user in list_three_filter:
            if len(user) == 0:
                score = []
                similarity = []
            else:
                slice_txt = [example_sentece[sent:sent+len(user)] for sent in range(len(example_sentece)-len(user)+1)]
                similarity = difflib.get_close_matches(user, slice_txt, cutoff=0.5)
                score = [difflib.SequenceMatcher(None, user, i).ratio() for i in similarity]

            if len(score) > 0:
                list_three_score.append(np.mean(score))
            else:
                list_three_score.append(0)

            # 取出有根據自有關聯的清單名稱
            if len(similarity) > 0:
                list_threepass_user.append(user)
            else:
                list_threepass_user.append("")
            if len(similarity) > 0:
                list_three_times.append(len(similarity))
            else:
                list_three_times.append(0)

        for i in range(len(one_u)):
            lenword = len(one_u[i]) - min(len_one_u)
            # basevalue = one_s[len_one_u.index(min(len_one_u))] / min(len_one_u)
            # if min(len_one_u) <4:
            #     scorev3 = round(two_s[i],3) + basevalue*lenword * one_v[i] + basevalue * two_v[i]
            # else:
            scorev3 = list_three_score[i]
            list_table.append([
                lenword,  # 字數差
                round(one_s[i], 3),  # 過濾一分數
                one_v[i], round(two_s[i], 3), two_v[i], len_two_u[i], scorev3
            ])

        dict_df = {}
        for t, i in enumerate(list_table):
            dict_df["x"+str(t)] = i

        df = pd.DataFrame(dict_df)
        # print(df)
        w = cal_weight(df)  # 調用cal_weight
        # w.index = df.columns
        # w.columns = ['weight']
        # print(w)
        c = self.c
        c.threepasssummary = {"pandas_list_three": list_table, "weight": w}
        return list_result2

    def output_layer(self, list_result2, level=3):
        def get_index1(lst=None, item=''):
            tmp = []
            tag = 0
            for i in lst:
                if i == item:
                    tmp.append(tag)
                tag += 1
            return tmp

        one_u, one_s, one_v, two_u, two_s, two_v, three_p, three_w = self_argument(self)
        list_onepass_user = one_u  # 符合句子的數量

        if level == 1:
            list_score = one_s  # 分數
            result = [list_onepass_user[i] for i in range(len(list_score)) if (list_score[i] > 0)]
            return (result)
        elif level == 2:
            list_score = two_s
            if max(list_score) == 0:
                list_score = one_s
            result = [list_onepass_user[i] for i in range(len(list_score)) if (list_score[i] > 0)]
            return (result)
        elif level == 3:
            try:
                list_score = [round(round(three_p[i][-1], 3), 3) for i in range(len(one_u))]
            except :
                list_score = two_s  # 分數
            result = [list_onepass_user[i] for i in range(len(list_score)) if (list_score[i] > 0)]
            return (result)
        elif level == 4:
            try:
                # list_score = [round(round(three_p[i][-1], 3)*round(1-round(three_w[i], 3), 3), 3) for i in range(len(one_u))]
                list_score = list(map(lambda x: 1-x, three_w))
                if max(list_score) == 0:
                    if max(two_s) == 0:
                        list_score = list(map(lambda x: 1-x, three_w))
                    else:
                        list_score = two_s
            except :
                list_score = two_s  # 分數
            result = [list_onepass_user[i] for i in range(len(list_score)) if (list_score[i] == max(list_score))]
            return (result)
        else:
            # result = [{list_onepass_user[i]:max(list_score)} for i in get_index1(list_score, max(list_score))]
            # list_recommend = copy.copy(list_onepass_user)
            # for i in result:
            #     list_recommend.remove(list(i.keys())[0])
            return ([])
        

    def summary(self):
        one_u, one_s, one_v, two_u, two_s, two_v, three_p, three_w = self_argument(self)
        list_table = []
        for i in range(len(one_u)):
            list_table.append([
                one_u[i],
                round(one_s[i], 3), one_v[i], two_u[i], round(two_s[i], 3), two_v[i],
                round(three_p[i][-1], 3),
                round(1-round(three_w[i], 3), 3),
                round(round(three_p[i][-1], 3)*round(1-round(three_w[i], 3), 3), 3)
            ])
        print("")
        print("Summary:")
        print("======================================================")

        table = PrettyTable([
            "name", "lv1 Score", "lv1 Quantity", "lv2 filter", "lv2 Score", "lv2 Quantity", "lv3 Score", "lv3 Weight", "result Score"
        ])
        for r in list_table:
            table.add_row(r)
        table.vertical_char = ' '

        print(table)
        return "======================================================"


def main_layer(list_, example_sentece, threshold=0.5, level=2):
    copy_sentece = copy.copy(example_sentece)
    max_list_len = max(map(lambda x: len(x), list_))
    if (max_list_len > len(example_sentece)):
        example_sentece = example_sentece * math.ceil(max_list_len/len(example_sentece))
    elif (max_list_len == len(example_sentece)):
        example_sentece = example_sentece * 2

    hb = HiBOUSelect(example_sentece)
    x = hb.filter_layer(list_, threshold)  # 過濾層

    # 決策層
    if (len(x) == 1):
        return (x, hb)
    elif (len(x) > 1):
        x = hb.modify_layer(x)  # 修改層
        x = hb.streng_layer(x)  # 強化層
        x = hb.reward_layer(x)  # 獎勵層(權重層)
        x = hb.output_layer(x, level)  # 輸出層
        return (x, hb)
    else:  # 抽樣比對 Sampling comparison
        try:
            x = hb.sc_layer(list_, copy_sentece)  # 抽樣比對層
            x = hb.modify_layer(x)
            x = hb.streng_layer(x)
            x = hb.reward_layer(x)
            x = hb.output_layer(x, level)
            return (x, hb)
        except:
            return ([], None)
        # x = hb.sc_layer(list,copy_sentece)#抽樣比對層
        # x = hb.modify_layer(x)
        # x = hb.streng_layer(x)
        # x = hb.reward_layer(x)
        # x,rec = hb.output_layer(x,level)
        # return x,rec,hb.summary()


def calculate_similarity(sentence_list, nickname_list, level=2):
    start_sim = time.time()
    candidate_list = list()
    for sentence in sentence_list:
        result, hb = main_layer(nickname_list, sentence, threshold=0.5, level=level)
        candidate_list += result
        if (hb is not None):
            hb.summary()
        print(sentence, result)
    print("calculate similarity: " + str(time.time()-start_sim) + "s")
    return (list(set(candidate_list)))


if __name__ == '__main__':
    # list1 = ['阿弟', '老爸台新', '媽媽一銀', '老媽台新', '媽媽的第一銀行', '老媽彰銀', '爸第一', "陳台新台新", "鋼琴家教"]
    list1 = ['台新Richart', '永豐大戶', '傑克台北富邦', '元大交割戶', '英國匯豐', '伶伃的koko']
    # example_sentece = ['媽媽一銀', '媽媽彰銀']
    example_sentece = ['伶伃koko']
    calculate_similarity(example_sentece, list1)
