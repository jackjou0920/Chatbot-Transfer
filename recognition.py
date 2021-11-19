import re
import time
import pandas as pd
from model import Record
from ckiptagger import construct_dictionary, WS, POS, NER


CN_NUM = {'〇': 0, '一': 1, '二': 2, '三': 3, '四': 4, '五': 5, '六': 6, '七': 7, '八': 8, '九': 9, '零': 0, '壹': 1, '貳': 2, '叄': 3, '肆': 4, '伍': 5, '陸': 6, '柒': 7, '捌': 8, '玖': 9, '貮': 2, '兩': 2}
CN_UNIT = {'十': 10, '拾': 10, '百': 100, '佰': 100, '千': 1000, '仟': 1000, '萬': 10000, '億': 100000000, '兆': 1000000000000}
TRANS_FROM = ["薪轉帳戶", "台幣帳戶", "數位帳戶", "薪轉戶", "台幣戶", "數位戶", "薪轉"]
ACTION = ['轉帳', '匯款', '轉錢', '轉出', '匯出', '提撥']


def chinese_to_arabic(cn: str) -> int:
    unit = 0
    ldig = list()
    for cndig in reversed(cn):
        if cndig in CN_UNIT:
            unit = CN_UNIT.get(cndig)
            if unit == 10000 or unit == 100000000:
                ldig.append(unit)
                unit = 1
        elif cndig in CN_NUM:
            dig = CN_NUM.get(cndig)
            if unit:
                dig *= unit
                unit = 0
                ldig.append(dig)
                if unit == 10:
                    ldig.append(10)
            else:
                ldig.append(dig)
        else:
            if cndig.isdigit():
                ldig.append(int(cndig))

    if unit:
        ldig.append(unit)

    val, tmp = 0, 0
    for x in reversed(ldig):
        if x == 10000 or x == 100000000:
            val += tmp * x
            tmp = 0
        else:
            tmp += x
    val += tmp
    return val


def check_chinese_number(text):
    chinese_number = list(CN_NUM.keys()) + list(CN_UNIT.keys())
    for txt in text:
        if (txt not in chinese_number):
            return False
    return True


def check_money(money):
    remove_chars = '[A-Za-z’!"#$%&\'()*+,-./:;<=>?@，。?★、…【】《》？“”‘’！[\\]^_`{|}~]+'
    money = re.sub(remove_chars, '', money)
    money_digit = money.replace('塊錢', "").replace('元', "").replace('塊', "")

    if money_digit.isdigit():
        return (money_digit)
    elif (check_chinese_number(money_digit)):
        return (chinese_to_arabic(money_digit))
    else:
        return None


def get_amount(money_list, cardinal_list):
    amount_list = list()
    for money in money_list:
        if check_money(money) is not None:
            amount_list.append(check_money(money))

    if (len(amount_list) == 0):
        for cardinal in cardinal_list:
            if check_money(cardinal) is not None:
                amount_list.append(check_money(cardinal))
    return (amount_list)


def get_element_from_list(word_list, compare_list):
    return (list(set(word_list) & set(compare_list)))


def get_pos_noun_word(word_sentence, pos_sentence):
    assert len(word_sentence) == len(pos_sentence)
    pos_noun = ['Na', 'Nb', 'Nc', 'FW']
    noun_list = [word for word, pos in zip(word_sentence, pos_sentence) if (pos in pos_noun)]
    return (list(set(noun_list)))


def get_entity_amount(entity_set):
    money, cardinal = list(), list()
    for entity in entity_set:
        if (entity[2] == "MONEY"):
            money.append(entity[3])
        elif (entity[2] == "CARDINAL"):
            cardinal.append(entity[3])

    amount_list = get_amount(money, cardinal)
    return (amount_list)


def get_entity_person(entity_set):
    pserson_list = [entity[3] for entity in entity_set if (entity[2] == "PERSON")]
    return (list(set(pserson_list)))


def transto_candidate(subject_list, bank_list):
    if (len(bank_list) == 0):
        return (list(set(subject_list)))
    if (len(subject_list) == 0):
        return (list(set(bank_list)))

    candidate = list()
    for subject in subject_list:
        for bank in bank_list:
            candidate.append(subject+bank)
    return (list(set(candidate)))


def keyword_extract(ws, pos, ner, req_sentence, all_digital_bank, all_bank_list):
    req_sentence = req_sentence.strip()

    # prepare weight dictionary
    ws_input = req_sentence.replace(" ", ",")
    action_dict = dict.fromkeys(ACTION , 2)
    digital_dict = dict.fromkeys(all_digital_bank , 2)
    bank_dict = dict.fromkeys(TRANS_FROM + all_bank_list , 1)
    word_to_weight = dict(action_dict.items() | digital_dict.items() | bank_dict.items())

    # ckip tagger
    start_ckip = time.time()
    dictionary = construct_dictionary(word_to_weight)
    word_sentence_list = ws([ws_input], coerce_dictionary=dictionary)
    pos_sentence_list = pos(word_sentence_list)
    entity_sentence_list = ner(word_sentence_list, pos_sentence_list)
    print("predict ws/pos/ner: " + str(time.time()-start_ckip) + "s")
    print(word_sentence_list)
    print(pos_sentence_list)
    print(entity_sentence_list)
    print("*"*50)

    # extract all category list
    trans_from = get_element_from_list(word_sentence_list[0], TRANS_FROM)
    print("trans_from:", trans_from)

    digital_bank = get_element_from_list(word_sentence_list[0], all_digital_bank)
    print("digital_bank:", digital_bank)

    bank_list = get_element_from_list(word_sentence_list[0], all_bank_list)
    print("bank_list:", bank_list)
    bank_list = list(set(bank_list+digital_bank))

    noun_list = get_pos_noun_word(word_sentence_list[0],  pos_sentence_list[0])
    noun_list = list(set(noun_list)-set(trans_from+bank_list))
    print("noun_list:", noun_list)

    person_list = get_entity_person(entity_sentence_list[0])
    person_list = list(set(person_list)-set(trans_from+bank_list))
    print("person_list:", person_list)
    subject_list = list(set(person_list+noun_list))
    
    amount_list = get_entity_amount(entity_sentence_list[0])
    amount_list = [str(amount) for amount in amount_list]
    print("amount_list:", amount_list)

    trans_to = transto_candidate(subject_list, bank_list)
    print("*"*50)
    return (trans_from, trans_to, amount_list)


if __name__ == "__main__":
    start = time.time()

    # Load model with GPU
    ws = WS("./data", disable_cuda=True)
    pos = POS("./data", disable_cuda=False)
    ner = NER("./data", disable_cuda=False)
    print("Successfully load model: " + str(time.time()-start) + "s")

    df = pd.read_csv("./ner/data_subject.csv")
    all_bank_list = df["bank1"][df['bank1'].isna()==False].values.tolist() + \
                    df["bank2"][df['bank2'].isna()==False].values.tolist() + \
                    df["bank3"][df['bank3'].isna()==False].values.tolist() + \
                    df["bank4"][df['bank4'].isna()==False].values.tolist()
    all_digital_bank = df['digital'][df['digital'].isna()==False].values.tolist()

    req_sentence = "Bruce要從他的薪轉轉帳給Jack的E財寶1003塊"
    # req_sentence = "從上海轉500給媽媽的土地"
    print("req_sentence:", req_sentence)

    start = time.time()
    trans_from, trans_to, amount_list = keyword_extract(ws, pos, ner, req_sentence, all_digital_bank, all_bank_list)
    print(trans_from, trans_to, amount_list)
    print("took time: " + str(time.time()-start) + "s")
