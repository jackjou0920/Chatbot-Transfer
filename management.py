import uuid
import time
import pytz
from sqlalchemy import create_engine, and_
from sqlalchemy.orm import sessionmaker
from recognition import keyword_extract, check_money
from simlistsel import calculate_similarity
from model import Record, Account
from datetime import datetime as dt

tz = pytz.timezone('Asia/Taipei')


COMFIRM_OK = ["好", "好的", "是", "是的", "對", "對的", "確定", "yes", "Yes"]
COMFIRM_CANCEL = ["錯", "錯的", "不是", "都不是", "不對", "都不對", "否", "以上皆非"]


def open_connection(host="192.168.10.201", user="diia", passwd="diia16313302", db="hibou"):
    sql_connect = "mysql+pymysql://"+user+":"+passwd+"@"+host+"/"+db+"?charset=utf8mb4&binary_prefix=true"
    engine = create_engine(sql_connect, echo=False)
    DBSession = sessionmaker(bind=engine)
    session = DBSession()
    return (session)


def close_connection(session):
    session.close()


def generate_resp(trans_from, trans_to, amount):
    if (len(trans_from) > 1):
        return ("請問您想轉出的帳號是" + ("還是".join(trans_from)) + "？", 2)
    elif (len(trans_to) > 1):
        return ("請問您想轉入的帳號是" + ("還是".join(trans_to)) + "？", 3)
    elif (len(amount) > 1):
        return ("請問您想轉出的金額是" + ("還是".join(amount)) + "元？", 4)
    elif (len(trans_from) == 0):
        return ("請問您想從台幣、薪轉還是數位帳戶轉出呢？", 2)
    elif (len(trans_to) == 0):
        return ("請問您想轉入的帳號是？", 3)
    elif (len(amount) == 0):
        return ("請問您想轉出的金額是？", 4)
    elif (len(trans_from) == 1 and len(trans_to) == 1 and len(amount) == 1):
        return ("請問您是想從" + trans_from[0] + "轉出" + amount[0] + "元到" + trans_to[0] + "嗎？", 1)
    else:
        return ("很抱歉，無法處理您的服務", 0)


def save_record(session, line_record):
    session.add(Record(
        user_id=line_record["user_id"],
        session_id=line_record["session_id"],
        req_sentence=line_record["req_sentence"],
        resp_status=line_record["resp_status"],
        resp_sentence=line_record["resp_sentence"],
        is_transfer=line_record["is_transfer"],
        trans_from=line_record["trans_from"],
        trans_to=line_record["trans_to"],
        amount=line_record["amount"]
    ))
    session.commit()


def dialog(session, tc, ws, pos, ner, all_digital_bank, all_bank_list, trans_from_dict, line_record):
    user_id = line_record["user_id"]
    req_sentence = line_record["req_sentence"]
    response = ""

    # check session_id status and get last record
    record = session.query(Record).filter(and_(Record.user_id == user_id)).order_by(Record.pk_id.desc()).first()
    
    if (record is not None and record.resp_status == 1 and req_sentence in COMFIRM_OK):  # comfirm transfer
        line_record["session_id"] = record.session_id
        line_record["is_transfer"] = record.is_transfer
        line_record["trans_from"] = record.trans_from
        line_record["trans_to"] = record.trans_to
        line_record["amount"] = record.amount

        trans_from = [] if (record.trans_from is None) else record.trans_from.split(",")
        trans_to = [] if (record.trans_to is None) else record.trans_to.split(",")
        amount = [] if (record.amount is None) else record.amount.split(",")

        if (len(trans_from) == 1 and len(trans_to) == 1 and len(amount) == 1):
            line_record["resp_status"] = 1
            line_record["resp_sentence"] = "轉帳成功！"

            result = session.query(Account).filter(and_(
                Account.user_id == user_id, Account.nickname == trans_to[0]
            )).first()
            from_ = trans_from[0] + "(" + trans_from_dict[trans_from[0]] + ")"
            to_ = trans_to[0] + "(" + result.account_number + ")"

            response = ("{:6}: {}".format("轉出帳號", from_)) + "\n"
            response += ("{:6}: {}".format("轉入帳號", to_)) + "\n"
            response += ("{:6}: {}".format("金額", amount[0])) + "\n\n"
            response += "轉帳成功！"
        else:
            response = "很抱歉，無法處理您的服務"
            line_record["resp_status"] = 0
            line_record["resp_sentence"] = response
    elif (record is not None and req_sentence in COMFIRM_CANCEL):   # cancel transfer
        line_record["session_id"] = record.session_id
        line_record["is_transfer"] = record.is_transfer
        line_record["trans_from"] = record.trans_from
        line_record["trans_to"] = record.trans_to
        line_record["amount"] = record.amount
        line_record["resp_status"] = 0

        response = "轉帳取消，請重新輸入！"
        line_record["resp_sentence"] = response
    elif (record is None or record.resp_status == 0 or record.resp_status == 1):  # new session
        start_cls = time.time()
        is_transfer = tc.predict([req_sentence])
        print("predict classification: " + str(time.time()-start_cls) + "s")

        line_record["session_id"] = uuid.uuid1()
        line_record["is_transfer"] = is_transfer

        if is_transfer:
            # get user all account number
            nickname_dict = dict()
            acc_result = session.query(Account).filter(Account.user_id == user_id).order_by(Account.pk_id).all()
            for r in acc_result:
                nickname_dict[r.nickname] = r.account_number

            # extract keywords
            trans_from, trans_to, amount = keyword_extract(ws, pos, ner, req_sentence, all_digital_bank, all_bank_list)
            line_record["amount"] = ",".join(amount) if (len(amount) != 0) else None

            # check similarity with trans_from
            if (len(trans_from) != 0):
                trans_from = calculate_similarity(trans_from, list(trans_from_dict.keys()), level=2)
                line_record["trans_from"] = ",".join(trans_from) if (len(trans_from) != 0) else None
                # full up account number
                for i, from_ in enumerate(trans_from):
                    trans_from[i] = from_ + "(" + trans_from_dict[from_] + ")"

            # check similarity with trans_to nickname
            if (len(trans_to) != 0 and acc_result is not None):
                trans_to = calculate_similarity(trans_to, list(nickname_dict.keys()), level=2)
                line_record["trans_to"] = ",".join(trans_to) if (len(trans_to) != 0) else None
                # full up account number
                for i, to in enumerate(trans_to):
                    trans_to[i] = to + "(" + nickname_dict[to] + ")"

            # generate response question
            response, status = generate_resp(trans_from, trans_to, amount)
            line_record["resp_status"] = status
            line_record["resp_sentence"] = response
            if (len(trans_from) == 0 and status == 2):
                response += "\n\n"
                for i, (key, value) in enumerate(trans_from_dict.items()):
                    response += str(i+1) + ". " + key + "(" + value + ")\n"
                response = response[:-1]
            
            if (len(trans_to) == 0 and status == 3):
                response += "\n\n"
                for i, (key, value) in enumerate(nickname_dict.items()):
                    response += str(i+1) + ". " + key + "(" + value + ")\n"
                response = response[:-1]

            # response += ("{:8}: {}".format("TransFr", "None" if (len(trans_from) == 0) else ", ".join(trans_from))) + "\n"
            # response += ("{:8}: {}".format("TransTo", "None" if (len(trans_to) == 0) else ", ".join(trans_to))) + "\n"
            # response += ("{:8}: {}".format("Amount", "None" if (len(amount) == 0) else ", ".join(amount)))
        else:
            response = "很抱歉，無法處理您的服務，請重新輸入轉帳語句"
            line_record["resp_status"] = 0
            line_record["resp_sentence"] = response
    else:  # incomplete session
        line_record["session_id"] = record.session_id
        line_record["is_transfer"] = record.is_transfer
        line_record["trans_from"] = record.trans_from
        line_record["trans_to"] = record.trans_to
        line_record["amount"] = record.amount
        trans_from = [] if (record.trans_from is None) else record.trans_from.split(",")
        trans_to = [] if (record.trans_to is None) else record.trans_to.split(",")
        amount = [] if (record.amount is None) else record.amount.split(",")

        # get user all account number
        nickname_dict = dict()
        acc_result = session.query(Account).filter(Account.user_id == user_id).order_by(Account.pk_id).all()
        for r in acc_result:
            nickname_dict[r.nickname] = r.account_number

        if (record.resp_status == 2 and len(trans_from) != 1):
            trans_from_list = list(trans_from_dict.keys())
            try:
                idx = int(req_sentence)
                trans_from = [trans_from_list[idx-1]]
            except:
                # check similarity with trans_from
                trans_from = calculate_similarity([req_sentence], trans_from_list, level=4)
            line_record["trans_from"] = ",".join(trans_from) if (len(trans_from) != 0) else None
        elif (record.resp_status == 3 and len(trans_to) != 1):
            nickname_list = list(nickname_dict.keys())
            try:
                idx = int(req_sentence)
                trans_to = [nickname_list[idx-1]]
            except:
                # check similarity with trans_to nickname
                if (len(trans_to) == 0):
                    trans_to = calculate_similarity([req_sentence], nickname_list, level=4)
                else:
                    trans_to = calculate_similarity([req_sentence], trans_to, level=4)
            line_record["trans_to"] = ",".join(trans_to) if (len(trans_to) != 0) else None
        elif (record.resp_status == 4 and len(amount) != 1):  # amount
            if check_money(req_sentence) is not None:
                money = str(check_money(req_sentence))
            if (len(amount) == 0 or money in amount):
                amount = [money]
            line_record["amount"] = ",".join(amount) if (len(amount) != 0) else None

        # full up account number
        for i, from_ in enumerate(trans_from):
            trans_from[i] = from_ + "(" + trans_from_dict[from_] + ")"
        # full up account number
        for i, to in enumerate(trans_to):
            trans_to[i] = to + "(" + nickname_dict[to] + ")"

        # generate response question
        response, status = generate_resp(trans_from, trans_to, amount)
        line_record["resp_status"] = status
        line_record["resp_sentence"] = response
        if (len(trans_from) == 0 and status == 2):
            response += "\n\n"
            for i, (key, value) in enumerate(trans_from_dict.items()):
                response += str(i+1) + ". " + key + "(" + value + ")\n"
            response = response[:-1]
        
        if (len(trans_to) == 0 and status == 3):
            response += "\n\n"
            for i, (key, value) in enumerate(nickname_dict.items()):
                response += str(i+1) + ". " + key + "(" + value + ")\n"
            response = response[:-1]

    save_record(session, line_record)
    return (response)


if __name__ == "__main__":
    dialog()
