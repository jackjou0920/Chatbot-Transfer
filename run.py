import time, sys
import pandas as pd
from transfer_classification import TransferClassifier
from ckiptagger import WS, POS, NER
from recognition import keyword_extract
from management import open_connection, close_connection, dialog
from model import Account
from sqlalchemy import and_, or_
from flask import Flask, request, abort
from linebot import LineBotApi, WebhookHandler
from linebot.exceptions import InvalidSignatureError
from linebot.models import MessageEvent, TextMessage, TextSendMessage
from argparse import ArgumentParser


line_bot_api = LineBotApi('vroZ6O7nBStiHuUt/0y0k1fRgFOKVmu8L95O2Opugvui1hqjbb5tbnryly149ezaC2Oseg/arUWbj8og1v+uLXvIPCfIyMSus4j+dsoBfh+gV5oFRlVMqYKvoobMzrp0tE3mej2Byy76zty2fbWXTgdB04t89/1O/w1cDnyilFU=')
handler = WebhookHandler('0df34aad2df527da2fed3dc3dabdbf83')
# ========================================================================================================

parser = ArgumentParser()
parser.add_argument('--cpu', action='store_true', help='run with cpu')
parser.add_argument('--gpu', action='store_true', help='run with gpu')
args = parser.parse_args()

if (args.cpu and args.gpu):
    sys.exit()

if args.cpu:  # Load model with CPU
    start = time.time()
    tc = TransferClassifier(mode="cpu", model_path="./classification/model_classification_cpu_epoch_1_batch_32")
    time_load_bert = time.time() - start

    start_load_ckip = time.time()
    ws = WS("./data")
    pos = POS("./data")
    ner = NER("./data")
    print("Successfully load BERT model: " + str(time_load_bert) + "s")
    print("Successfully load CKIP model: " + str(time.time()-start_load_ckip) + "s")
else:  # Load model with GPU
    start = time.time()
    tc = TransferClassifier(model_path="./classification/model_classification_gpu_epoch_1_batch_64")
    # tn = TransferNer(model_path="./ner/model_ner_gpu_epoch_2_batch_64")
    time_load_bert = time.time() - start

    start_load_ckip = time.time()
    ws = WS("./data", disable_cuda=False)
    pos = POS("./data", disable_cuda=False)
    ner = NER("./data", disable_cuda=False)
    print("Successfully load BERT model: " + str(time_load_bert) + "s")
    print("Successfully load CKIP model: " + str(time.time()-start_load_ckip) + "s")


df = pd.read_csv("./ner/data_subject.csv")
all_bank_list = df["bank1"][df['bank1'].isna()==False].values.tolist() + \
                df["bank2"][df['bank2'].isna()==False].values.tolist() + \
                df["bank3"][df['bank3'].isna()==False].values.tolist() + \
                df["bank4"][df['bank4'].isna()==False].values.tolist()
all_digital_bank = df['digital'][df['digital'].isna()==False].values.tolist()

trans_from_dict = {"台幣帳戶": "013272506140153",  "數位帳戶": "013699504118537", "薪轉帳戶": "81220651000284021"}

session = open_connection()
print("Successfully connect to db...")
# ========================================================================================================

app = Flask(__name__)


@app.route("/",)
def index():
    return "Hello World"


@handler.add(MessageEvent, message=TextMessage)
def handle_message(event):
    user_id = event.source.user_id
    req_sentence = event.message.text.strip()
    action = req_sentence.lower().split(",")
    print(user_id, "----->", req_sentence)

    # try:
    #     user_check = session.query(Account).filter(Account.user_id == user_id).first()
    #     if (user_check is None):
    #         session.add(Account(user_id=user_id, nickname="Jack上海商銀", account_number="011447517631614"))
    #         session.commit()
    # except:
    #     print("insert default account error")

    start_all = time.time()

    # certain keywords in request sentence
    if ("說明" == req_sentence):
        response = "[查詢語句]\n" + "轉出清單: 查看轉出清單\n" + "常用清單: 查看常用轉帳帳戶清單\n\n" + \
                "[命令語句]\n" + "add,常用帳戶暱稱,常用帳戶帳號\n" + "del,常用帳戶暱稱/代號\n" + \
                "範例:add,測試帳戶,447517631614\n範例:del,測試帳戶\n\n" + \
                "[轉帳語句]\n" +  "請從台幣戶轉帳10000塊到Jack的上海商銀\n" + "從薪轉戶轉帳給我媽媽3000元"
    elif ("常用清單" in req_sentence):
        result = session.query(Account).filter(Account.user_id == user_id).order_by(Account.pk_id).all()
        response = ""
        for i, r in enumerate(result):
            response += str(i+1) + ". " + r.nickname + "(" + r.account_number + ")\n"
        response = response[:-1]
    elif ("轉出清單" in req_sentence):
        response = ""
        for i, (key, value) in enumerate(trans_from_dict.items()):
            response += str(i+1) + ". " + key + "(" + value + ")\n"
        response = response[:-1]
    elif (len(action) == 3 and action[0] == "add"):
        try:
            session.add(Account(
                user_id=user_id,
                nickname=action[1].strip(),
                account_number=action[2].strip()
            ))
            session.commit()
            response = "常用帳號新增成功！可輸入[常用清單]查看結果"
        except:
            response = "很抱歉，無法處理您的服務"
    elif (len(action) == 2 and action[0] == "del"):
        try:
            idx = int(action[1].strip())
            result = session.query(Account).filter(Account.user_id == user_id).order_by(Account.pk_id).all()
            nickname_dict = dict()
            for i, r in enumerate(result):
                nickname_dict[i+1] = r.nickname
            nickname = nickname_dict[idx]
        except:
            nickname = action[1].strip()

        try:
            session.query(Account).filter(Account.nickname == nickname).delete(synchronize_session=False)
            session.commit()
            response = nickname + "已刪除！可輸入[常用清單]查看結果"
        except:
            response = "很抱歉，無法處理您的服務"
    else:  # other sentence for transfer
        if (req_sentence == "" or len(req_sentence) > 70):
            response = "很抱歉，無法處理您的服務"
        else:
            line_record = {
                "user_id": user_id, "session_id": None,
                "req_sentence": req_sentence, "resp_status": None, "resp_sentence": None, "is_transfer": None,
                "trans_from": None, "trans_to": None, "amount": None
            }
            response = dialog(session, tc, ws, pos, ner, all_digital_bank, all_bank_list, trans_from_dict, line_record)

    print("*"*50)
    print(response)
    print("all time: " + str(time.time()-start_all) + "s\n")
    line_bot_api.reply_message(event.reply_token, TextSendMessage(text=response))


@app.route("/callback", methods=['POST'])
def callback():
    # get X-Line-Signature header value
    signature = request.headers['X-Line-Signature']

    # get request body as text
    body = request.get_data(as_text=True)
    app.logger.info("Request body: " + body)

    # handle webhook body
    try:
        handler.handle(body, signature)
    except InvalidSignatureError:
        abort(400)
    return 'OK'


if __name__ == "__main__":
    app.run(host='0.0.0.0', port=80, debug=True, use_reloader=False)

    close_connection(session)

    # Release model
    del ws
    del pos
    del ner
