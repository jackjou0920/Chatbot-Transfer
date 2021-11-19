import time
from transfer_classification import TransferClassifier
from transfer_ner import TransferNer

def main():
    text = ['給媽媽一銀轉兩千塊從我的帳戶裡']
    # text = ['要從他的薪轉戶轉帳給一千三百']

    # start = time.time()
    # tc = TransferClassifier(model_path="./classification/model_classification_gpu_epoch_1_batch_64")
    # print("load model time:", time.time()-start, "s")
    
    # start = time.time()
    # result = tc.predict(text)
    # print("text:", text)
    # print(result, tc.get_label(result))
    # print("classifier predict time:", time.time()-start, "s")

    start = time.time()
    tn = TransferNer(model_path="./ner/model_ner_gpu_epoch_2_batch_64")
    print("load model time:", time.time()-start, "s")

    start = time.time()
    result = tn.predict(text)
    print(result)
    print("ner predict time:", time.time()-start, "s")
    

if __name__ == "__main__":
    main()