from functions import cryptoData, lstmModel, preprocess
import time


start_pgm = time.time()
print('The program begins!! ')
main_df = cryptoData.load_data()
clean_df = cryptoData.modify_data(main_df)

normalize_df = preprocess.preprocess_df(clean_df)