import matplotlib.pyplot as plt
import datetime


def bitcoin_price_graph(df):
    # Printing the graph of bitcoin price in the input file.
    ax1 = plt.axes()
    ax1.set_ylabel('Closing Price ($)', fontsize=12)
    ax1.set_xlabel('Time (MM/YY)', fontsize=12)
    # ax1.set_xticks([datetime.time(i, j, 1).strftime('%b %d %Y') for i in range(2018, 2019)
    #                 for j in [1, 12]])
    ax1.plot(df.index, df['Close'])
    plt.show()


# def bitcoin_pred_graph(main_df, test_input, model, split_date, window_size):
def bitcoin_pred_graph(main_df, pred_prices, split_date, window_size, pred_range):
    temp_df = main_df[main_df.index >= split_date]
    temp_x1 = temp_df.index[window_size + 3:].astype(datetime.date)
    temp_x2 = temp_df.index[window_size + 3:].astype(datetime.date)  # [pred_range: pred_range + pred_range]
    temp_y1 = temp_df['Close'][window_size + 3:]
    temp_y2 = pred_prices['Pred']

    # print('Temp_x1 = ', temp_x1, '\n', 'Temp_y1 = ', temp_y1)
    # print('Temp_x2 = ', temp_x2, '\n', 'Temp_y2 = ', temp_y2)

    plt.grid(True)
    plt.plot(temp_x1, temp_y1, label='Actual')
    plt.plot(temp_x2, temp_y2, label='Predicted')
    plt.ylabel('Bitcoin Price ($)', fontsize=12)
    plt.xlabel('Time (MM/YY)', fontsize=12)
    plt.legend(bbox_to_anchor=(0.1, 1), loc=2, borderaxespad=0., prop={'size': 12})
    plt.show()

