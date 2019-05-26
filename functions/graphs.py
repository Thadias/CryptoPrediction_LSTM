import matplotlib.pyplot as plt
import numpy as np
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

    print('Temp_x1 = ', temp_x1, '\n', 'Temp_y1 = ', temp_y1)
    print('Temp_x2 = ', temp_x2, '\n', 'Temp_y2 = ', temp_y2)

    # plt.xlim(temp_x1[0], temp_x1[-1])
    plt.grid(True)
    # plt.xticks([datetime.date(2018, i + 1, 1) for i in range(12)],
    #            [datetime.date(2018, j + 1, 1).strftime('%b %d %Y') for j in range(12)])
    plt.plot(temp_x1, temp_y1, label='Actual')
    plt.plot(temp_x2, temp_y2, label='Predicted')
    plt.ylabel('Bitcoin Price ($)', fontsize=12)
    plt.xlabel('Time (MM/YY)', fontsize=12)
    plt.legend(bbox_to_anchor=(0.1, 1), loc=2, borderaxespad=0., prop={'size': 12})
    plt.show()

    # ----------------------

    # temp1 = temp_df.values[window_size:].astype(datetime.date)
    # temp21 = ((np.transpose(model.predict(test_input)) + 1) *
    #          temp_df['Close'].values[:-window_size])[0]
    # temp22 =
    #
    # # fig, ax1 = plt.subplots(1, 1)
    # # ax1 = plt.plot()
    # # ax1.set_xticks([datetime.date(2018, j + 1, 1) for j in range(12)])
    # # ax1.set_xticklabels([datetime.date(2018, j + 1, 1).strftime('%b %d %Y')
    # #                     for j in range(12)])
    # plt.xticks([datetime.date(2018, j + 1, 1) for j in range(12)])
    # #plt.xticklabels([datetime.date(2018, j + 1, 1).strftime('%b %d %Y')
    # #                     for j in range(12)])
    # plt.plot(temp1, temp_df['Close'][window_size:], label='Actual')
    # # plt.show()
    # # #
    # # ax1.plot(temp1, temp2, label='Predicted')
    # #
    # temp_01 = np.mean(np.abs((np.transpose(model.predict(test_input)) + 1)
    #                        - (temp_df['Close'].values[window_size:])
    #                        / (temp_df['Close'].values[:-window_size])))
    # # plt.annotate(f'MAE: {temp_01}', xy=(0.75, 0.9), xycoords='axes fraction',
    # #              xytext=(0.75, 0.9), textcoords='axes fraction')
    # #
    # plt.ylabel('Bitcoin Price ($)', fontsize=12)
    # plt.xlabel('Time (MM/YY)', fontsize=12)
    # # plt.set_ylabel('Bitcoin Price ($)', fontsize=12)
    # # plt.set_xlabel('Time (MM/YY)', fontsize=12)
    # plt.legend(bbox_to_anchor=(0.1, 1), loc=2, borderaxespad=0., prop={'size': 12})
    #
    # print('Temp1 = ', temp1, '\n', 'Temp2 = ', temp2)
    # # print('Temp1 dtypes: ', temp1.dtypes(), '\n', 'Temp2 dtypes: ', temp2.dtypes())
    # plt.show()
