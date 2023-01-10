import pandas as pd
import os

def ClassifyCodes(stocks, N=50, option=1):
    topN = []

    for stock in stocks :
        
        # data path
        path_data_m = os.path.join('./DATA\\min', stock+'_concat.csv')
        path_data_d = os.path.join('./DATA\\day(3month)', stock+'DAY_3month.csv')

        # read data
        vol_m = pd.read_csv(path_data_m)
        vol_d = pd.read_csv(path_data_d)

        # Classificatin 1 : ratio = current/average
        if option == 1:
            vol_now = vol_m.iloc[0,10]
            vol_d_30 = vol_d.iloc[:30,10]

            vol_d_30 = vol_d_30.loc[::-1].reset_index(drop=True)

            # vol_now versus Mean_vol of 30 days
            vol_mean_30 = vol_d_30.mean()
            rating =vol_now/vol_mean_30

        # Classification 2 : rating = max-standard
        elif option == 2:
            vol_m_a = vol_m[::-1].reset_index(drop=True)
            one_day = vol_m_a.iloc[-2730:].VOL.values
            
            # 기준 설명 #
            # standard line을 넘는 volumn이 최근 7일간에 존재하면 list에 추가
            # standard와 최근7일 변동량의 max간 차이로 리스트 내 순위 결정
            # TOP 100까지 볼 예정
            standard_line = 0.7*(vol_m['VOL'].values.max())
            rating = one_day.max()-standard_line
    

        # List top 100 hot stock codes
        if len(topN)<N:
            topN.insert(-1,[stock, rating])
        elif len(topN)==N and topN[-1][1] < rating:
            topN[-1] = [stock, rating]
        for i in range(1, len(topN)):
            if topN[-i-1][1] < rating:
                topN[-i-1], topN[-i] = topN[-i], topN[-i-1]
                    
            

    clf_stock = []
    for j in range(N) :
        clf_stock.append(topN[j][0])
    print('<Search Range>\n' , clf_stock)
    return clf_stock