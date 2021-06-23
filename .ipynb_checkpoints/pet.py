import pandas as pd
import numpy as np
from icecream import ic


def quiz(qnum, df):
    res = df
    if qnum == 2:
        ic(pd.__version__)
    elif qnum == 3:
        ic(pd.show_versions())
    elif qnum == 4:
        data = {'animal': ['cat', 'cat', 'snake', 'dog', 'dog', 'cat', 'snake', 'cat', 'dog', 'dog'],
                'age': [2.5, 3, 0.5, np.nan, 5, 2, 4.5, np.nan, 7, 3],
                'visits': [1, 3, 2, 3, 2, 3, 1, 1, 2, 1],
                'priority': ['yes', 'yes', 'no', 'yes', 'no', 'no', 'no', 'yes', 'no', 'no']}
        labels = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j']
        df_n = pd.DataFrame(data, labels)
        res = df_n
        ic(df_n)
    elif qnum == 5:
        ic(df.describe())
    elif qnum == 6:
        ic(df.head(3))
    elif qnum == 7:
        ic(df[['animal', 'age']])
    elif qnum == 8:
        ic(df[['animal', 'age']].loc[['c', 'd', 'h']])
    elif qnum == 9:
        ic(df.loc[df['visits'] > 3])
    elif qnum == 10:
        ic(df.loc[df['age'].isnull()])
    elif qnum == 11:
        ic(df.loc[(df['age'] < 3) & (df['animal'] == 'cat')])
    elif qnum == 12:
        ic(df.loc[(df['age'] >= 2) & (df['age'] < 4)])
    elif qnum == 13:
        df.loc['f', 'age'] = 1.5
        ic(df)
    elif qnum == 14:
        ic(df['visits'].sum())
    elif qnum == 15:
        ic(df['age'].groupby(df['animal']).mean())
    elif qnum == 16: # 16 + 16-1
        df.loc['k'] = ['dog', 5.5, 2, 'no']
        ic(df)
        df.drop(['k'], inplace=True)
        ic(df)
    elif qnum == 17:
        ic(len(df['animal'].unique()))
    elif qnum == 18:
        ic(df.sort_values('age', ascending=False).sort_values('visits'))
    elif qnum == 19:
        df['priority'] = df['priority'].map({'yes': True, 'no': False})
        ic(df)
    elif qnum == 20:
        df['animal'] = df['animal'].replace({'snake': 'python'})
        ic(df)
    elif qnum == 21:
        ic(df.pivot_table(index='animal', columns='visits', values='age', aggfunc='mean'))
    elif qnum == 22:
        df_n = pd.DataFrame({'A': [1, 2, 2, 3, 4, 5, 5, 5, 6, 7, 7]})
        ic(df_n)
        # ic(df_n['A'].unique())
        df_n = df_n.drop_duplicates()
        ic(df_n)
    elif qnum == 23:
        df_n = pd.DataFrame(np.random.random(size=(5, 3)))
        ic(df_n)

        '''
        def sub(val, subt):
            return val - subt
        for i in df_n.index.values:
            avg = df_n.loc[i].mean()
            df_n.loc[i] = df_n.loc[i].map(lambda a: sub(a, avg))
        '''
        df_n = df_n.sub(df_n.mean(axis=1), axis=0)
        ic(df_n)

    elif qnum == 24:
        df_n = pd.DataFrame(np.random.random(size=(5, 10)), columns=list('abcdefghij'))
        ic(df_n)
        '''
        smallest = len(df_n.index.values)
        idx = 0
        for i in df_n.columns.values:
            if df_n[i].sum() > smallest:
                continue
            else:
                smallest = df_n[i].sum()
                idx = i
        ic(df_n[idx])
        '''
        df_n = df_n.sum().idxmin()
        ic(df_n)
    elif qnum == 25:
        df_n = pd.DataFrame(np.random.randint(0, 2, size=(10, 3)))
        ic(df_n)
    elif qnum == 26:
        nan = np.nan
        data = [[0.04, nan, nan, 0.25, nan, 0.43, 0.71, 0.51, nan, nan],
                [nan, nan, nan, 0.04, 0.76, nan, nan, 0.67, 0.76, 0.16],
                [nan, nan, 0.5, nan, 0.31, 0.4, nan, nan, 0.24, 0.01],
                [0.49, nan, nan, 0.62, 0.73, 0.26, 0.85, nan, nan, nan],
                [nan, nan, 0.41, nan, 0.05, nan, 0.61, nan, 0.48, 0.68]]
        columns = list('abcdefghij')
        df_n = pd.DataFrame(data, columns=columns)
        ic(df_n)
        cnt = [0 for i in df_n.index.values]
        answer = ['' for i in df_n.index.values]
        for i in df_n.columns.values:
            gette = df_n.loc[df_n[i].isnull()]
            for j in gette.index.values:
                cnt[j] += 1
                if cnt[j] == 3:
                    answer[j] = i
        ic(answer)
    elif qnum == 27:
        pass
    elif qnum == 28:
        df_n = pd.DataFrame({'A': [1, 2, 3], 'B': [4, 5, 6]})
        ic(df_n)
        ic([np.array(df_n[i].tolist()) for i in df_n.columns.values])
    elif qnum == 29:
        df_n = pd.DataFrame({'A': [1, 2, 3], 'B': [4, 5, 6]})
        ic(df_n)
        ic(df_n.to_dict('list'))
    elif qnum == 30:
        df_n = pd.DataFrame({"customer_id": ['kim', 'lee', 'park', 'song', 'yoon', 'kang', 'tak', 'ryu', 'jang'],
                           "product_code": ['com', 'phone', 'tv', 'com', 'phone', 'tv', 'com', 'phone', 'tv'],
                           "grade": ['A', 'A', 'A', 'A', 'A', 'A', 'B', 'B', 'B'],
                           "purchase_amount": [30, 10, 0, 40, 15, 30, 0, 0, 10]})
        ic(df_n)
        ic(df_n.pivot_table(index='customer_id', columns='product_code', values='purchase_amount', aggfunc='any'))
    elif qnum == 31:
        pass

    return res


if __name__ == '__main__':
    ic('Hello')
    dframe = ''
    while 1:
        menu = int(input('[Q-Number (2~31), 0 = Exit]'))
        if menu == 0:
            ic('ByeBye')
            break
        elif menu >= 2 & menu <= 31:
            dframe = quiz(menu, dframe)
        else:
            ic('Question Number Error -> Out of Bounds')
