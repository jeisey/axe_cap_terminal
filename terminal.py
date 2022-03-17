import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.graph_objects as go
import datetime
from datetime import timedelta
from plotly.subplots import make_subplots
import plotly.figure_factory as ff
import pandas_ta as ta
import time
import calendar
import tweepy
import webbrowser
import numba as nb

##>>PAGE LAYOUT <<##
st.set_page_config(page_title="Axe Cap Terminal", page_icon="💡",layout="wide")

##>> FUNCTIONS <<##

#@st.cache(allow_output_mutation=True)
@st.experimental_memo(ttl=300, max_entries=10)
def load_indicator_strategy():
    TotalStrat = ta.Strategy(
        name="totalstrat",
        ta=[
            # >>> OVERLAP INDICATORS <<< #
            
            #SMA
            {"kind": "sma","close": "Adj Close", "length": 5},
            {"kind": "sma","close": "Adj Close", "length": 10},
            {"kind": "sma","close": "Adj Close", "length": 15},
            {"kind": "sma","close": "Adj Close", "length": 20},
            {"kind": "sma","close": "Adj Close", "length": 50},
            {"kind": "sma","close": "Adj Close", "length": 100},
            {"kind": "sma","close": "Adj Close", "length": 200},
            
            #EMA
            {"kind": "ema", "close": "Adj Close", "length": 5},
            {"kind": "ema", "close": "Adj Close", "length": 10},
            {"kind": "ema", "close": "Adj Close", "length": 15},
            {"kind": "ema", "close": "Adj Close", "length": 20},
            {"kind": "ema", "close": "Adj Close", "length": 50},
            {"kind": "ema", "close": "Adj Close", "length": 100},
            {"kind": "ema", "close": "Adj Close", "length": 200},
            
            #HMA
            {"kind": "hma", "close": "hl2", "length": 21},

            
            #VWAP
            {"kind": "vwap", "high":"high", "low":"low", "close": "Adj Close", "volume":"Volume"},
            
            # >>> MOMENTUM INDICATORS <<< #
            
            #Momentum
            {"kind": "mom", "close": "Adj Close", "length":1},
            {"kind": "mom", "close": "Adj Close", "length":5},
            
            #RSI
            {"kind": "rsi", "close": "Adj Close"},
            
            #Stoch_RSI
            {"kind": "stochrsi", "close":"Adj Close"},
            
            #QQE_Slow
            {"kind": "qqe", "close": "Adj Close", "length":20, 
            "col_names": ("s_QQE", "s_RSI_MA", "s_QQE_L", "s_QQE_S")},
            
            #QQE_Fast
            {"kind": "qqe", "close": "Adj Close", "length":6, "smooth":3, "factor":2.621,
            "col_names": ("f_QQE", "f_RSI_MA", "f_QQE_L", "f_QQE_S")},
        
            # >>> TREND INDICATORS <<< #
            
            #ADX
            {"kind": "adx", "high":"high", "low":"low", "close": "Adj Close"},
            
            #MACD
            {"kind": "macd", "close": "Adj Close"},
            
            #TTM Trend
            {"kind": "ttm_trend", "high":"high", "low":"low", "close": "Adj Close"},
                    
            # >>> VOLATILITY INDICATORS <<< #
            
            #ATR
            {"kind": "atr", "high":"high", "low":"low", "close": "Adj Close"},
            
            # >>> VOLUME INDICATORS <<< #
            
            #AD
            {"kind": "ad", "high":"high", "low":"low", "close": "Adj Close", "volume":"Volume","open":"open"},
            
            #CMF
            {"kind": "cmf", "high":"high", "low":"low", "close": "Adj Close", "volume":"Volume","open":"open"},
            
            #OBV
            {"kind": "obv", "close": "Adj Close", "volume":"Volume"}
        ]
    )
    return TotalStrat

#@st.cache(allow_output_mutation=True)
@st.experimental_memo(ttl=300, max_entries=10)
def APMDaily(data):
    """Requires the open, high, and low values to exist in the provided dataframe >>>'data'. 
    For plotting, fill functions can be used inbetween the adrlow10+adrlow5 and the adrhigh10+adrhigh5.
    
    Parameters
    ----------
    data : dataframe
        A dataframe with the requirement being open, high, and low values.
        
    Returns
    -------
    DataFrame
        Appends the ADRHIGH10, ADRLOW10, ADRHIGH5, and ADRLOW5 to the dataframe"""

    lb=11 #if need to make dynamic can make this a variable and turn the r1/adr calcs into a loop based on size of "LB"
    data['dayrange'] = data['high'] - data['low']
    data['r1']=data['dayrange'].tail(lb)[1]
    data['r2']=data['dayrange'].tail(lb)[2]
    data['r3']=data['dayrange'].tail(lb)[3]
    data['r4']=data['dayrange'].tail(lb)[4]
    data['r5']=data['dayrange'].tail(lb)[5]
    data['r6']=data['dayrange'].tail(lb)[6]
    data['r7']=data['dayrange'].tail(lb)[7]
    data['r8']=data['dayrange'].tail(lb)[8]
    data['r9']=data['dayrange'].tail(lb)[9]
    data['r10']=data['dayrange'].tail(lb)[10]
    data['adr_5'] = (data['r1'] + data['r2'] + data['r3'] + data['r4'] + data['r5']) / 5
    data['adr_10'] = (data['r1'] + data['r2'] + data['r3'] + data['r4'] + data['r5'] + data['r6'] + data['r7'] + data['r8'] + data['r9'] + data['r10']) / 10
    data['ADRHIGH10'] = data['open']+(data['adr_10']/2)
    data['ADRHIGH5'] = data['open']+(data['adr_5']/2)
    data['ADRLOW10'] = data['open']-(data['adr_10']/2)
    data['ADRLOW5'] = data['open']-(data['adr_5']/2)
    data.drop(columns=['r1', 'r2','r3','r4','r5','r6','r7','r8','r9','r10','adr_5','adr_10'],inplace=True)
        
    return data

#@st.cache(allow_output_mutation=True)
#@st.experimental_memo(ttl=300)
def APMMonthly(data):
    """Requires the open, high, and low values to exist in the provided dataframe >>>'data'. 
    For plotting, fill functions can be used inbetween the adrlow10+adrlow5 and the adrhigh10+adrhigh5.
    
    Parameters
    ----------
    data : dataframe
        A dataframe with the requirement being open, high, and low values.
        
    Returns
    -------
    DataFrame
        Appends the ADRHIGH10, ADRLOW10, ADRHIGH5, and ADRLOW5 to the dataframe"""
    data['dayrange'] = data['m_high'] - data['m_low']
    data['r1']=data['dayrange'].tail(20)[0]
    data['r2']=data['dayrange'].tail(40)[0]
    data['r3']=data['dayrange'].tail(60)[0]
    data['r4']=data['dayrange'].tail(80)[0]
    data['r5']=data['dayrange'].tail(100)[0]
    data['r6']=data['dayrange'].tail(120)[0]
    data['r7']=data['dayrange'].tail(140)[0]
    data['r8']=data['dayrange'].tail(160)[0]
    data['r9']=data['dayrange'].tail(180)[0]
    data['r10']=data['dayrange'].tail(200)[0]
    data['adr_5'] = (data['r1'] + data['r2'] + data['r3'] + data['r4'] + data['r5']) / 5
    data['adr_10'] = (data['r1'] + data['r2'] + data['r3'] + data['r4'] + data['r5'] + data['r6'] + data['r7'] + data['r8'] + data['r9'] + data['r10']) / 10
    data['m_ADRHIGH10'] = data['m_open']+(data['adr_10']/2)
    data['m_ADRHIGH5']  = data['m_open']+(data['adr_5']/2)
    data['m_ADRLOW10']  = data['m_open']-(data['adr_10']/2)
    data['m_ADRLOW5']   = data['m_open']-(data['adr_5']/2)
    data.drop(columns=['r1','r2','r3','r4','r5','r6','r7','r8','r9','r10','adr_5','adr_10','dayrange'],inplace=True)
        
    return data

#@st.cache
#@st.experimental_memo(ttl=300)
def RDS(data,w1=0.6,w3=0.3,w6=0.1,a=30,b=90,c=180):
    """Appends four columns of RDS - Relative Distance Strength - to the provided dataframe. "Adj Close" is a required named variable in your provided dataframe.
    
    Parameters
    ----------
    data : dataframe
        A dataframe with the only requirement being a named column of "Adj Close"
    w1,w2,w3 : float, optional
        The custom weightings for RDS. Default is set to 60%/30%/10% for weighting of 1m,3m and 6months
    a,b,c : float, optional
        The custom time recency periods for RDS. Default is set to a=1month, b=3months, c=6months.
        
    Returns
    -------
    DataFrame
        Appends the Weighted, 1, 3, and 6 months RDS to the provided DataFrame."""
    x=0
    y=1
    size = len(data)
    df = pd.DataFrame()
    while x<size:
        lp      = data["Adj Close"].iloc[size-y]
        m1_low  = data["Adj Close"].iloc[0 if (size-x-a)<0 else (size-x-a):size-x].min()
        m3_low  = data["Adj Close"].iloc[0 if (size-x-b)<0 else (size-x-b):size-x].min()
        m6_low  = data["Adj Close"].iloc[0 if (size-x-c)<0 else (size-x-c):size-x].min()
        m1_high = data["Adj Close"].iloc[0 if (size-x-a)<0 else (size-x-a):size-x].max()
        m3_high = data["Adj Close"].iloc[0 if (size-x-b)<0 else (size-x-b):size-x].max()
        m6_high = data["Adj Close"].iloc[0 if (size-x-c)<0 else (size-x-c):size-x].max()
        d_from_m1_low = (lp/m1_low)-1
        d_from_m3_low = (lp/m3_low)-1
        d_from_m6_low = (lp/m6_low)-1
        d_to_m1_high  = (m1_high/lp)-1
        d_to_m3_high  = (m3_high/lp)-1
        d_to_m6_high  = (m6_high/lp)-1
        RDS1m = round((d_from_m1_low * d_to_m1_high),4)*100
        RDS3m = round((d_from_m3_low * d_to_m3_high),4)*100
        RDS6m = round((d_from_m6_low * d_to_m6_high),4)*100
        RDSWeighted = round(((RDS1m*w1)+(RDS3m*w3)+(RDS6m*w6)),2)
        RDS = {'Index':data.index[size-y],'RDS1m':RDS1m, 'RDS3m':RDS3m, 'RDS6m':RDS6m, 'RDSWeighted':RDSWeighted}
        df = df.append(RDS, ignore_index=True)
        x+=1
        y+=1
    df   = df.reindex(index=df.index[::-1])
    df   = df.set_index('Index')
    data = data.join(df) 
    return data

#@st.cache
#@st.experimental_memo(ttl=300)
def PivotPoints(data):  
    PP = pd.Series((data['high'] + data['low'] + data['Adj Close']) / 3)  
    R1 = pd.Series(2 * PP - data['low'])  
    S1 = pd.Series(2 * PP - data['high'])  
    R2 = pd.Series(PP + data['high'] - data['low'])  
    S2 = pd.Series(PP - data['high'] + data['low'])  
    R3 = pd.Series(data['high'] + 2 * (PP - data['low']))  
    S3 = pd.Series(data['low'] - 2 * (data['high'] - PP))  
    pivots = {'PP':PP, 'R1':R1, 'S1':S1, 'R2':R2, 'S2':S2, 'R3':R3, 'S3':S3}  
    pdf = pd.DataFrame(pivots)  
    pdf=pdf.shift(periods=1, axis="index")
    data= data.join(pdf)  
    return data

def SMA(x, length):
    # simple moving average
    return x.rolling(length).mean()

def WildMA(x, length):
    # Wilder's moving average
    a = 1/length
    return x.ewm(alpha=a, min_periods=length).mean()

# def TrendUp_pd(n, Up, AdjC):
#     TrendUp = np.empty(n)
#     TrendUp[0] = Up.head(1)
#     for i in range(1,n):
#         if AdjC.iloc[i-1] > TrendUp[i-1]:
#             TrendUp[i] = (max(Up.iloc[i], TrendUp[i-1]))
#         else:
#             TrendUp[i] = Up.iloc[i]
#     return TrendUp

# def TrendDn_pd(n, Dn, AdjC):
#     TrendDn = np.empty(n)
#     TrendDn[0] = Dn.head(1)
#     for i in range(1,n):
#         if AdjC.iloc[i-1] < TrendDn[i-1]:
#             TrendDn[i] = (min(Dn.iloc[i], TrendDn[i-1]))
#         else:
#             TrendDn[i] = Dn.iloc[i]
#     return TrendDn

@nb.njit()
def TrendUp_np(Up, AdjC):
    n = len(Up)
    TrendUp = np.empty(n)
    TrendUp[0] = Up[0]
    for i in range(1,n):
        if AdjC[i-1] > TrendUp[i-1]:
            TrendUp[i] = (max(Up[i], TrendUp[i-1]))
        else:
            TrendUp[i] = Up[i]
    return TrendUp

@nb.njit()
def TrendDn_np(Dn, AdjC):
    n = len(Dn)
    TrendDn = np.empty(n)
    TrendDn[0] = Dn[0]
    for i in range(1,n):
        if AdjC[i-1] < TrendDn[i-1]:
            TrendDn[i] = (min(Dn[i], TrendDn[i-1]))
        else:
            TrendDn[i] = Dn[i]
    return TrendDn

@nb.njit()
def extremum_np(Trend, high, low):
    n = len(Trend)
    ex = np.empty(n)
    ex[0] = 0
    for i in range(1,n):
        if (Trend[i] > 0) and (Trend[i-1] < 0):
            ex[i] = high[i]
        elif (Trend[i] < 0) and (Trend[i-1] > 0):
            ex[i] = low[i]
        elif Trend[i] == True:
            ex[i] = max(ex[i-1], high[i])
        elif Trend[i] == False:
            ex[i] = min(ex[i-1], low[i])
        else:
            ex[i] = ex[i-1]
    return ex

#@st.cache(allow_output_mutation=True)
#@st.experimental_memo(ttl=300)
def SwingArms(data, ATRPeriod=28, ATRFactor=5):
    # SwingArms Technical Indicator TrendUp and TrendDn controls Sup/Res, Trend controls Direction and Trail overall SwingArms function

    # data['HL'] = data['high'] - data['low']
    # data['HLSMA'] = SMA(data['HL'], ATRPeriod) * 1.5
    # data['HiLo'] = data[['HL', 'HLSMA']].min(axis=1)

    # same but numpy's fmin is the fastest way to compare 2 dataframe columns min
    data['HiLo'] = np.fmin(*(data['high'] - data['low']).align(1.5 * SMA(data['high'] - data['low'], ATRPeriod)))

    data['HRef'] = np.where(data['low'] <= data['high'].shift(1),
                        data['high'] - data['Adj Close'].shift(1),
                        data['high'] - data['Adj Close'].shift(1) - 0.5 * (data['low'] - data['high'].shift(1)))

    data['LRef'] = np.where(data['high'] != data['low'].shift(1),
                        data['Adj Close'].shift(1) - data['low'],
                        data['Adj Close'].shift(1) - data['low'] - 0.5 * (data['low'].shift(1) - data['high']))
    
    # data['trueRange'] = data[['HiLo','HRef','LRef']].max(axis=1)

    # same max but faster implementation in numpy with nan handling
    data['trueRange'] = np.maximum.reduce(np.nan_to_num([data['HiLo'],data['HRef'],data['LRef']]))
    
    # this WildMA depends on what value it started from because it's a recursive function
    # different starting values eg. data since IPO will perform slightly differently vs data since 1yr, 2yr 
    data['loss'] = ATRFactor * WildMA(data['trueRange'], ATRPeriod)
    
    data['Up'] = data['Adj Close'] - data['loss']
    data['Dn'] = data['Adj Close'] + data['loss']
    
    # TrendUp and TrendDn Calculation
    # get length of df to use in later size calc, can also use length of df columns but need to make sure NaNs are counted for
    n = len(data)
    # using pandas series - slower
    # data['TrendUp'] = TrendUp_pd(n, data['Up'], data['Adj Close'])
    # data['TrendDn'] = TrendDn_pd(n, data['Dn'], data['Adj Close'])

    # let's write it using numpy arrays - faster, added numba loop opt
    data['TrendUp'] = TrendUp_np(data['Up'].values, data['Adj Close'].values)
    data['TrendDn'] = TrendDn_np(data['Dn'].values, data['Adj Close'].values)
    
    # If close > TrendDn (Close above downtrend), the Trend is True, and vice versa, the SwingArms value is returned by Trail (the thicker line in the SwingArms)
    # Works by assigning T or F where the condition is met accordingly, or NaN otherwise. ffill then just forward fills the NaNs with the last non-null value

    data['Trend'] = np.select([data['Adj Close'] > data['TrendDn'].shift(1), data['Adj Close'] < data['TrendUp'].shift(1)], 
                                            [True, False], default=np.nan)

    data['Trend'].fillna(method='ffill', inplace=True)

    data['Trail'] = np.where(data['Trend'] == True,
                        data['TrendUp'],
                        data['TrendDn'])

    data['ex'] = extremum_np(data['Trend'].values, data['high'].values, data['low'].values)

    fib1Level = 61.8
    fib2Level = 78.6
    fib3Level = 88.6

    data['f1'] = data['ex'] + (data['Trail'] - data['ex']) * fib1Level / 100
    data['f2'] = data['ex'] + (data['Trail'] - data['ex']) * fib2Level / 100
    data['f3'] = data['ex'] + (data['Trail'] - data['ex']) * fib3Level / 100

    data.drop(columns=['HiLo','HRef','LRef','trueRange','loss','Up','Dn'],inplace=True)

    return data

@nb.njit()
def MashumeHull_np(HMA, lookback = 2):
    
    size = len(HMA)
    concavity = np.empty(size)
    concavity[0] = 0
    HMA_col = np.empty(size)
    HMA_col[0] = 0

    for i in range(1, size):
        # concavity = HMA > (HMA[prev] + delta/lookback)
        # concavity = HMA > nextbar, then assign color codes to HMA_col
        concavity[i] = HMA[i] > (HMA[i-1] + (HMA[i-1] - HMA[i-1-lookback])/lookback)
        if concavity[i]: 
            if HMA[i] < HMA[i-1]: 
                HMA_col[i] = 3
            else: 
                HMA_col[i] = 4
        else:
            if HMA[i] > HMA[i-1]: 
                HMA_col[i] = 1
            else: 
                HMA_col[i] = 2

    return concavity, HMA_col

#@st.cache(allow_output_mutation=True)
#@st.experimental_memo(ttl=300)
def MashumeHull(data, lookback=2):
    HMA = data['HMA_21'].values
    data['concavity'], data['HMA_col'] = MashumeHull_np(HMA, lookback)

    data['HMA_col'].fillna(method='ffill', inplace=True)

    return data

#@st.cache(allow_output_mutation=True)
@st.experimental_memo(ttl=300, max_entries=10)
def options_chain(_tk):
    exps = tk.options
    # Get options for each expiration
    options = pd.DataFrame()
    for e in exps:
        opt = tk.option_chain(e)
        opt = pd.DataFrame().append(opt.calls).append(opt.puts)
        opt['expirationDate'] = e
        options = options.append(opt, ignore_index=True)
    options['expirationDate'] = pd.to_datetime(options['expirationDate'])
    options['dte'] = (options['expirationDate'] - datetime.datetime.today()).dt.days / 365

    #Calculate minimum expiration date
    min_exp_date = options.expirationDate.min()
    #Start of Week
    start = min_exp_date - timedelta(days=min_exp_date.weekday())
    options['start'] = start
    #End of Week
    end = start + timedelta(days=6)
    options['end']=end
    #Boolean if option is a weekly expiry
    options['Weekly'] = options['expirationDate'] <= options['end']

    # Boolean column if the option is a CALL
    options.loc[options['contractSymbol'].str[4:].apply(lambda x: "C" in x), 'direction'] = 'Call'  
    options.loc[options['contractSymbol'].str[4:].apply(lambda x: "P" in x), 'direction'] = 'Put'  
    
    options[['bid', 'ask', 'strike','volume','openInterest']] = options[['bid', 'ask', 'strike','volume','openInterest']].apply(pd.to_numeric)
    options.volume = options.volume.round(0)
    options.openInterest = options.openInterest.round(0)
    options['mark'] = (options['bid'] + options['ask']) / 2 # Calculate the midpoint of the bid-ask
    options['vol_prems'] = round(options['mark']*options['volume'],4)
    options['oi_prems']  = round(options['mark']*options['openInterest'],4)
    options['bool_VolSpike'] = options['volume']>options['openInterest']
    options['vol_vs_oi_weighted'] = ((options['volume']*options['openInterest'])-1)*options['volume']
    options['OnlyTodaysTrades'] = options.lastTradeDate.dt.strftime("%Y-%m-%d") == datetime.date.today().strftime("%Y-%m-%d")
    
    # Drop unnecessary and meaningless columns
    options = options.drop(columns = ['contractSize', 'currency'])

    return options

# ----------------------------------- Load Stock Data -----------------------------------
#@st.cache(allow_output_mutation=True)
@st.experimental_memo(ttl=300, max_entries=10)
def load_data(symbol):
    try:
        symb = symbol.upper()
        ticker_info = yf.Ticker(symb)
        stock_data = yf.download(tickers=symb,period=tp, interval=intv)
        stock_data = stock_data.rename(columns={"Close": "close", "High": "high","Low":"low","Open":"open"})
        stock_data['hl2']=(stock_data['high']+stock_data['low'])/2
        stock_data['month'] = stock_data.index.to_numpy().astype('datetime64[M]')
        stock_data['hl_range'] = round(stock_data['high']-stock_data['low'],4)
        stock_data['oc_range'] = round(stock_data['close']-stock_data['open'],4) 
        stock_data['gap_range'] = round(stock_data['open']-(stock_data.shift(periods=1).close),4)
        stock_data['bullbear'] = stock_data['close'] >= (stock_data.shift(periods=1).close)
        stock_data['me_range'] =  round(stock_data['high']-(stock_data.shift(periods=1).close),4) #max extension
        msd = yf.download(tickers=symb,period=tp, interval='1mo')
        msd = msd.reset_index()
        msd = msd.rename(columns={"Close": "m_close", "High": "m_high","Low":"m_low","Open":"m_open","Adj Close":"m_Adj Close","Volume":"m_volume","Date":"month"})
        stock_data = stock_data.reset_index().merge(msd,on='month' ,how="left").set_index('Date')

    except:
        "Not a valid ticker, please try again"
        stock_data = "Not valid"
        ticker_info = "Not valid"
    return stock_data,ticker_info

#@st.cache()
#@st.experimental_memo(ttl=300)
def load_range_dist(df,freq):
    df=df.tail(freq)
    fig = make_subplots(rows=2, cols=2, subplot_titles=("Open-Close Range", "High-Low Range", "Gap Up/Down Range", "Max Extension"))
    oc = go.Box(
        x=df.oc_range,
        name='Open-Close Range', # name used in legend and hover labels
        marker_color='blue',
        opacity=0.5,
        #xbins=dict(size=1)
        )
    hl = go.Box(
        x=df.hl_range,
        name='High-Low Range', # name used in legend and hover labels
        marker_color='red',
        opacity=0.5,
        #xbins=dict(size=1)
        )
    gr = go.Box(
        x=df.gap_range,
        name='Gap Range', # name used in legend and hover labels
        marker_color='purple',
        opacity=0.5
        )
    me = go.Box(
        x=df.me_range,
        name='Max Extension', # name used in legend and hover labels
        marker_color='green',
        opacity=0.5,
        )
    fig.add_annotation(x=df.tail(1).oc_range[0],  y=0.2, text = 'Current', showarrow = True, row=1, col=1)
    fig.add_annotation(x=df.tail(1).hl_range[0],  y=0.2, text = 'Current', showarrow = True, row=1, col=2)
    fig.add_annotation(x=df.tail(1).gap_range[0], y=0.2, text = 'Current', showarrow = True, row=2, col=1)
    fig.add_annotation(x=df.tail(1).me_range[0],  y=0.2, text = 'Current', showarrow = True, row=2, col=2)
    fig.update_layout(title={'text':'$'+utick.upper()+' Range of Move Distribution based on past '+ str(range_freq)+' days' ,'x':0.5,'xanchor': 'center'})
    fig.update_layout(template='plotly_white', paper_bgcolor='rgba(0,0,0,0)',plot_bgcolor='rgba(0,0,0,0)',showlegend=False)
    fig.update_yaxes(showticklabels =False)
    fig.append_trace(oc, 1,1)
    fig.append_trace(hl, 1,2)
    fig.append_trace(gr, 2,1)
    fig.append_trace(me, 2,2)
    return fig

#@st.cache()
#@st.experimental_memo(ttl=300)
def load_range_means(df,freq=500):
    df=df.tail(freq)
    oc_range_mean = round(df.oc_range.describe().loc['mean'],2)
    hl_range_mean = round(df.hl_range.describe().loc['mean'],2)
    gap_range_mean = round(df.gap_range.describe().loc['mean'],2)
    me_range_mean = round(df.me_range.describe().loc['mean'],2)
    op = round(df.tail(1).open[0],2) #last open price of current candle
    lc = round(df.tail(1).close[0],2) #last close price of current candle
    hi = round(df.tail(1).high[0],2) #high of current candle
    lo = round(df.tail(1).high[0],2) #high of current candle
    max_avg_oc_range = op+oc_range_mean
    remaining_room = round(max_avg_oc_range - lc,4)
    ext_room = ''
    if lc >= max_avg_oc_range: 
        ext_room='Extended Past Average Range of ' + str(max_avg_oc_range)
    else:
        ext_room='Remaining Room of: '+ str(remaining_room)

    oc_msg = ''
    if oc_range_mean > 0: 
        oc_msg = utick.upper()+' is in a Bullish skewed Open-Close Cycle, indicating a general uptrend based on the '+str(range_freq)+' periods selected.'
    else: 
        oc_msg = utick.upper()+' is in a Bearish skewed Open-Close Cycle, indicating a general downtrend based on the '+str(range_freq)+' periods selected.'

    hl_msg = ''
    if df.tail(1).hl_range[0] < hl_range_mean:
        hl_msg = 'High-Low Range smaller than average, potentially more range left in current period of: '+ str(round(df.tail(1).hl_range[0],2))+' vs avg of '+str(hl_range_mean)
    else: 
        hl_msg = 'High-Low Range greater than average, potentially capped out range extension in current period of '+ str(round(df.tail(1).hl_range[0],2))+' vs avg of '+str(hl_range_mean)

    gap_msg = 'The average gap up/down is '+str(gap_range_mean)+' with a 75% confidence (1stdv) of being in the range of '+ str(round(df.gap_range.describe().loc['25%'],2))+' to '+str(round(df.gap_range.describe().loc['75%'],2))
    me_msg = ''
    if df.tail(1).me_range[0] < me_range_mean:
        me_msg = 'There is an estimated '+ str(round((me_range_mean-df.tail(1).me_range[0]),2))+' price to upside based on the average max extension of '+str(me_range_mean) #this is useful if the stock is in a longer-term bullish cycle or setting up for a bullish b/o pattern
    else:
        me_msg = 'The average max extension range has been exceeded, the average is '+str(me_range_mean)+' and for current period is '+str(round(df.tail(1).me_range[0],2))
    return op,lc,oc_range_mean,hl_range_mean,gap_range_mean,me_range_mean,ext_room,oc_msg,hl_msg,gap_msg,me_msg

# ----------------------------------- Load Main Chart & User Strategies/Indicators -----------------------------------

#@st.cache(allow_output_mutation=True)
#@st.experimental_memo(ttl=300)
def load_main_chart(df):
 # removing all empty dates
    # build complete timeline from start date to end date
    dt_all = pd.date_range(start=df.index[0],end=df.index[-1])
    # retrieve the dates that ARE in the original datset
    dt_obs = [d.strftime("%Y-%m-%d") for d in pd.to_datetime(df.index)]
    # define dates with missing values
    dt_breaks = [d for d in dt_all.strftime("%Y-%m-%d").tolist() if not d in dt_obs]
    fig = make_subplots(rows=4, cols=1, shared_xaxes=True,
                        vertical_spacing=0.01, 
                        row_heights=[0.5,0.1,0.2,0.2])
    # Plot OHLC on 1st subplot (using the codes from before)
    fig.add_trace(go.Candlestick(x=df.index,
                                open=df['open'],
                                high=df['high'],
                                low=df['low'],
                                close=df['close'],
                                name=utick.upper(),
                                showlegend=True))
    # # add hull moving average
    # if 'Xiu_Hull' in ind_selected:
    #     fig.add_trace(go.Scatter(x=df.index, y=df['HMA_21'], line={'color': 'lime'},name='HULL_up'))
    #     fig.add_trace(go.Scatter(x=df.index, y=df['HMA_21'].where(df['HMA_21'] >= df['close']), line={'color': 'red'}, name='HULL_down'))

    # # MashumeHull Concavity
    # if 'Xiu_Hull' in ind_selected:
    #     fig.add_trace(go.Scatter(x=df.index, y=df['HMA_21'], line={'color': 'lime'},name='HULL_up'))
    #     fig.add_trace(go.Scatter(x=df.index, y=df['HMA_21'].where(df['concavity']==True), line={'color': 'red'}, name='HULL_down'))

    # MashumeHull
    # forward fills the gaps between colors with previous value carried forward by 1 if curr value nan
    # plotly doesn't support conditional line coloring so line segments needed, unfortunately doesn't look as good cover, with markers to hide?
    if 'Xiu_Hull' in ind_selected:
        fig.add_trace(go.Scatter(x=df.index, y=df['HMA_21'].where(df['HMA_col']==4).ffill(limit=1), line={'color': 'lightgreen'},name='Hull_G'))
        fig.add_trace(go.Scatter(x=df.index, y=df['HMA_21'].where(df['HMA_col']==3).ffill(limit=1), line={'color': 'darkgreen'},name='Hull_DG'))
        fig.add_trace(go.Scatter(x=df.index, y=df['HMA_21'].where(df['HMA_col']==2).ffill(limit=1), line={'color': 'red'},name='Hull_R'))
        fig.add_trace(go.Scatter(x=df.index, y=df['HMA_21'].where(df['HMA_col']==1).ffill(limit=1), line={'color': 'orange'},name='Hull_O'))
        # add turning points, looked kinda ugly so put on hold for now
        # fig.add_trace(go.Scatter(x=df.index, y=df['HMA_21'].where(np.logical_and(df['concavity']==True,df['concavity']!=df['concavity'].shift(1))), marker=dict(
        #     color='white',
        #     symbol = 'triangle-up',
        #     line = dict(
        #     color = 'black',
        #     width = 1),
        #     size=5),name='Hull_Up', mode='markers'))
        # fig.add_trace(go.Scatter(x=df.index, y=df['HMA_21'].where(np.logical_and(df['concavity']==False,df['concavity']!=df['concavity'].shift(1))), marker=dict(
        #     color='white',
        #     symbol = 'triangle-down',
        #     line = dict(
        #     color = 'black',
        #     width = 1),
        #     size=5),name='Hull_Dn', mode='markers'))

    # SwingArms
    if 'Mach_SwingArms' in ind_selected:
        #Trail
        fig.add_trace(go.Scatter(x=df.index, y=df['Trail'], line={'color': 'orange'}, name='SwingArmsFlip'))

        #TrendUp
        fig.add_trace(go.Scatter(x=df.index, y=df['TrendUp'].where(df['Trend']==True), line={'color': 'green'}, name='SwingArmsG'))
        #f1 TrendUp
        fig.add_trace(go.Scatter(x=df.index, y=df['f1'].where(df['Trend']==True), line={'color': 'green'}, showlegend=False, name='Upf1', opacity=0.2))
        #f2 TrendUp
        fig.add_trace(go.Scatter(x=df.index, y=df['f2'].where(df['Trend']==True), line={'color': 'green'}, showlegend=False, name='Upf2', opacity=0.4))
        #f3 TrendUp
        fig.add_trace(go.Scatter(x=df.index, y=df['f3'].where(df['Trend']==True), line={'color': 'green'}, showlegend=False, name='Upf3', opacity=0.5))

        #TrendDn
        fig.add_trace(go.Scatter(x=df.index, y=df['TrendDn'].where(df['Trend']==False), line={'color': 'red'}, name='SwingArmsR'))
        #f1 TrendDn
        fig.add_trace(go.Scatter(x=df.index, y=df['f1'].where(df['Trend']==False), line={'color': 'red'}, showlegend=False, name='Dnf1', opacity=0.2))
        #f2 TrendDn
        fig.add_trace(go.Scatter(x=df.index, y=df['f2'].where(df['Trend']==False), line={'color': 'red'}, showlegend=False, name='Dnf2', opacity=0.4))
        #f3 TrendDn
        fig.add_trace(go.Scatter(x=df.index, y=df['f3'].where(df['Trend']==False), line={'color': 'red'}, showlegend=False, name='Dnf3', opacity=0.5))

    #ADR
    #ADRHigh10
    if 'Xiu_ADR' in ind_selected:
        fig.add_trace(go.Scatter(x=df.index,
                                y=df['m_ADRHIGH10'],
                                marker=dict(color='indigo',size=3,
                                line=dict(color='indigo', width=0.25)),
                                line_color='indigo',
                                opacity=0.5, 
                                fill=None,
                                mode='lines',
                                name='APM High'
                                ))
        #ADRHigh5
        fig.add_trace(go.Scatter(x=df.index,
                                y=df['m_ADRHIGH5'],
                                marker=dict(color='indigo',size=3,
                                line=dict(color='indigo', width=0.25)),
                                mode='lines',
                                fill='tonexty',
                                fillcolor='rgba(147,112,219,0.25)',
                                opacity=0.5,
                                name='mADRHigh5', showlegend=False
                                ))

        #ADRLow10
        fig.add_trace(go.Scatter(x=df.index,
                                y=df['m_ADRLOW10'],
                                marker=dict(color='green',size=3,
                                line=dict(color='green', width=0.25)),
                                line_color='green',
                                mode='lines',
                                fill=None,
                                opacity=0.5,
                                name='APM Low'
                                ))
        #ADRLow5
        fig.add_trace(go.Scatter(x=df.index,
                                y=df['m_ADRLOW5'],
                                marker=dict(color='green',size=3,
                                line=dict(color='green', width=0.25)),
                                fill='tonexty',
                                fillcolor='rgba(0,204,102,0.25)',
                                mode='lines',
                                opacity=0.5,
                                name='mADRLow5',showlegend=False
                                ))

    # Plot Volume trace on 2nd row 
    colors = ['green' if row['close'] - row['open'] >= 0 
            else 'red' for index, row in df.iterrows()]
    fig.add_trace(go.Bar(x=df.index, 
                        y=df['Volume'],
                        marker_color=colors,
                        showlegend=False
                        ), row=2, col=1)

    if 'Xiu_QQE' in ind_selected:
        #Plot Slow QQE trace on 5th row 
        fig.add_trace(go.Scatter(x=df.index,
                                y=df['s_QQE'],
                                line=dict(color='orange', width=2),
                                showlegend=False,
                                name="Slow_QQE"
                                ), row=3, col=1)
        fig.add_trace(go.Scatter(x=df.index,
                                y=df['s_RSI_MA'],
                                line=dict(color='green', width=2),
                                showlegend=False,
                                name="RSI_MA"
                                ), row=3, col=1)

        #Plot Fast QQE trace on 6th row 
        fig.add_trace(go.Scatter(x=df.index,
                                y=df['f_QQE'],
                                line=dict(color='orange', width=2),
                                showlegend=False,
                                name="Fast_QQE"
                                ), row=4, col=1)
        fig.add_trace(go.Scatter(x=df.index,
                                y=df['f_RSI_MA'],
                                line=dict(color='green', width=2),
                                showlegend=False,
                                name="RSI_MA"
                                ), row=4, col=1)
        fig.update_yaxes(title_text="Slow QQE", row=3, col=1)
        fig.update_yaxes(title_text="Fast QQE", row=4, col=1)
    
    if 'Bossx_MovingAverages' in ind_selected:
        fig.add_trace(go.Scatter(x=df.index,y=df['close'].rolling(window=20).mean(),marker_color='rgba(204, 204, 40, 0.5)',name='20 Day MA'))
        fig.add_trace(go.Scatter(x=df.index,y=df['close'].rolling(window=50).mean(),marker_color='rgba(0, 94, 255, 0.5)',name='50 Day MA'))
        fig.add_trace(go.Scatter(x=df.index,y=df['close'].rolling(window=100).mean(),marker_color='rgba(23, 238, 8, 0.5)',name='100 Day MA'))
        fig.add_trace(go.Scatter(x=df.index,y=df['close'].rolling(window=200).mean(),marker_color='rgba(176, 8, 238, 0.5)',name='200 Day MA'))

    if 'Bossx_GOI_GVOL_Strikes' in ind_selected:    
        fig.add_hline(y=hoi_strike_1, line_dash="dot", row=1, col=1,
                    annotation_text="Highest Collective OI Strike: "+str(hoi_strike_1), 
                    annotation_position="top left",line_color="midnightblue",annotation_font_color="midnightblue")
        fig.add_hline(y=hv_strike_1, line_dash="dot",row=1, col=1,
                    annotation_text="Highest Collective Vol Strike: "+str(hv_strike_1), 
                    annotation_position="bottom left",line_color="midnightblue",annotation_font_color="midnightblue")

    if 'Metaal_OptionLevels' in ind_selected:  
        #WEEKY PUT OI STRIKES
        if put_OI:
        #    st.write(('state:',put_OI_2))
            fig.add_hline(y=t3_PUT_Weekly_TCoiStrike_1, line_dash="dot", row=1,col=1,
                        annotation_text="WEEKLY PUT OI #1: "+str(t3_PUT_Weekly_TCoiStrike_1), 
                        annotation_position="bottom left", opacity=0.75, line_width=0.5,line_color="maroon",annotation_font_color="maroon")
            #if we want to move price to right axis for strike, this is one way to do it
            #fig.add_hline(y=t3_PUT_Weekly_TCoiStrike_1, line_dash="dot",annotation_text=str(t3_PUT_Weekly_TCoiStrike_1), annotation_position="bottom right", opacity=0.01, line_width=0.5,line_color="maroon",annotation_font_color="maroon")
        if put_OI:
            fig.add_hline(y=t3_PUT_Weekly_TCoiStrike_2, line_dash="dot", row=1,col=1,
                        annotation_text="WEEKLY PUT OI #2: "+str(t3_PUT_Weekly_TCoiStrike_2), 
                        annotation_position="bottom left", opacity=0.75, line_width=0.5,line_color="maroon",annotation_font_color="maroon")   
        if put_OI:
            fig.add_hline(y=t3_PUT_Weekly_TCoiStrike_3, line_dash="dot", row=1,col=1,
                        annotation_text="WEEKLY PUT OI #3: "+str(t3_PUT_Weekly_TCoiStrike_3), 
                        annotation_position="bottom left", opacity=0.75, line_width=0.5,line_color="maroon",annotation_font_color="maroon")
        
        if put_vol:
        #WEEKY PUT VOL STRIKES
            fig.add_hline(y=t3_PUT_Weekly_TCvolStrike_1, line_dash="dot", row=1,col=1,
                        annotation_text="WEEKLY PUT Vol #1: "+str(t3_PUT_Weekly_TCvolStrike_1), 
                        annotation_position="bottom left", opacity=0.75, line_width=0.5,line_color="maroon",annotation_font_color="maroon")  
        if put_vol:
            fig.add_hline(y=t3_PUT_Weekly_TCvolStrike_2, line_dash="dot", row=1,col=1,
                        annotation_text="WEEKLY PUT Vol #2: "+str(t3_PUT_Weekly_TCvolStrike_2), 
                        annotation_position="bottom left", opacity=0.75, line_width=0.5,line_color="maroon",annotation_font_color="maroon")              
        if put_vol:
            fig.add_hline(y=t3_PUT_Weekly_TCvolStrike_3, line_dash="dot", row=1,col=1,
                        annotation_text="WEEKLY PUT Vol #3: "+str(t3_PUT_Weekly_TCvolStrike_3), 
                        annotation_position="bottom left", opacity=0.75, line_width=0.5,line_color="maroon",annotation_font_color="maroon")
        #WEEKY CALL OI STRIKES
        if call_OI:
        #    st.write(('state:',put_OI_2))
            fig.add_hline(y=t3_CALL_Weekly_TCoiStrike_1, line_dash="dot", row=1,col=1,
                        annotation_text="WEEKLY CALL OI #1: "+str(t3_CALL_Weekly_TCoiStrike_1), 
                        annotation_position="top left", opacity=0.75, line_width=0.5,line_color="green",annotation_font_color="green")
        if call_OI:
            fig.add_hline(y=t3_CALL_Weekly_TCoiStrike_2, line_dash="dot", row=1,col=1,
                        annotation_text="WEEKLY CALL OI #2: "+str(t3_CALL_Weekly_TCoiStrike_2), 
                        annotation_position="top left", opacity=0.75, line_width=0.5,line_color="green",annotation_font_color="green")
        if call_OI:
            fig.add_hline(y=t3_CALL_Weekly_TCoiStrike_3, line_dash="dot", row=1,col=1,
                        annotation_text="WEEKLY CALL OI #3: "+str(t3_CALL_Weekly_TCoiStrike_3), 
                        annotation_position="top left", opacity=0.75, line_width=0.5,line_color="green",annotation_font_color="green")

        #WEEKY CALL Vol STRIKES
        if call_vol:
        #    st.write(('state:',put_OI_2))
            fig.add_hline(y=t3_CALL_Weekly_TCvolStrike_1, line_dash="dot", row=1,col=1,
                        annotation_text="WEEKLY CALL Vol #1: "+str(t3_CALL_Weekly_TCvolStrike_1), 
                        annotation_position="top left", opacity=0.75, line_width=0.5,line_color="green",annotation_font_color="green")
        if call_vol:
            fig.add_hline(y=t3_CALL_Weekly_TCvolStrike_2, line_dash="dot", row=1,col=1,
                        annotation_text="WEEKLY CALL Vol #2: "+str(t3_CALL_Weekly_TCvolStrike_2), 
                        annotation_position="top left", opacity=0.75, line_width=0.5,line_color="green",annotation_font_color="green")
        if call_vol:
            fig.add_hline(y=t3_CALL_Weekly_TCvolStrike_3, line_dash="dot", row=1,col=1,
                        annotation_text="WEEKLY CALL Vol #3: "+str(t3_CALL_Weekly_TCvolStrike_3), 
                        annotation_position="top left", opacity=0.75, line_width=0.5,line_color="green",annotation_font_color="green")


    # update layout by changing the plot size, hiding legends & rangeslider, and removing gaps between dates
    fig.update_layout(#showlegend=False, 
                    xaxis_rangeslider_visible=False,
                    xaxis_rangebreaks=[dict(values=dt_breaks)])
    # update y-axis label
    fig.update_yaxes(title_text="Price", row=1, col=1)
    fig.update_yaxes(title_text="Volume", row=2, col=1)


    # update title
    #fig3.update_layout(title={'text':utick+' chart, strategy: '+acm, 'x':0.5})

    # update legend
    fig.update_layout(legend=dict(
        orientation="h",
        yanchor="top",
        y=1.05,
        xanchor="left",
        x=0.01
    ))

    # removing white spaces
    fig.update_layout(margin=go.layout.Margin(
            l=20, #left margin
            r=20, #right margin
            b=20, #bottom margin
            t=20, #bottom margin
        ))
    return fig

#@st.cache
#@st.experimental_memo(ttl=300)
def load_joegopgo():
    joe_approved=False
    lp = round(df.tail(1).close[0],2)
    lvc = df.tail(1).Volume[0] #last volume current
    lvp = df.tail(2).Volume[0] #last volume prior
    joe_sma20 =  round(df.tail(1).SMA_20[0],2)
    joe_sma50 =  round(df.tail(1).SMA_50[0],2)
    joe_sma5 =   round(df.tail(1).SMA_5[0],2)
    joe_sma200 = round(df.tail(1).SMA_200[0],2)
    joe_20dhigh = round(df.tail(20).close.max(),2)
    joe_20dhigh_5perc_limit = round(joe_20dhigh-(joe_20dhigh*0.05),2)
    joe_20dhigh_perc_check = round(((lp/joe_20dhigh)-1),3)
    joe_20dhigh_5perc_bool = lp>joe_20dhigh_5perc_limit

    if (lp>joe_sma20)\
        and (joe_20dhigh_5perc_bool)\
        and (lp>5)\
        and (joe_sma50<joe_sma20)\
        and (joe_sma200<joe_sma5)\
        and (lvc>1000000)\
        and (lvc>lvp):
        joe_approved=True
    else:
        joe_approved=False
    
    return joe_approved,lp,lvc, lvp, joe_sma5, joe_sma20, joe_sma50, joe_sma200

# ----------------------------------- Load Options Data -----------------------------------

#@st.cache(allow_output_mutation=True)
#@st.experimental_memo(ttl=300)
def load_voi(odf):
    #max oi strike
    voi = odf[['expirationDate','direction', 'strike','volume','openInterest','lastTradeDate','Weekly','OnlyTodaysTrades','bool_VolSpike']].sort_values(by='openInterest', ascending=False)
    voi_new = voi.loc[voi['strike'] > oi_min]
    voi_new.reset_index(drop=True, inplace=True)
    hoi_strike = int(voi_new.head(1).strike[0])
    # max volume strike
    maxvol = odf['volume'].max()
    maxvolstrike = odf.loc[odf['volume'] == maxvol]
    maxvolstrike.reset_index(drop=True, inplace=True)
    mvs = int(maxvolstrike.head(1).strike[0])
    hv_strike_1 = int(voi.groupby(['strike']).sum().reset_index()[['strike','volume','openInterest']].sort_values(by='volume', ascending=False).strike.iloc[0])
    hv_strike_2 = int(voi.groupby(['strike']).sum().reset_index()[['strike','volume','openInterest']].sort_values(by='volume', ascending=False).strike.iloc[1])
    hv_strike_3 = int(voi.groupby(['strike']).sum().reset_index()[['strike','volume','openInterest']].sort_values(by='volume', ascending=False).strike.iloc[2])
    hoi_strike_1 = int(voi.groupby(['strike']).sum().reset_index()[['strike','volume','openInterest']].sort_values(by='openInterest', ascending=False).strike.iloc[0])
    hoi_strike_2 = int(voi.groupby(['strike']).sum().reset_index()[['strike','volume','openInterest']].sort_values(by='openInterest', ascending=False).strike.iloc[1])
    hoi_strike_3 = int(voi.groupby(['strike']).sum().reset_index()[['strike','volume','openInterest']].sort_values(by='openInterest', ascending=False).strike.iloc[2])
    return voi, hoi_strike, mvs, hv_strike_1,hv_strike_2,hv_strike_3,hoi_strike_1,hoi_strike_2,hoi_strike_3
    
#@st.cache(allow_output_mutation=True)
#@st.experimental_memo(ttl=300)
def load_strikes(odf):
    #####TOTAL VOLUME RATIOS
    #total volume
    total_call_vol=odf.groupby(by="direction").sum().sort_values(by='direction',ascending=True)['volume'][0]
    total_put_vol=odf.groupby(by="direction").sum().sort_values(by='direction',ascending=True)['volume'][1]
    total_call_vol_perc = round(total_call_vol/(total_call_vol+total_put_vol),2)
    total_put_vol_perc  = round(total_put_vol/(total_call_vol+total_put_vol),2)
    #total oi
    total_call_oi=odf.groupby(by="direction").sum().sort_values(by='direction',ascending=True)['openInterest'][0]
    total_put_oi=odf.groupby(by="direction").sum().sort_values(by='direction',ascending=True)['openInterest'][1]
    total_call_oi_perc = round(total_call_oi/(total_call_oi+total_put_oi),2)
    total_put_oi_perc  = round(total_put_oi/(total_call_oi+total_put_oi),2)
    #total volume prems (current)
    total_call_vol_prems=odf.groupby(by="direction").sum().sort_values(by='direction',ascending=True)['vol_prems'][0]
    total_put_vol_prems=odf.groupby(by="direction").sum().sort_values(by='direction',ascending=True)['vol_prems'][1]
    #total oi prems (current)
    total_call_oi_prems=odf.groupby(by="direction").sum().sort_values(by='direction',ascending=True)['oi_prems'][0]
    total_put_oi_prems=odf.groupby(by="direction").sum().sort_values(by='direction',ascending=True)['oi_prems'][1]

    #####WEEKLY VOLUME RATIOS
    #total WEEKLY volume
    W_total_call_vol =odf[odf["Weekly"] == True].groupby(by="direction").sum().sort_values(by='direction',ascending=True)['volume'][0]
    W_total_put_vol  =odf[odf["Weekly"] == True].groupby(by="direction").sum().sort_values(by='direction',ascending=True)['volume'][1]
    W_total_call_vol_perc = round(W_total_call_vol/(W_total_call_vol+W_total_put_vol),2)
    W_total_put_vol_perc  = round(W_total_put_vol/(W_total_call_vol+W_total_put_vol),2)
    #total WEEKLY oi
    W_total_call_oi=odf[odf["Weekly"] == True].groupby(by="direction").sum().sort_values(by='direction',ascending=True)['openInterest'][0]
    W_total_put_oi=odf[odf["Weekly"] == True].groupby(by="direction").sum().sort_values(by='direction',ascending=True)['openInterest'][1]
    W_total_call_oi_perc = round(W_total_call_oi/(W_total_call_oi+W_total_put_oi),2)
    W_total_put_oi_perc  = round(W_total_put_oi/(W_total_call_oi+W_total_put_oi),2)
    #total WEEKLY volume prems (current)
    W_total_call_vol_prems=odf[odf["Weekly"] == True].groupby(by="direction").sum().sort_values(by='direction',ascending=True)['vol_prems'][0]
    W_total_put_vol_prems=odf[odf["Weekly"] == True].groupby(by="direction").sum().sort_values(by='direction',ascending=True)['vol_prems'][1]
    W_total_call_vol_prems_perc =round(W_total_call_vol_prems/(W_total_call_vol_prems+W_total_put_vol_prems),2)
    W_total_put_vol_prems_perc  =round(W_total_put_vol_prems/(W_total_call_vol_prems+W_total_put_vol_prems),2)
    #total WEEKLY oi prems(current)
    W_total_call_oi_prems=odf[odf["Weekly"] == True].groupby(by="direction").sum().sort_values(by='direction',ascending=True)['oi_prems'][0]
    W_total_put_oi_prems=odf[odf["Weekly"] == True].groupby(by="direction").sum().sort_values(by='direction',ascending=True)['oi_prems'][1]
    W_total_call_oi_prems_perc =round(W_total_call_oi_prems/(W_total_call_oi_prems+W_total_put_oi_prems),2)
    W_total_put_oi_prems_perc  =round(W_total_put_oi_prems/(W_total_call_oi_prems+W_total_put_oi_prems),2)

    ##>>>>> MetaaL's top 3 Weekly Options
    # TOTAL COLLECTIVE GREATEST openInterest STRIKE for WEEKLY PUTS
    t3_PUT_Weekly_TCoiStrike = odf.loc[(odf["direction"] == 'Put') & (odf["Weekly"] == True)]\
                        .groupby('strike')['openInterest'].sum()\
                        .sort_values(ascending=False).reset_index()[:3].strike
    t3_PUT_Weekly_TCoiStrike_1 = int(t3_PUT_Weekly_TCoiStrike[0]) #1st greatest total collective OI strike
    t3_PUT_Weekly_TCoiStrike_2 = int(t3_PUT_Weekly_TCoiStrike[1]) #2nd greatest total collective OI strike
    t3_PUT_Weekly_TCoiStrike_3 = int(t3_PUT_Weekly_TCoiStrike[2]) #3rd greatest total collective OI strike

    # TOTAL COLLECTIVE GREATEST volume STRIKE for WEEKLY PUTS
    t3_PUT_Weekly_TCvolStrike = odf.loc[(odf["direction"] == 'Put') & (odf["Weekly"] == True)]\
                        .groupby('strike')['volume'].sum()\
                        .sort_values(ascending=False).reset_index()[:3].strike
    t3_PUT_Weekly_TCvolStrike_1 = int(t3_PUT_Weekly_TCvolStrike[0]) #1st greatest total collective vol strike
    t3_PUT_Weekly_TCvolStrike_2 = int(t3_PUT_Weekly_TCvolStrike[1]) #2nd greatest total collective vol strike
    t3_PUT_Weekly_TCvolStrike_3 = int(t3_PUT_Weekly_TCvolStrike[2]) #3rd greatest total collective vol strike

    # TOTAL COLLECTIVE GREATEST openInterest STRIKE for WEEKLY CALLS
    t3_CALL_Weekly_TCoiStrike = odf.loc[(odf["direction"] == 'Call') & (odf["Weekly"] == True)]\
                        .groupby('strike')['openInterest'].sum()\
                        .sort_values(ascending=False).reset_index()[:3].strike
    t3_CALL_Weekly_TCoiStrike_1 = int(t3_CALL_Weekly_TCoiStrike[0]) #1st greatest total collective OI strike
    t3_CALL_Weekly_TCoiStrike_2 = int(t3_CALL_Weekly_TCoiStrike[1]) #2nd greatest total collective OI strike
    t3_CALL_Weekly_TCoiStrike_3 = int(t3_CALL_Weekly_TCoiStrike[2]) #3rd greatest total collective OI strike

    # TOTAL COLLECTIVE GREATEST volume STRIKE for WEEKLY CALLS
    t3_CALL_Weekly_TCvolStrike = odf.loc[(odf["direction"] == 'Call') & (odf["Weekly"] == True)]\
                        .groupby('strike')['volume'].sum()\
                        .sort_values(ascending=False).reset_index()[:3].strike
    t3_CALL_Weekly_TCvolStrike_1 = int(t3_CALL_Weekly_TCvolStrike[0]) #1st greatest total collective vol strike
    t3_CALL_Weekly_TCvolStrike_2 = int(t3_CALL_Weekly_TCvolStrike[1]) #2nd greatest total collective vol strike
    t3_CALL_Weekly_TCvolStrike_3 = int(t3_CALL_Weekly_TCvolStrike[2]) #3rd greatest total collective vol strike

    return total_call_vol, total_put_vol, total_call_vol_perc, total_put_vol_perc, \
           total_call_oi, total_put_oi, total_call_oi_perc, total_put_oi_perc, \
           total_call_vol_prems, total_put_vol_prems, total_call_oi_prems, total_put_oi_prems, \
           W_total_call_vol, W_total_put_vol, W_total_call_vol_perc, W_total_put_vol_perc, \
           W_total_call_oi, W_total_put_oi, W_total_call_oi_perc, W_total_put_oi_perc, \
           W_total_call_vol_prems, W_total_put_vol_prems, W_total_call_vol_prems_perc, W_total_put_vol_prems_perc, \
           W_total_call_oi_prems, W_total_put_oi_prems, W_total_call_oi_prems_perc, W_total_put_oi_prems_perc, \
           t3_PUT_Weekly_TCoiStrike, t3_PUT_Weekly_TCoiStrike_1, t3_PUT_Weekly_TCoiStrike_2, t3_PUT_Weekly_TCoiStrike_3, \
           t3_PUT_Weekly_TCvolStrike, t3_PUT_Weekly_TCvolStrike_1, t3_PUT_Weekly_TCvolStrike_2, t3_PUT_Weekly_TCvolStrike_3, \
           t3_CALL_Weekly_TCoiStrike, t3_CALL_Weekly_TCoiStrike_1, t3_CALL_Weekly_TCoiStrike_2, t3_CALL_Weekly_TCoiStrike_3, \
           t3_CALL_Weekly_TCvolStrike, t3_CALL_Weekly_TCvolStrike_1, t3_CALL_Weekly_TCvolStrike_2, t3_CALL_Weekly_TCvolStrike_3

#@st.cache(allow_output_mutation=True)
#@st.experimental_memo(ttl=300)
def load_pcratios():
    #Total put call ratios
    tpcr = go.Figure()
    tpcr.add_trace(go.Bar(
        y=['Volume', 'Open Interest'],
        x=[total_call_vol_perc, total_call_oi_perc],
        name='Calls',
        orientation='h',
        marker=dict(
            color='rgba(63, 191, 116, 0.6)',
            line=dict(color='rgba(63, 191, 116, 1.0)', width=3)
        )
    ))
    tpcr.add_trace(go.Bar(
        y=['Volume', 'Open Interest'],
        x=[total_put_vol_perc, total_put_oi_perc],
        name='Puts',
        orientation='h',
        marker=dict(
            color='rgba(191, 63, 74, 0.6)',
            line=dict(color='rgba(191, 63, 74, 1.0)', width=3)
        )
    ))

    tpcr.update_layout(barmode='stack',
                    paper_bgcolor='rgb(248, 248, 255)',
                    plot_bgcolor='rgb(248, 248, 255)',
                    autosize=False,
                    width=500,
                    height=100,
                    title={
                        'text': "Total",'y':.94,'x':0.5,'xanchor': 'center','yanchor': 'top'},
                    xaxis=dict(tickformat='.0%'),
                    margin=dict(
                        l=5,
                        r=5,
                        b=10,
                        t=35,
                        pad=4
                    ),
                    )
    #Weekly put call ratios
    wpcr = go.Figure()
    wpcr.add_trace(go.Bar(
        y=['Volume', 'Open Interest'],
        x=[W_total_call_vol_perc, W_total_call_oi_perc],
        name='Calls',
        orientation='h',
        marker=dict(
            color='rgba(63, 191, 116, 0.6)',
            line=dict(color='rgba(63, 191, 116, 1.0)', width=3)
        )
    ))
    wpcr.add_trace(go.Bar(
        y=['Volume', 'Open Interest'],
        x=[W_total_put_vol_perc, W_total_put_oi_perc],
        name='Puts',
        orientation='h',
        marker=dict(
            color='rgba(191, 63, 74, 0.6)',
            line=dict(color='rgba(191, 63, 74, 1.0)', width=3)
        )
    ))

    wpcr.update_layout(barmode='stack',
                    paper_bgcolor='rgb(248, 248, 255)',
                    plot_bgcolor='rgb(248, 248, 255)',
                    autosize=False,
                    width=500,
                    height=100,
                    title={
                        'text': "Weeklies",'y':.94,'x':0.5,'xanchor': 'center','yanchor': 'top'},
                    xaxis=dict(tickformat='.0%'),
                    margin=dict(
                        l=15,
                        r=15,
                        b=10,
                        t=35,
                        pad=4
                    ),
                    )  
    return tpcr, wpcr

# ----------------------------------- Multi-use / Misc. Variables -----------------------------------

#@st.cache
@st.experimental_memo(ttl=300, max_entries=10)
def load_multi_use_vars(df):
    #Multi-use case variables
    lp = df.tail(1).close[0]  #===> last price 
    lpp = df.tail(2).close[0]  #===> prior period price 
    oi_min = lp*.2 #===> 80% threshold for OI strike consideration, could convert this to user controlled variable
    return lp, lpp, oi_min

# ----------------------------------- Load Seasonality Charts -----------------------------------

#@st.cache(allow_output_mutation=True)
@st.experimental_memo(ttl=300, max_entries=10)
def seas_charts(tick, cal_input):
    #download data for seasonality based on a Calendar input they can change
    df= yf.download(tick,cal_input,datetime.date.today().strftime("%Y-%m-%d"))
    df.reset_index(inplace=True)
    #create pct changes 
    df['daily_change'] = round(df['Adj Close'].pct_change(1),4)
    #df['monthly_change'] = round(df['Adj Close'].pct_change(21),4) 
    #df['yearly_change'] = round(df['Adj Close'].pct_change(252),4)
    df['month'] = df['Date'].dt.month
    df['day'] = df['Date'].dt.day
    df['weekday'] = df['Date'].dt.weekday
    df['year'] = df['Date'].dt.year
    df['wd_name'] = df['weekday'].apply(lambda w:calendar.day_name[w])
    df['m_name'] = df['month'].apply(lambda m:calendar.month_name[m])
    #create daily aggregated
    day_seas = df.groupby('wd_name')['daily_change'].mean()
    day_seas=day_seas.to_frame()
    day_seas=day_seas.reset_index()
    day_seas['daily_change'] = round(day_seas['daily_change'],4)
    day_seas['wd_num'] = day_seas['wd_name'].apply( lambda d:time.strptime(d,"%A").tm_wday )
    day_seas = day_seas.sort_values(by="wd_num")
    #create monthly_change
    s = df[['Date','Adj Close']]
    s.set_index('Date',inplace=True)
    m_change = s.resample('BM').apply(lambda x: x[-1])
    m_change['perc_change'] = m_change['Adj Close'].pct_change()
    m_change = m_change.reset_index()
    m_change['m_num'] = m_change['Date'].dt.month
    m_change['m_name'] = m_change['m_num'].apply(lambda m:calendar.month_name[m])
    m_change['year'] = m_change['Date'].dt.year
    #create monthly aggregate
    m_change_avg = m_change.groupby('m_name')['perc_change'].mean()
    m_change_avg = m_change_avg.to_frame()
    m_change_avg = m_change_avg.reset_index()
    m_change_avg['perc_change'] = round(m_change_avg['perc_change'],3)
    m_change_avg['m_num'] = m_change_avg['m_name'].apply( lambda m:datetime.datetime.strptime(m, '%B').month )                            
    m_change_avg = m_change_avg.sort_values(by="m_num")
    #daily seas chart
    colors = ['seagreen' if row['daily_change'] >= 0 else 'red' for index, row in day_seas.iterrows()]
    wd_fig = go.Figure()
    wd_fig.add_trace(go.Bar(x=day_seas['wd_name'],
                        y=day_seas['daily_change'],
                        marker_color = colors,
                        text= day_seas['daily_change'].apply( lambda e:"{:.2%}".format(e) ) ,
                        textposition='auto'
                        )
                )
    wd_fig.update_layout(title=utick.upper()+' Daily Seasonality from '+str(df['year'].min())+' to '+str(df['year'].max()) )
    wd_fig.update_yaxes(visible=False)
    #daily seas trend chart
    dt_fig = go.Figure()
    dt_fig.add_trace(go.Scatter(x=df[df['wd_name']=='Monday']['Date'], y=df[df['wd_name']=='Monday']['daily_change'].rolling(inp_ma).mean(),name='Monday',line=dict(color='rgb(2,48,71)', width=4)))
    dt_fig.add_trace(go.Scatter(x=df[df['wd_name']=='Tuesday']['Date'], y=df[df['wd_name']=='Tuesday']['daily_change'].rolling(inp_ma).mean(),name='Tuesday',line=dict(color='rgb(33, 158, 188)', width=4)))
    dt_fig.add_trace(go.Scatter(x=df[df['wd_name']=='Wednesday']['Date'], y=df[df['wd_name']=='Wednesday']['daily_change'].rolling(inp_ma).mean(),name='Wednesday',line=dict(color='rgb(142, 202, 230)', width=4)))
    dt_fig.add_trace(go.Scatter(x=df[df['wd_name']=='Thursday']['Date'], y=df[df['wd_name']=='Thursday']['daily_change'].rolling(inp_ma).mean(),name='Thursday',line=dict(color='rgb(255, 183, 3)', width=4)))
    dt_fig.add_trace(go.Scatter(x=df[df['wd_name']=='Friday']['Date'], y=df[df['wd_name']=='Friday']['daily_change'].rolling(inp_ma).mean(),name='Friday',line=dict(color='rgb(251, 133, 0)', width=4)))
    dt_fig.update_layout(title=str(inp_ma)+'sma for Daily % Change')
    dt_fig.update_yaxes(tickformat=',.1%')
    #monthly aggregate chart
    ma_colors = ['seagreen' if row['perc_change'] >= 0 else 'red' for index, row in m_change_avg.iterrows()]
    ma_fig = go.Figure()
    ma_fig.add_trace(go.Bar(x=m_change_avg['m_name'],
                        y=m_change_avg['perc_change'],
                        marker_color = ma_colors,
                        text= m_change_avg['perc_change'].apply( lambda e:"{:.1%}".format(e) ) ,
                        textposition='auto'
                        )
                )
    ma_fig.update_layout(title=utick.upper()+' Monthly Seasonality from '+str(df['year'].min())+' to '+str(df['year'].max()) )
    ma_fig.update_yaxes(visible=False)
    #monthly by year
    myr_fig = go.Figure()
    for each in m_change['year'].unique():
        colors = ['seagreen' if row['perc_change'] >= 0 else 'red' for index, row in m_change[m_change['year']==each].iterrows()]
        myr_fig.add_trace(go.Bar(x=m_change[m_change['year']==each]['m_name'],y=m_change[m_change['year']==each]['perc_change'],marker_color = colors, text= m_change[m_change['year']==each]['perc_change'].apply( lambda e:"{:.1%}".format(e) ),name=str(each)))

    myr_fig.update_layout(title=utick.upper()+' Monthly Seasonality from '+str(df['year'].min())+' to '+str(df['year'].max())+', by year' )
    myr_fig.update_yaxes(tickformat=',.0%')
    myr_fig.update_layout(barmode='group', showlegend=False)
    #daily change w/ volume colors
    #this chart APPEARS to be too intensive to load for data points past 5 yrs, it kept crashing when going back to 2010 or earlier, could come back and troubleshoot but likely just phase this out
    #dcv_fig = go.Figure()
    #dcv_fig.add_trace(go.Scatter(x=df['Date'], y=df['daily_change'], mode='markers',name='daily_change'\
    #                        ,marker=dict(color=df['Volume'], showscale=True)))
    #dcv_fig.update_layout(title=utick+' Daily % Change, colored by volume')
    #dcv_fig.update_yaxes(ticklabelposition="inside top", title='Daily % Change')
    #dcv_fig.update_layout(uniformtext_minsize=8, uniformtext_mode='hide')


    #plot charts
    return wd_fig,ma_fig,myr_fig,dt_fig

# ----------------------------------- Load Ideas from Google Docs & Twitter -----------------------------------

#@st.cache(suppress_st_warning=True)
def load_ac_tweets():
    bt=st.secrets["TWEEPY_BT"]
    client = tweepy.Client(bearer_token=bt)
    TWITTER_USERNAMES = ['Xiuying', 'artem_essega','MetaaL_','TradesWithTom','nutfreak26','SpinTrades'] 
    z=0
    selected_ticker = utick.upper()
    for username in TWITTER_USERNAMES:
        query = 'from:'+username
        tweets = client.search_recent_tweets(query=query, tweet_fields=['created_at'],
                                            media_fields=['media_key','type','preview_image_url'], 
                                            expansions='attachments.media_keys',
                                            max_results=100)
        for tweet in tweets.data:
            words = tweet.text.split(' ')
            for word in words:
                if word.startswith('$') and word[1:].isalpha():
                    symbol = word[1:].upper()
                    tweet_date = tweet.created_at.strftime('%B %d, %Y')
                    if symbol == selected_ticker:
                        st.markdown('Idea #'+str(z)+' -- created on '+ tweet_date+(' by '+'**{}**').format(username))
                        st.markdown(tweet.text)
                        st.markdown('--------------------------------')
                        z+=1
    if z == 0:
        st.markdown('No tweets could be found for '+str(utick.upper())) 
                        
def load_ac_ideas():
    #acess public gsheets csv file and store in dataframe
    gdf = pd.read_csv('https://docs.google.com/spreadsheets/d/e/2PACX-1vSAOcVEnPyR3jB_k9uAgN5GIIFI2S_d4l9Lt6w0XnnJMEFKKZVLxiiNP5P5RxqUHBXJl1vjV-4APFJY/pub?output=csv')
    #pre-process dataframe
        #rename columns
        #importantly updating the 'image_upload' into correct format
        #convert ticker to uppercase
    gdf = gdf.rename(columns={ "Select your username": "username",'Enter stock ticker:':'ticker','Describe your trade idea':'idea',
                                'What is your Price Target? (enter number)': 'price_target', 'What day do you project this Price Target to be hit by?':'pt_date',
                                'Upload image to accompany idea':'image_upload'
                    })
    new_dtypes = {"pt_date": 'datetime64[ns]',"Timestamp": 'datetime64[ns]', "ticker": 'string', "username": 'string', "idea": 'string'}
    for index, row in gdf.iterrows():
        if not pd.isna(row['image_upload']):
            row['image_upload'] = list(row['image_upload'])
            row['image_upload'][25:29] = ['u','c']
            row['image_upload']="".join(row['image_upload'])
            gdf.at[index,'image_upload'] = row['image_upload']    
    gdf = gdf.astype(new_dtypes)    
    gdf['ticker'] = gdf['ticker'].str.upper()
    idea_count =0
    selected_ticker = utick.upper()
    #todo: put dataframe in order of most recent to latest

    #Loop through dataframe
    #1st, check if selected ticker 'utick' is equal to 'ticker' column
        #if true, print each idea with streamlit print commands
        #if false, print something salty like "No ideas returned, Axe Cap is out of milk"
    for index, row in gdf.iterrows():
        if row['ticker'] == selected_ticker:
            idea_count+=1
    st.markdown("There are currently "+str(idea_count)+' idea(s).')
    for index, row in gdf.iterrows():
        if row['ticker'] == selected_ticker:
            st.markdown(':wave: ' +('**{}**').format(row['username']) + '--'+ 'created on '+str(row['Timestamp'].strftime('%B %d, %Y')) )
            st.markdown(':pushpin: **Price Target:** '+str(row['price_target']) + ' by ' + str(row['pt_date'].strftime('%B %d, %Y')) )  
            st.markdown(':bulb: **Idea:** '+row['idea']) 
            st.image(row['image_upload'])
            st.markdown('--------------------------------')

@st.experimental_memo(ttl=300, max_entries=10)
def load_gsheet_data():
    df = pd.read_csv('https://docs.google.com/spreadsheets/d/e/2PACX-1vS-8BzYdnRw_BMYJCEy9HkPX-CIcgBilenf4VCFAfGAacooTCCdhdiLCcJEFvkUysfsPZzp9fOrSOMZ/pub?output=csv')
    return df

def load_gantt():
    df = load_gsheet_data()
    gdf = df[['Enter stock ticker:','Timestamp','What day do you project this Price Target to be hit by?','Hit','Win Date']]
    for i, row in gdf.iterrows():
        if str(row['Hit'])=='False':
            gdf.at[i,'Win Date'] = row['What day do you project this Price Target to be hit by?']
    gdf = gdf.rename(columns={"Enter stock ticker:": "Task", "Timestamp": "Start",'Win Date':'Finish'})
    colors = {False: 'rgb(220, 0, 0)', True: 'rgb(0, 255, 100)'}

    fig = ff.create_gantt(gdf, colors=colors, index_col='Hit', show_colorbar=True,
                        group_tasks=True)
    return fig

def load_rankings_table():
    df = load_gsheet_data()
    number_of_ideas = df.groupby('Select your username')["Timestamp"].count().sort_values(ascending=False).rename_axis(['Username']).rename("Number of Ideas") 
    avg_days_to_hit = df[df['Hit']==True].groupby('Select your username')["Days to Hit"].mean().sort_values(ascending=False).rename_axis(['Username']).rename("Avg Days to Hit")
    number_of_hits = df[df['Hit']==True].groupby('Select your username')["Timestamp"].count().sort_values(ascending=False).rename_axis(['Username']).rename("Number of Hits") 
    max_up = df.groupby('Select your username')["Max Move Upside"].max().sort_values(ascending=False).rename_axis(['Username']).rename("Max Upside %") 
    max_down = df.groupby('Select your username')["Max Move Downside"].min().sort_values(ascending=False).rename_axis(['Username']).rename("Max Downside %") 
    avg_max_up = df.groupby('Select your username')["Max Move Upside"].mean().sort_values(ascending=False).rename_axis(['Username']).rename("Avg Upside %") 
    avg_max_down = df.groupby('Select your username')["Max Move Downside"].mean().sort_values(ascending=False).rename_axis(['Username']).rename("Avg Downside %")
    #concat all the series into one DataFrame
    tdf = pd.concat([number_of_ideas,number_of_hits,avg_days_to_hit, max_up, max_down, avg_max_up, avg_max_down],axis=1)
    #Calculate Win Rates
    tdf['Win Rate'] = tdf['Number of Hits'] / tdf['Number of Ideas']
    #Fill Nas
    tdf['Win Rate']  = tdf['Win Rate'].fillna(0)
    tdf['Number of Ideas']  = tdf['Number of Ideas'].fillna(0)
    tdf['Number of Hits']  = tdf['Number of Hits'].fillna(0)
    tdf['Avg Days to Hit']  = tdf['Avg Days to Hit'].fillna(0)
    tdf['Max Upside %']  = tdf['Max Upside %'].fillna(0)
    tdf['Max Downside %']  = tdf['Max Downside %'].fillna(0)
    tdf['Avg Upside %']  = tdf['Avg Upside %'].fillna(0)
    tdf['Avg Downside %']  = tdf['Avg Downside %'].fillna(0)
    tdf_styler = ({'Number of Hits': '{:,.0f}','Avg Days to Hit': '{:,.0f}',
                    'Max Upside %': '{:.2%}','Max Downside %': '{:.2%}',
                    'Avg Upside %': '{:.2%}','Avg Downside %': '{:.2%}',
                    'Win Rate': '{:.2%}'})
    st.dataframe(tdf.style.format(tdf_styler))
           


# ----------------------------------- SIDEBAR -----------------------------------

#### USER INPUT / INITIAL SIDEBAR VARIABLES ####
st.sidebar.image('Axe-cap-custom-logo.png')
if st.sidebar.button("Refresh Data"):
    # Clear all in-memory and on-disk memo caches; this clears values from *all* memoized functions:
    st.experimental_memo.clear()
    # Clear values from *all* st.cache functions:
    # st.legacy_caching.clear_cache()
    
st.sidebar.write("**[Submit trade idea?](https://forms.gle/aNfTSnnjuss68Nrq5)**")
#Placeholder to implement once a second view/dashboard has been created to toggle between.
#view_type = st.sidebar.selectbox(
#     'Select dashboard view',
#     ('Primary (Stock & Options)','Research (Axe Cap Reports)','Economic Insights (Trend Analysis)', 'Models & Correlation (Explore What Moves Markets)'))

#st.sidebar.title('Axe Cap Terminal')
utick = st.sidebar.text_input('Enter valid ticker below', 'AMD', help="Please select valid NYSE/NASDAQ & Optionable ticker. Only optionable data available on Yahoo Finance will work currently.")

tp = st.sidebar.selectbox(
     'Select length of time for analysis',
     ('2y','1d','5d','1mo','3mo','6mo','1y','2y','5y','10y','ytd','max'), help="This is the length of time up until today to use for charting")
#Issue with intervals is that smaller timeframes will result in error if time period for analysis is not adjusted first, such as:
##----15m data not available for startTime=1611327019 and endTime=1642863019. The requested range must be within the last 60 days.
intv = st.sidebar.selectbox(
     'Select candle timeframe interval for analysis',
     ('1d','1d'), help="This is the candle length used for charting, currently Daily is the most stable interval until Pandas-TA strateies are modified for filling NaNs")
     #('1d','1m','2m','5m','15m','30m','60m','90m','1h','5d','1wk','1mo','3mo') << Replace w/ this once interval testing has been completed
ind_selected = st.sidebar.multiselect(
     'Select indicators',
     ['Bossx_MovingAverages', 'Bossx_GOI_GVOL_Strikes', 'Metaal_OptionLevels', 'Mach_SwingArms', 'Xiu_ADR','Xiu_Hull','Xiu_QQE'],
     ['Metaal_OptionLevels', 'Xiu_QQE'])

if 'Metaal_OptionLevels' in ind_selected: 
    with st.sidebar.expander("MetaaL's Option Switches"):
        put_OI = st.checkbox('Show Weekly Put OI Strike Levels?',value=False)
        put_vol = st.checkbox('Show Weekly Put Vol Strike Levels?',value=True)
        call_OI = st.checkbox('Show Weekly Call OI Strike Levels?',value=False)
        call_vol = st.checkbox('Show Weekly Call vol Strike Levels?',value=True)
else:
    put_OI = False
    put_vol = True
    call_OI = False
    call_vol = True
    
st.sidebar.caption('The data and information shared here is not financial advice. Investments or strategies mentioned on this website may not be suitable for you. Data and material on this website may be incorrect or delaying real-time financial data at any given time.')

# ----------------------------------- LOAD STOCK & OPTION DATA -----------------------------------

##>>START DATA LOAD BASED ON DEFAULT VALUES<<##
df,tk = load_data(utick)

#Multi-use case variables
lp, lpp, oi_min = load_multi_use_vars(df)

#Full Options dataframe
odf = options_chain(tk)

#Volume & OI dataframe
voi, hoi_strike, mvs, hv_strike_1,hv_strike_2,hv_strike_3,hoi_strike_1,hoi_strike_2,hoi_strike_3 = load_voi(odf)

#Specific strikes and ratios based on volume, open intereste, direction ,etc.
total_call_vol, total_put_vol, total_call_vol_perc, total_put_vol_perc, \
total_call_oi, total_put_oi, total_call_oi_perc, total_put_oi_perc, \
total_call_vol_prems, total_put_vol_prems, total_call_oi_prems, total_put_oi_prems, \
W_total_call_vol, W_total_put_vol, W_total_call_vol_perc, W_total_put_vol_perc, \
W_total_call_oi, W_total_put_oi, W_total_call_oi_perc, W_total_put_oi_perc, \
W_total_call_vol_prems, W_total_put_vol_prems, W_total_call_vol_prems_perc, W_total_put_vol_prems_perc, \
W_total_call_oi_prems, W_total_put_oi_prems, W_total_call_oi_prems_perc, W_total_put_oi_prems_perc, \
t3_PUT_Weekly_TCoiStrike, t3_PUT_Weekly_TCoiStrike_1, t3_PUT_Weekly_TCoiStrike_2, t3_PUT_Weekly_TCoiStrike_3, \
t3_PUT_Weekly_TCvolStrike, t3_PUT_Weekly_TCvolStrike_1, t3_PUT_Weekly_TCvolStrike_2, t3_PUT_Weekly_TCvolStrike_3, \
t3_CALL_Weekly_TCoiStrike, t3_CALL_Weekly_TCoiStrike_1, t3_CALL_Weekly_TCoiStrike_2, t3_CALL_Weekly_TCoiStrike_3, \
t3_CALL_Weekly_TCvolStrike, t3_CALL_Weekly_TCvolStrike_1, t3_CALL_Weekly_TCvolStrike_2, t3_CALL_Weekly_TCvolStrike_3 = load_strikes(odf)


# ----------------------------------- LOAD INDICATORS -----------------------------------

APMMonthly(df)
SwingArms(df)
TotalStrat = load_indicator_strategy()
df.ta.strategy(TotalStrat)
MashumeHull(df)

##=======>Placeholder for topline metrics, currently just have the current price with % change from prior period closing price, default is daily
#col1, col2, col3 = st.columns(3)
#col1.metric(label='$'+utick+'last price', value=round(lp,2), delta=str(round(((lp/lpp)-1),4)*100)+'%')
#col2.metric(label='$'+utick, value=round(lp,2), delta=str(round(((lp/lpp)-1),4)*100)+'%')
#col3.metric(label='$'+utick, value=round(lp,2), delta=str(round(((lp/lpp)-1),4)*100)+'%')

# ----------------------------------- PLOT AXE CAP STRATEGY CHARTS -----------------------------------
plotly_config={'modeBarButtonsToAdd': ['drawline','drawopenpath','drawclosedpath','drawcircle','drawrect','eraseshape']}
st.subheader('$'+utick.upper()+' (' +str(round(lp,2))+')')
main_chart = load_main_chart(df)
st.plotly_chart(main_chart, use_container_width=True,config=plotly_config)

# ----------------------------------- PLOT PUT/CALL RATIOS -----------------------------------
st.subheader('$'+utick.upper()+' Option Chain')
tpcr, wpcr = load_pcratios()
pcr_col_1, pcr_col_2 = st.columns(2)
pcr_col_1.plotly_chart(tpcr, config={'displayModeBar': False})
pcr_col_2.plotly_chart(wpcr, config={'displayModeBar': False})

# ----------------------------------- PLOT OPTION TABLE -----------------------------------
cb_1, cb_2, cb_3 = st.columns(3)
with cb_1:
    cb_weeklies = st.checkbox('Only show weekly options?',value=False, help="Filters option table to only show current/upcoming Weekly/Monthly options")
with cb_2:
    cb_todays_trades = st.checkbox('Only show Today\'s Trades?',value=False, help="Since Yahoo finance options are on a 15m delay, this filter can help show only the chain that is most up to date")
with cb_3:
    cb_spiking = st.checkbox('Only show Vol > OI options?',value=False, help="See which contracts are particulary popular today")

#To style the dataframe to be more end-user friendly for legibility
df_styler = ({'volume': '{:,.0f}','openInterest': '{:,.0f}','strike': '{:,.2f}',"expirationDate": lambda t: t.strftime("%Y-%b-%d")})

if (cb_weeklies and cb_todays_trades==False and cb_spiking==False):
    st.dataframe(voi.loc[voi.Weekly].sort_values(by='volume', ascending=False).style.format(df_styler))
elif (cb_todays_trades and cb_weeklies==False  and cb_spiking==False):
    st.dataframe(voi.loc[voi.OnlyTodaysTrades].sort_values(by='volume', ascending=False).style.format(df_styler))
elif (cb_spiking and cb_todays_trades==False and cb_weeklies==False):
    st.dataframe(voi.loc[voi.bool_VolSpike].sort_values(by='volume', ascending=False).style.format(df_styler))
elif (cb_weeklies and cb_todays_trades and cb_spiking==False):
    st.dataframe(voi.loc[(voi.Weekly) & (voi.OnlyTodaysTrades)].sort_values(by='volume', ascending=False).style.format(df_styler))
elif (cb_spiking and cb_todays_trades and cb_weeklies==False):
    st.dataframe(voi.loc[(voi.bool_VolSpike) & (voi.OnlyTodaysTrades)].sort_values(by='volume', ascending=False).style.format(df_styler))
elif (cb_weeklies and cb_spiking and cb_todays_trades==False):
    st.dataframe(voi.loc[(voi.bool_VolSpike) & (voi.Weekly)].sort_values(by='volume', ascending=False).style.format(df_styler))
elif (cb_weeklies and cb_todays_trades and cb_spiking):
    st.dataframe(voi.loc[(voi.Weekly) & (voi.OnlyTodaysTrades) & (voi.bool_VolSpike)].sort_values(by='volume', ascending=False).style.format(df_styler))
else:
    st.dataframe(voi.sort_values(by='volume', ascending=False).style.format(df_styler))

# ----------------------------------- PLOT RANGE DISTRIBUTION -----------------------------------
with st.expander('$'+utick.upper()+' Price Range Distribution'):
    rd_1,rd_2 = st.columns(2)
    with rd_1:
        cb_range = st.checkbox('Show range distribution plots?',value=False)
    with rd_2:
        range_freq = st.slider('How may days for building range distribution?', min_value=5, max_value=1000, value=30)
    if cb_range:
        fig_range = load_range_dist(df,range_freq)
        op,lc,oc_range_mean,hl_range_mean,gap_range_mean,me_range_mean,ext_room,oc_msg,hl_msg,gap_msg,me_msg = load_range_means(df,range_freq)
        #st.text(utick.upper()+' has an average Open to Close range of: '+str(oc_range_mean))
        #st.text(utick.upper()+' opened at: ' + str(op) +' and is currently at: '+str(lc)+ ' which means: '+ ext_room)
        st.text(oc_msg)
        st.text(hl_msg)
        st.text(gap_msg)
        st.text(me_msg)
        st.plotly_chart(fig_range, use_container_width=True,config=plotly_config)
        st.caption('Open-Close: Distance from interval close to interval open')
        st.caption('High-Low: Distance from interval high to interval low')
        st.caption('Gap Up/Down Range: Distance from prior interval close to next interval open')
        st.caption('Max Extension Range: Distance from prior interval close to next interval high')

# ----------------------------------- PLOT SEASONALITY -----------------------------------
with st.expander("Explore $"+utick.upper()+" Seasonality"):
    cb_show_seas = st.checkbox('Show seasonality charts?', value=False)
    seas_col_1, seas_col_2 = st.columns(2)
    with seas_col_1:
        cal_input = st.date_input("Select start date for calculating seasonality (default is 2000 to date)",datetime.date(2000, 1, 1))
    with seas_col_2:
        inp_ma = st.number_input('How many weeks for moving average?',min_value=1,max_value=100,step=1,value=10)
    if cb_show_seas:
        wd_fig,ma_fig,myr_fig,dt_fig = seas_charts(utick,cal_input)
        st.plotly_chart(wd_fig, use_container_width=True,config=plotly_config)
        st.plotly_chart(dt_fig, use_container_width=True,config=plotly_config)
        st.plotly_chart(ma_fig, use_container_width=True,config=plotly_config)
        st.plotly_chart(myr_fig, use_container_width=True,config=plotly_config)

# ----------------------------------- DISPLAY TRADING IDEAS -----------------------------------
with st.expander("Explore Axe Cap Member's $"+utick.upper()+" Trading Ideas"):
    cb_show_ideas = st.checkbox('Show trading ideas?', value=False,help='Checks Axe Cap Twitter acocunts for tweets about the currently selected ticker')
    if cb_show_ideas:
        st.markdown('#### :bulb: Ideas submitted by Axe Cap Users through the Terminal. :bulb:')
        load_ac_ideas()
        st.markdown('#### :bird: Ideas from Axe Cap Twitter Accounts :bird:')
        load_ac_tweets()
        
# ----------------------------------- DISPLAY RANKINGS -----------------------------------
with st.expander("BETA: View Axe Cap User Rankings/Leaderboard"):
    cb_show_rankings = st.checkbox('Show rankings?', value=False,help='Computes performance and rankings for submitted trade ideas')
    if cb_show_rankings:
        st.markdown('###### Rankings')
        load_rankings_table()
        gantt = load_gantt()
        st.plotly_chart(gantt, use_container_width=True,config=plotly_config)

# ----------------------------------- DISPLAY SEAL OF APPROVALS -----------------------------------
#Joe's Seal of Approval --If a few folks contribute their scanner criteria for a bullish or bearish trade, can turn this into a few columns
#so it would be "Seal of Approvals"
with st.expander("See Joe's Seal of Approval"):
    joe_approved,lp_joe,lvc, lvp, joe_sma5, joe_sma20, joe_sma50, joe_sma200 = load_joegopgo()
    if joe_approved:
        st.subheader('$'+utick.upper()+' gets the JOE SEAL OF APPROVAL')
        st.text('Joe approves this Ticker because...')
        st.text('Last Price: '+str(lp_joe)+' is greater than five buckaroos')
        st.text('Last Price: '+str(lp_joe)+' is greater than the 20sma: '+str(joe_sma20))
        st.text('The 50sma: '+str(joe_sma50)+' is less than the 20sma: '+str(joe_sma20))
        st.text('The 200sma: '+str(joe_sma200)+' is less than the 5sma: '+str(joe_sma5))
        st.text('The current volume: '+str(lvc)+' is greater than one milly')
        st.text('The current volume: '+str(lvc)+' is greater than the prior period volume: '+str(lvp))
    else:
        st.subheader('JOE SAYS GTFO '+'$'+utick.upper())
        st.text('This ticker does not get the JOE SEAL OF APPROVAL')
