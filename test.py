import pandas as pd

def WildMA(x, length):
    # Wilder's moving average
    a = 1/length
    return x.ewm(alpha=a, min_periods=length).mean()

ATRPeriod= 28
ATRFactor = 5
print('hello')

var = [3.8300
,2.3400
,2.1700
,6.7000
,4.9000
,5.3900
,4.5800
,5.2000
,4.0100
,2.9000
,3.3700
,4.3350
,3.4500
,2.8700
,2.0800
,1.8800
,3.3700
,3.6900
,2.1800
,3.1400
,4.9900
,2.9700
,1.4400
,2.5300
,2.8400
,3.6800
,2.1600
,3.0600
,2.2100
,2.7800
,5.5600
,3.2300
,1.6500
,2.3600
,1.8200
,2.0500
,2.7700
,2.2600
,3.3000
,2.8800
,2.2200
,1.4300
,1.3500
,1.4600
,3.1400]

myvar = pd.Series(var)

wildvar = ATRFactor * WildMA(myvar, ATRPeriod)
print(wildvar)