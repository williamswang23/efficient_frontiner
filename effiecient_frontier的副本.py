#author:williams


##
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import quandl
##

quandl.ApiConfig.api_key = 'ir9irKQbEXWGvTcnJrCV'

# In[]
stocks = ['AAPL','AMZN','GOOGL','FB']
data = quandl.get_table('WIKI/PRICES', ticker = stocks,
                        qopts = { 'columns': ['date', 'ticker', 'adj_close'] },
                        date = { 'gte': '2016-1-1', 'lte': '2017-12-31' }, paginate=True)
data.head()

data.info()

df = data.set_index('date')
df.head()
##

table = df.pivot(columns='ticker')
# By specifying col[1] in below list comprehension
# You can select the stock names under multi-level column
table.columns = [col[1] for col in table.columns]
table.head()


plt.figure(figsize=(14, 7))
for c in table.columns.values:
    plt.plot(table.index, table[c], lw=3, alpha=0.8,label=c)
plt.legend(loc='upper left', fontsize=12)
plt.ylabel('price in $')
plt.show()



##

returns = table.pct_change()

plt.figure(figsize=(14, 7))
for c in returns.columns.values:
    plt.plot(returns.index, returns[c], lw=3, alpha=0.8,label=c)
plt.legend(loc='upper right', fontsize=12)
plt.ylabel('daily returns')

plt.show()
##

def portfolio_annualised_performance(weights, mean_returns, cov_matrix):
    returns = np.sum(mean_returns*weights ) *252
    std = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights))) * np.sqrt(252)
    return std, returns




def random_portfolios(num_portfolios, mean_returns, cov_matrix, risk_free_rate):
    results = np.zeros((3,num_portfolios))
    weights_record = []
    for i in range(num_portfolios):
        weights = np.random.random(4)
        weights /= np.sum(weights)
        weights_record.append(weights)
        portfolio_std_dev, portfolio_return = portfolio_annualised_performance(weights, mean_returns, cov_matrix)
        results[0,i] = portfolio_std_dev
        results[1,i] = portfolio_return
        results[2,i] = (portfolio_return - risk_free_rate) / portfolio_std_dev
    return results, weights_record
##
returns = table.pct_change()
mean_returns = returns.mean()
cov_matrix = returns.cov()
num_portfolios = 25000
risk_free_rate = 0.0178

##
def display_simulated_ef_with_random(mean_returns, cov_matrix, num_portfolios, risk_free_rate):
    results, weights = random_portfolios(num_portfolios, mean_returns, cov_matrix, risk_free_rate)

    max_sharpe_idx = np.argmax(results[2])
    sdp, rp = results[0, max_sharpe_idx], results[1, max_sharpe_idx]
    max_sharpe_allocation = pd.DataFrame(weights[max_sharpe_idx], index=table.columns, columns=['allocation'])
    max_sharpe_allocation.allocation = [round(i * 100, 2) for i in max_sharpe_allocation.allocation]
    max_sharpe_allocation = max_sharpe_allocation.T

    min_vol_idx = np.argmin(results[0])
    sdp_min, rp_min = results[0, min_vol_idx], results[1, min_vol_idx]
    min_vol_allocation = pd.DataFrame(weights[min_vol_idx], index=table.columns, columns=['allocation'])
    min_vol_allocation.allocation = [round(i * 100, 2) for i in min_vol_allocation.allocation]
    min_vol_allocation = min_vol_allocation.T



    print("-----------------")
    print ("Maximum Sharpe Ratio Portfolio Allocation\n")
    print ("Annualised Return:", round(rp, 2))
    print ("Annualised Volatility:", round(sdp, 2))
    print ("\n")
    print (max_sharpe_allocation)
    print ("-" * 80)
    print ("Minimum Volatility Portfolio Allocation\n")
    print ("Annualised Return:", round(rp_min, 2))
    print ("Annualised Volatility:", round(sdp_min, 2))
    print ("\n")
    print (min_vol_allocation)



    plt.figure(figsize=(10, 7))
    plt.scatter(results[0, :], results[1, :], c=results[2, :], cmap='YlGnBu', marker='o', s=10, alpha=0.3)
    plt.colorbar()
    plt.scatter(sdp, rp, marker='*', color='r', s=500, label='Maximum Sharpe ratio')
    plt.scatter(sdp_min, rp_min, marker='*', color='g', s=500, label='Minimum volatility')
    plt.title('Simulated Portfolio Optimization based on Efficient Frontier')
    plt.xlabel('annualised volatility')
    plt.ylabel('annualised returns')
    plt.legend(labelspacing=0.8)



##
display_simulated_ef_with_random(mean_returns, cov_matrix, num_portfolios, risk_free_rate)
plt.show()



# In[]
"""asset allocation"""

"""input stock code below """

stockname_input= ['AAPL','AMZN','GOOGL','FB','GM','AMD','MSFT','MCD','MMM','SHLD','WMT','INTC']


# In[]

def stockpridownload (stockname):
    ic=((len(stockname))-1)
    df_port=pd.DataFrame()
    idx_sn=1
    pricex=quandl.get_table('WIKI/PRICES', ticker=stockname[0],
                             qopts={'columns': ['date', 'adj_close']},
                             date={'gte': '2017-1-1', 'lte': '2019-12-31'}, paginate=True)
    df_port=pd.concat([df_port,pricex],axis=1)
    for idx_sn in range(ic):
        pricex=quandl.get_table('WIKI/PRICES', ticker = stockname[idx_sn],
                        qopts = { 'columns': ['adj_close'] },
                        date={'gte': '2017-1-1', 'lte': '2019-12-31'}, paginate=True);
        pd.DataFrame(pricex);
        pprix=pricex['adj_close'];
        df_port=pd.concat([df_port,pprix],axis=1);
        idx_sn=idx_sn+1;

    df_port=df_port.set_index('date')
    df_port.columns=stockname

    return df_port


portdf=stockpridownload(stockname_input)
portdf.head()
type(portdf)

# In[]
"""pd"""


"""weighted"""
def weighted_P(stocks):
    w_temp=np.random.rand(len(stocks))
    weighted=w_temp/w_temp.sum()
    return weighted

"""cov matrix"""
def cov(df_cov):
    covP=df_cov.cov()
    return covP

"""portfolio return"""
def return_P(stock_return, weighted):
    port_r=np.multiply(stock_return,weighted).sum()
    return port_r


"""single portfolio sigma"""

def sigma_P(weights, cov_mat):
    port_vol = np.sqrt(np.dot(weights.T, np.dot(cov_mat, weights))
    return port_vol

"""sample def"""

def sample_r(stock_return):
    sampler=np.random.choice(stock_return)
    return sampler


"""sample return list"""
def sample_r_list(stocknamelist):
    stock_return=[]
    idx_s_r=0
    for idx_s_r in range(len(stocknamelist)):
        s_r_temp=sample_r(portdf.iloc(idx_s_r,:))
        stock_return=np.append(stock_return,s_r_temp)
        idx_s_r=idx_s_r+1
    return stock_return



# In[]


""""super-parameter"""
sigmaP=0.1

mc1_amount=100000



""""mc-1"""
idx_mc1=0
return_cache=0
sigma_cache=0
weight_cache=0
port_cov=cov(portdf)
for idx_mc1 in range(mc1_amount):
    portweight=weighted_P(stockname_input)
    samplerl=sample_r_list(stockname_input)
    portr=return_P(samplerl,portweight)
    ports=sigma_P(portweight,port_cov)
    if ports<sigmaP:
        return_pri=portr
        sigma_pri=ports
        weight_pri=portweight
    else:
        pass
    if return_pri > return_cache:
        return_cache=return_pri
        sigma_cache=sigma_pri
        weight_cache=weight_pri
        #return_cache_temp = return_pri
        #sigma_cache_temp = sigma_pri
        #weight_cache_temp = weight_pri
    else :
        pass

    idx_mc1=idx_mc1+1;

# In[]

print("the target portfolio'return is :")
print(return_pri)
print("its variance is:")
print(sigma_cache)
print("the stock name and weighted is:")
print(stockname_input)
print(weight_cache)





















