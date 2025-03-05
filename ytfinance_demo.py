"""
YFinance Functionality Demo
This script demonstrates the main features and capabilities of the yfinance library
"""

import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

def main():
    print("YFinance Functionality Demo")
    print("===========================\n")
    
    # 1. Basic ticker information
    demo_basic_info()
    
    # 2. Historical data
    demo_historical_data()
    
    # 3. Financial statements
    demo_financials()
    
    # 4. Options data
    demo_options()
    
    # 5. Holders and institutional ownership
    demo_ownership()
    
    # 6. News and analysis
    demo_news()
    
    # 7. Multi-ticker analysis
    demo_multi_ticker()
    
    # 8. Advanced queries
    demo_advanced_queries()

def demo_basic_info():
    """Demonstrate basic ticker information retrieval"""
    print("\n1. BASIC TICKER INFORMATION")
    print("--------------------------")
    
    # Get ticker object
    ticker = yf.Ticker("AAPL")
    
    # Basic info as dictionary
    info = ticker.info
    print(f"Company: {info.get('shortName', 'N/A')}")
    print(f"Sector: {info.get('sector', 'N/A')}")
    print(f"Industry: {info.get('industry', 'N/A')}")
    print(f"Current Price: ${info.get('currentPrice', 'N/A')}")
    print(f"Market Cap: ${info.get('marketCap', 'N/A'):,}")
    print(f"52-Week Range: ${info.get('fiftyTwoWeekLow', 'N/A')} - ${info.get('fiftyTwoWeekHigh', 'N/A')}")
    
    # Company description
    print("\nBusiness Summary:")
    print(f"{info.get('longBusinessSummary', 'N/A')[:300]}...\n")
    
    # Key statistics
    print("Key Statistics:")
    print(f"P/E Ratio: {info.get('trailingPE', 'N/A')}")
    print(f"EPS (TTM): ${info.get('trailingEps', 'N/A')}")
    print(f"Dividend Yield: {info.get('dividendYield', 'N/A') * 100 if info.get('dividendYield') else 'N/A'}%")
    print(f"Beta: {info.get('beta', 'N/A')}")

def demo_historical_data():
    """Demonstrate historical data retrieval and analysis"""
    print("\n2. HISTORICAL DATA")
    print("----------------")
    
    # Get ticker
    ticker = yf.Ticker("MSFT")
    
    # Get 1 year of daily historical data
    end_date = datetime.now()
    start_date = end_date - timedelta(days=365)
    hist = ticker.history(start=start_date, end=end_date)
    
    # Display first few rows
    print("\nRecent price history:")
    print(hist.tail().to_string())
    
    # Calculate some technical indicators
    hist['MA20'] = hist['Close'].rolling(window=20).mean()
    hist['MA50'] = hist['Close'].rolling(window=50).mean()
    hist['Daily_Return'] = hist['Close'].pct_change() * 100
    
    # Summary statistics
    print("\nSummary Statistics:")
    print(f"Average Daily Volume: {hist['Volume'].mean():,.0f} shares")
    print(f"Average Daily Return: {hist['Daily_Return'].mean():.2f}%")
    print(f"Return Volatility: {hist['Daily_Return'].std():.2f}%")
    print(f"Price Change (1yr): {((hist['Close'].iloc[-1] / hist['Close'].iloc[0]) - 1) * 100:.2f}%")
    
    # Calculate Max Drawdown
    cumulative_returns = (1 + hist['Daily_Return']/100).cumprod()
    max_return = cumulative_returns.cummax()
    drawdown = ((cumulative_returns / max_return) - 1) * 100
    print(f"Maximum Drawdown: {drawdown.min():.2f}%")

def demo_financials():
    """Demonstrate financial statement data retrieval"""
    print("\n3. FINANCIAL STATEMENTS")
    print("---------------------")
    
    ticker = yf.Ticker("GOOGL")
    
    # Income Statement
    print("\nIncome Statement (Annual, in millions USD):")
    income_stmt = ticker.income_stmt
    if not income_stmt.empty:
        display_financials(income_stmt / 1e6, ['Total Revenue', 'Cost Of Revenue', 
                                             'Gross Profit', 'Research And Development',
                                             'Net Income'])
    
    # Balance Sheet
    print("\nBalance Sheet (Annual, in millions USD):")
    balance_sheet = ticker.balance_sheet
    if not balance_sheet.empty:
        display_financials(balance_sheet / 1e6, ['Total Assets', 'Total Liabilities',
                                              'Total Stockholder Equity', 'Cash And Cash Equivalents',
                                              'Short Long Term Debt'])
    
    # Cash Flow
    print("\nCash Flow Statement (Annual, in millions USD):")
    cash_flow = ticker.cashflow
    if not cash_flow.empty:
        display_financials(cash_flow / 1e6, ['Operating Cash Flow', 'Capital Expenditure',
                                          'Free Cash Flow', 'Dividend Paid', 
                                          'Stock Repurchase'])
    
    # Quarterly data
    print("\nQuarterly Revenue (in millions USD):")
    quarterly = ticker.quarterly_income_stmt
    if not quarterly.empty:
        display_financials(quarterly.loc['Total Revenue'] / 1e6, None, transpose=True)

def display_financials(df, items=None, transpose=False):
    """Helper function to display financial data"""
    if transpose:
        display_df = df
    elif items:
        display_df = df.loc[items]
    else:
        display_df = df
        
    # Format DataFrame for display
    pd.options.display.float_format = '${:,.2f}'.format
    display_sample = display_df.iloc[:, :4]  # Show last 4 periods
    print(display_sample.to_string())
    pd.reset_option('display.float_format')

def demo_options():
    """Demonstrate options data retrieval"""
    print("\n4. OPTIONS DATA")
    print("-------------")
    
    ticker = yf.Ticker("TSLA")
    
    # Get available expiration dates
    expirations = ticker.options
    if not expirations:
        print("No options data available")
        return
        
    print(f"\nAvailable option expiration dates: {', '.join(expirations[:5])}...")
    
    # Get the closest expiration
    exp = expirations[0]
    
    # Calls
    calls = ticker.option_chain(exp).calls
    print(f"\nCall Options (Expiring {exp}):")
    if not calls.empty:
        print(calls[['strike', 'lastPrice', 'bid', 'ask', 'volume', 'impliedVolatility']].head().to_string())
    
    # Puts
    puts = ticker.option_chain(exp).puts
    print(f"\nPut Options (Expiring {exp}):")
    if not puts.empty:
        print(puts[['strike', 'lastPrice', 'bid', 'ask', 'volume', 'impliedVolatility']].head().to_string())
    
    # Options analytics
    current_price = ticker.info.get('currentPrice', 0)
    if current_price > 0:
        print("\nOptions Analysis:")
        
        # Find ATM calls (closest to current price)
        atm_calls = calls.iloc[(calls['strike'] - current_price).abs().argsort()[:3]]
        print(f"\nAt-the-money Calls (Current Price: ${current_price}):")
        print(atm_calls[['strike', 'lastPrice', 'impliedVolatility']].to_string())
        
        # Calculate put/call ratio based on open interest
        if not (calls.empty or puts.empty):
            total_call_oi = calls['openInterest'].sum()
            total_put_oi = puts['openInterest'].sum()
            if total_call_oi > 0:
                put_call_ratio = total_put_oi / total_call_oi
                print(f"\nPut/Call Ratio (Open Interest): {put_call_ratio:.2f}")

def demo_ownership():
    """Demonstrate institutional ownership data retrieval"""
    print("\n5. OWNERSHIP DATA")
    print("---------------")
    
    ticker = yf.Ticker("NVDA")
    
    # Major holders (percentage)
    major_holders = ticker.major_holders
    if not major_holders.empty:
        print("\nMajor Holders:")
        print(major_holders.to_string(header=False))
    
    # Institutional holders
    inst_holders = ticker.institutional_holders
    if not inst_holders.empty:
        print("\nTop Institutional Holders:")
        print(inst_holders.head().to_string())
    
    # Mutual fund holders
    mutualfund_holders = ticker.mutualfund_holders
    if not mutualfund_holders.empty:
        print("\nTop Mutual Fund Holders:")
        print(mutualfund_holders.head().to_string())
    
    # Insider transactions
    insider = ticker.insider_transactions
    if not insider.empty:
        print("\nRecent Insider Transactions:")
        print(insider.head().to_string())

def demo_news():
    """Demonstrate news retrieval"""
    print("\n6. NEWS AND RECOMMENDATIONS")
    print("-------------------------")
    
    ticker = yf.Ticker("JPM")
    
    # Get recent news
    news = ticker.news
    if news:
        print("\nRecent News:")
        for i, item in enumerate(news[:5], 1):
            print(f"{i}. {item['title']}")
            print(f"   Published: {datetime.fromtimestamp(item['providerPublishTime'])}")
            print(f"   Link: {item['link']}\n")
    
    # Analyst recommendations
    recommendations = ticker.recommendations
    if not recommendations.empty:
        print("Recent Analyst Recommendations:")
        recent_recom = recommendations.tail(5)
        print(recent_recom.to_string())
        
        # Count recommendations by grade
        if not recommendations.empty:
            grade_counts = recommendations['To Grade'].value_counts()
            print("\nCurrent Recommendation Distribution:")
            print(grade_counts.to_string())

def demo_multi_ticker():
    """Demonstrate functionality for analyzing multiple tickers"""
    print("\n7. MULTI-TICKER ANALYSIS")
    print("----------------------")
    
    # Define list of tickers
    tickers = ["AAPL", "MSFT", "GOOGL", "AMZN", "META"]
    
    # Method 1: Using Ticker objects in a loop
    print("\nMethod 1: Individual Ticker Objects")
    market_caps = {}
    pe_ratios = {}
    
    for symbol in tickers:
        ticker = yf.Ticker(symbol)
        info = ticker.info
        market_caps[symbol] = info.get('marketCap', None)
        pe_ratios[symbol] = info.get('trailingPE', None)
    
    # Display results
    print("\nMarket Capitalization Comparison:")
    for symbol, mcap in market_caps.items():
        if mcap:
            print(f"{symbol}: ${mcap:,}")
    
    # Method 2: Using download function (more efficient)
    print("\nMethod 2: Bulk Download (More Efficient)")
    end_date = datetime.now()
    start_date = end_date - timedelta(days=30)
    
    # Download price data for all tickers at once
    data = yf.download(tickers, start=start_date, end=end_date)
    
    # Calculate returns
    returns = data['Adj Close'].pct_change().dropna()
    
    print("\n30-Day Returns:")
    for symbol in tickers:
        total_return = ((returns[symbol] + 1).cumprod().iloc[-1] - 1) * 100
        print(f"{symbol}: {total_return:.2f}%")
    
    # Correlation matrix
    print("\nPrice Correlation Matrix:")
    corr_matrix = returns.corr()
    print(corr_matrix.round(2).to_string())

def demo_advanced_queries():
    """Demonstrate advanced querying capabilities"""
    print("\n8. ADVANCED QUERIES")
    print("----------------")
    
    # Get data for specific ETF
    print("\nSector ETF Analysis - XLK (Technology):")
    xlk = yf.Ticker("XLK")
    
    # Get holdings for ETF
    holdings = xlk.holdings
    if not holdings.empty:
        print("\nTop 10 Holdings:")
        print(holdings.head(10).to_string())
    
    # Dividend history
    print("\nDividend History Analysis - JNJ:")
    jnj = yf.Ticker("JNJ")
    dividends = jnj.dividends
    
    if not dividends.empty:
        annual_div = dividends.resample('Y').sum()
        print("\nAnnual Dividend Amounts:")
        print(annual_div.tail(5).to_string())
        
        # Calculate dividend growth
        div_growth = annual_div.pct_change() * 100
        print("\nAnnual Dividend Growth Rate:")
        print(div_growth.tail(5).to_string())
        
        # Calculate average growth rate
        avg_growth = div_growth.tail(5).mean()
        print(f"\nAverage 5-Year Dividend Growth Rate: {avg_growth:.2f}%")
    
    # Stock splits
    print("\nStock Split History - TSLA:")
    tsla = yf.Ticker("TSLA")
    splits = tsla.splits
    
    if not splits.empty:
        print("\nHistorical Stock Splits:")
        for date, ratio in splits.items():
            print(f"{date.date()}: {ratio:.4f}-for-1 split")
    
    # Sustainability scores
    print("\nSustainability/ESG Scores - MSFT:")
    msft = yf.Ticker("MSFT")
    sustainability = msft.sustainability
    
    if not sustainability.empty:
        print(sustainability.to_string())

if __name__ == "__main__":
    main()
