"""
YFinance JSON Functions - Usage Examples
Examples showing how to use each function to get specific financial data
"""

import json
from ytfinance_json_functions import (
    get_basic_info,
    get_historical_data,
    get_financial_statements,
    get_options_data,
    get_ownership_data,
    get_news_and_recommendations,
    get_multi_ticker_analysis,
    get_advanced_queries,
    get_pe_ratios,
    get_debt_analysis,
    get_debt_profitability_ratios
)

def example_basic_info():
    """Example of getting basic ticker information"""
    print("\n=== BASIC INFO EXAMPLE ===")
    
    # Get Apple's basic information
    apple_info = get_basic_info("AAPL")
    
    # Print company details
    print(f"Company: {apple_info['company_info']['name']}")
    print(f"Sector: {apple_info['company_info']['sector']}")
    print(f"Industry: {apple_info['company_info']['industry']}")
    
    # Print key market data
    print(f"Current Price: ${apple_info['market_data']['current_price']}")
    print(f"Market Cap: ${apple_info['market_data']['market_cap']:,}")
    
    # Print key metrics
    print(f"P/E Ratio: {apple_info['key_metrics']['pe_ratio']}")
    print(f"Dividend Yield: {apple_info['key_metrics']['dividend_yield']}")
    
    # Save full data to JSON file
    with open('apple_info.json', 'w') as f:
        json.dump(apple_info, f, indent=2)
    print("Full data saved to 'apple_info.json'")

def example_historical_data():
    """Example of getting historical price data"""
    print("\n=== HISTORICAL DATA EXAMPLE ===")
    
    # Get Microsoft's historical data for the past 3 months
    msft_history = get_historical_data("MSFT", period="3mo", interval="1d")
    
    # Print performance metrics
    print(f"Period: {msft_history['metrics']['start_date']} to {msft_history['metrics']['end_date']}")
    print(f"Start Price: ${msft_history['metrics']['start_price']:.2f}")
    print(f"End Price: ${msft_history['metrics']['end_price']:.2f}")
    print(f"Total Return: {msft_history['metrics']['total_return_pct']:.2f}%")
    print(f"Volatility: {msft_history['metrics']['return_volatility']:.2f}%")
    print(f"Max Drawdown: {msft_history['metrics']['max_drawdown']:.2f}%")
    
    # Print the 5 most recent daily prices
    print("\nRecent prices:")
    for day in msft_history['price_data'][-5:]:
        print(f"{day['date']}: ${day['close']:.2f} (Volume: {day['volume']:,})")
    
    # Save full data to JSON file
    with open('msft_history.json', 'w') as f:
        json.dump(msft_history, f, indent=2)
    print("Full data saved to 'msft_history.json'")

def example_financial_statements():
    """Example of getting financial statement data"""
    print("\n=== FINANCIAL STATEMENTS EXAMPLE ===")
    
    # Get Google's financial statements
    googl_financials = get_financial_statements("GOOGL")
    
    # Access annual income statement data
    if googl_financials.get('annual_financials', {}).get('income_statement'):
        # Get the dates (years) in the financial statements
        years = list(googl_financials['annual_financials']['income_statement'].keys())
        latest_year = years[0]  # Most recent year
        
        # Print revenue and net income for the most recent year
        income_stmt = googl_financials['annual_financials']['income_statement'][latest_year]
        print(f"Google Annual Financials for {latest_year}:")
        print(f"Total Revenue: ${income_stmt.get('Total Revenue', 0) / 1e9:.2f} billion")
        print(f"Net Income: ${income_stmt.get('Net Income', 0) / 1e9:.2f} billion")
        
        # Print some key balance sheet items
        if googl_financials.get('annual_financials', {}).get('balance_sheet'):
            balance = googl_financials['annual_financials']['balance_sheet'][latest_year]
            print(f"Total Assets: ${balance.get('Total Assets', 0) / 1e9:.2f} billion")
            print(f"Total Liabilities: ${balance.get('Total Liabilities', 0) / 1e9:.2f} billion")
            print(f"Stockholder Equity: ${balance.get('Total Stockholder Equity', 0) / 1e9:.2f} billion")
    else:
        print("No financial statement data available")
    
    # Save full data to JSON file
    with open('googl_financials.json', 'w') as f:
        json.dump(googl_financials, f, indent=2)
    print("Full data saved to 'googl_financials.json'")

def example_options_data():
    """Example of getting options data"""
    print("\n=== OPTIONS DATA EXAMPLE ===")
    
    # Get Tesla's options data
    tesla_options = get_options_data("TSLA")
    
    # Show available expiration dates
    print(f"Available Expiration Dates: {', '.join(tesla_options['expiration_dates'][:5])}...")
    
    # Get data for the first expiration date
    if tesla_options.get('options_chains') and tesla_options['expiration_dates']:
        exp_date = tesla_options['expiration_dates'][0]
        options_chain = tesla_options['options_chains'][exp_date]
        
        # Print information about at-the-money options
        print(f"\nAt-the-money options (Current Price: ${tesla_options['current_price']}):")
        if options_chain.get('analysis', {}).get('atm_calls'):
            atm_call = options_chain['analysis']['atm_calls'][0]
            print(f"ATM Call: Strike=${atm_call['strike']}, Price=${atm_call['last_price']}, IV={atm_call['implied_volatility']:.2f}")
        
        if options_chain.get('analysis', {}).get('atm_puts'):
            atm_put = options_chain['analysis']['atm_puts'][0]
            print(f"ATM Put: Strike=${atm_put['strike']}, Price=${atm_put['last_price']}, IV={atm_put['implied_volatility']:.2f}")
        
        # Print put/call ratio
        if options_chain.get('analysis', {}).get('put_call_ratio'):
            print(f"Put/Call Ratio: {options_chain['analysis']['put_call_ratio']:.2f}")
        
        # Print some stats about available options
        print(f"Available call options: {len(options_chain['calls'])}")
        print(f"Available put options: {len(options_chain['puts'])}")
    else:
        print("No options chain data available")
    
    # Save full data to JSON file
    with open('tesla_options.json', 'w') as f:
        json.dump(tesla_options, f, indent=2)
    print("Full data saved to 'tesla_options.json'")

def example_ownership_data():
    """Example of getting ownership data"""
    print("\n=== OWNERSHIP DATA EXAMPLE ===")
    
    # Get NVIDIA's ownership data
    nvda_ownership = get_ownership_data("NVDA")
    
    # Print major holders information
    print("Major Holders:")
    for holder in nvda_ownership.get('major_holders', []):
        print(f"{holder.get('value')} - {holder.get('description')}")
    
    # Print top institutional holders
    print("\nTop 3 Institutional Holders:")
    for holder in nvda_ownership.get('institutional_holders', [])[:3]:
        print(f"{holder.get('holder')}: {holder.get('shares'):,} shares ({holder.get('percent_out')}%)")
    
    # Print recent insider transactions
    print("\nRecent Insider Transactions:")
    for transaction in nvda_ownership.get('insider_transactions', [])[:3]:
        print(f"{transaction.get('date')} - {transaction.get('insider')} ({transaction.get('relation')}): {transaction.get('transaction')} {transaction.get('shares'):,} shares")
    
    # Save full data to JSON file
    with open('nvda_ownership.json', 'w') as f:
        json.dump(nvda_ownership, f, indent=2)
    print("Full data saved to 'nvda_ownership.json'")

def example_news_and_recommendations():
    """Example of getting news and analyst recommendations"""
    print("\n=== NEWS AND RECOMMENDATIONS EXAMPLE ===")
    
    # Get JP Morgan news and recommendations
    jpm_news = get_news_and_recommendations("JPM")
    
    # Print recent news
    print("Recent News:")
    for article in jpm_news.get('news', [])[:3]:
        print(f"{article.get('publish_time')} - {article.get('title')}")
        print(f"  Source: {article.get('publisher')}")
        print(f"  Link: {article.get('link')}\n")
    
    # Print recent analyst recommendations
    print("Recent Analyst Recommendations:")
    for rec in jpm_news.get('recommendations', [])[:3]:
        print(f"{rec.get('date')} - {rec.get('firm')}: {rec.get('from_grade')} → {rec.get('to_grade')}")
    
    # Save full data to JSON file
    with open('jpm_news.json', 'w') as f:
        json.dump(jpm_news, f, indent=2)
    print("Full data saved to 'jpm_news.json'")

def example_multi_ticker_analysis():
    """Example of analyzing multiple tickers"""
    print("\n=== MULTI-TICKER ANALYSIS EXAMPLE ===")
    
    # Compare big tech companies
    tickers = ["AAPL", "MSFT", "GOOGL", "AMZN", "META"]
    big_tech = get_multi_ticker_analysis(tickers)
    
    # Print market cap comparison
    print("Market Cap Comparison:")
    for ticker, data in big_tech.get('ticker_data', {}).items():
        mcap = data.get('market_cap')
        if mcap:
            print(f"{ticker} ({data.get('company_name')}): ${mcap/1e9:.2f} billion")
    
    # Print 30-day return comparison
    print("\n30-Day Performance:")
    for ticker, data in big_tech.get('performance', {}).get('returns_30d', {}).items():
        print(f"{ticker}: {data.get('total_return_30d', 0):.2f}%")
    
    # Print correlation matrix
    print("\nCorrelation Matrix:")
    corr_matrix = big_tech.get('performance', {}).get('correlation_matrix', {})
    if corr_matrix:
        # Print header
        header = "      " + "  ".join(f"{t:6s}" for t in tickers)
        print(header)
        
        # Print rows
        for ticker1 in tickers:
            row = f"{ticker1:6s}"
            for ticker2 in tickers:
                if ticker1 in corr_matrix and ticker2 in corr_matrix[ticker1]:
                    row += f" {corr_matrix[ticker1][ticker2]:6.2f}"
                else:
                    row += "      "
            print(row)
    
    # Save full data to JSON file
    with open('big_tech_comparison.json', 'w') as f:
        json.dump(big_tech, f, indent=2)
    print("Full data saved to 'big_tech_comparison.json'")

def example_advanced_queries():
    """Example of using advanced queries"""
    print("\n=== ADVANCED QUERIES EXAMPLE ===")
    
    # Get dividend history for Johnson & Johnson
    jnj_dividends = get_advanced_queries("JNJ", query_type="dividends")
    
    # Print dividend history and growth
    print("Johnson & Johnson Dividend History:")
    annual_dividends = jnj_dividends.get('annual_dividends', {})
    growth_rates = jnj_dividends.get('dividend_growth_rates', {})
    
    # Get last 5 years of data
    years = sorted(list(annual_dividends.keys()))[-5:]
    
    for year in years:
        growth = growth_rates.get(year, None)
        growth_str = f" ({'' if growth < 0 else ' '}{growth:.2f}%)" if growth else ""
        print(f"{year}: ${annual_dividends.get(year, 0):.4f}{growth_str}")
    
    avg_growth = jnj_dividends.get('avg_5yr_growth')
    if avg_growth:
        print(f"Average 5-Year Dividend Growth Rate: {avg_growth:.2f}%")
    
    # Get ETF holdings for a Technology ETF
    xlk_holdings = get_advanced_queries("XLK", query_type="etf_holdings")
    
    # Print top holdings
    print("\nXLK (Technology ETF) Top Holdings:")
    for holding in xlk_holdings.get('holdings', [])[:5]:
        print(f"{holding.get('symbol')} ({holding.get('company')}): {holding.get('percent')}%")
    
    # Get stock split history for Tesla
    tsla_splits = get_advanced_queries("TSLA", query_type="splits")
    
    # Print split history
    print("\nTesla Stock Split History:")
    for split in tsla_splits.get('split_data', []):
        print(f"{split.get('date')}: {split.get('ratio')}-for-1 split")
    
    # Get ESG/sustainability data for Microsoft
    msft_esg = get_advanced_queries("MSFT", query_type="sustainability")
    
    # Print ESG scores
    print("\nMicrosoft ESG/Sustainability Scores:")
    for category, score in msft_esg.get('esg_data', {}).items():
        print(f"{category}: {score}")
    
    # Save full data to JSON files
    with open('jnj_dividends.json', 'w') as f:
        json.dump(jnj_dividends, f, indent=2)
    with open('xlk_holdings.json', 'w') as f:
        json.dump(xlk_holdings, f, indent=2)
    print("Full data saved to JSON files")

def example_pe_ratios():
    """Example of getting P/E ratio analysis and peer comparison"""
    print("\n=== P/E RATIO ANALYSIS EXAMPLE ===")
    
    # Get Apple's P/E ratio analysis with peer comparison
    apple_pe = get_pe_ratios("AAPL", ["MSFT", "GOOGL", "META"])
    
    # Print main valuation metrics
    print(f"{apple_pe.get('company_name')} Valuation Metrics:")
    metrics = apple_pe.get('valuation_metrics', {})
    print(f"Current Price: ${metrics.get('current_price')}")
    print(f"Trailing P/E: {metrics.get('trailing_pe')}")
    print(f"Forward P/E: {metrics.get('forward_pe')}")
    print(f"PEG Ratio: {metrics.get('peg_ratio')}")
    print(f"Price-to-Book: {metrics.get('price_to_book')}")
    print(f"EPS (TTM): ${metrics.get('eps_ttm')}")
    
    # Print peer comparison
    print("\nPeer Comparison (P/E Ratios):")
    peers = apple_pe.get('peers', {})
    print(f"{'Company':<10} {'Trailing P/E':<15} {'Forward P/E':<15} {'PEG Ratio':<15}")
    print("-" * 55)
    
    # Print main company first
    peg_ratio = metrics.get('peg_ratio') or  'N/A'
    print(f"{apple_pe.get('symbol'):<10} {metrics.get('trailing_pe', 'N/A'):<15} {metrics.get('forward_pe', 'N/A'):<15} {peg_ratio:<15})")
    
    # Then print peers
    for symbol, data in peers.items():
        print(f"{symbol:<10} {data.get('trailing_pe', 'N/A'):<15} {data.get('forward_pe', 'N/A'):<15} {data.get('peg_ratio') or 'N/A':<15}")
    
    # Print industry averages
    print("\nIndustry Averages:")
    avgs = apple_pe.get('industry_averages', {})
    for metric, value in avgs.items():
        print(f"{metric.replace('_', ' ').title()}: {value:.2f}")
    
    # Save full data to JSON file
    with open('apple_pe_analysis.json', 'w') as f:
        json.dump(apple_pe, f, indent=2)
    print("Full data saved to 'apple_pe_analysis.json'")

def example_debt_analysis():
    """Example of getting company debt analysis"""
    print("\n=== DEBT ANALYSIS EXAMPLE ===")
    
    # Get debt analysis for a company with significant debt (using AT&T as example)
    ticker_symbol = "T"  # AT&T
    debt_data = get_debt_analysis(ticker_symbol, period_years=3)
    
    # Print company information
    print(f"Debt Analysis for {debt_data.get('company_name', ticker_symbol)}")
    
    # Print current debt metrics
    metrics = debt_data.get('current_debt_metrics', {})
    print("\nCurrent Debt Metrics:")
    
    # Format large numbers in billions
    total_debt = metrics.get('total_debt')
    if total_debt:
        print(f"Total Debt: ${total_debt/1e9:.2f} billion")
    
    cash = debt_data.get('current_debt_metrics', {}).get('cash_and_equivalents')
    if cash:
        print(f"Cash & Equivalents: ${cash/1e9:.2f} billion")
    
    net_debt = metrics.get('net_debt')
    if net_debt:
        print(f"Net Debt: ${net_debt/1e9:.2f} billion")
    
    # Print debt ratios
    print("\nDebt Ratios:")
    debt_to_equity = metrics.get('debt_to_equity')
    if debt_to_equity:
        print(f"Debt-to-Equity: {debt_to_equity:.2f}")
    
    interest_coverage = metrics.get('interest_coverage')
    if interest_coverage:
        print(f"Interest Coverage: {interest_coverage:.2f}x")
    
    debt_to_ebitda = metrics.get('total_debt_to_ebitda')
    if debt_to_ebitda:
        print(f"Total Debt to EBITDA: {debt_to_ebitda:.2f}x")
    
    # Print debt components
    components = debt_data.get('debt_components', {})
    print("\nDebt Components:")
    
    short_term = components.get('short_term_debt')
    if short_term:
        print(f"Short-term Debt: ${short_term/1e9:.2f} billion")
    
    long_term = components.get('long_term_debt')
    if long_term:
        print(f"Long-term Debt: ${long_term/1e9:.2f} billion")
    
    # Print historical debt trend
    annual_trend = debt_data.get('historical_debt', {}).get('annual_trend', [])
    if annual_trend:
        print("\nHistorical Debt Trend:")
        for period in annual_trend:
            total = period.get('total_debt')
            if total:
                print(f"{period.get('date')}: ${total/1e9:.2f} billion")
    
    # Print debt servicing capabilities
    servicing = debt_data.get('debt_servicing', {})
    if servicing:
        print("\nDebt Servicing Capabilities:")
        
        ocf = servicing.get('operating_cash_flow')
        if ocf:
            print(f"Operating Cash Flow: ${ocf/1e9:.2f} billion")
        
        fcf = servicing.get('free_cash_flow')
        if fcf:
            print(f"Free Cash Flow: ${fcf/1e9:.2f} billion")
        
        debt_to_fcf = servicing.get('debt_to_free_cash_flow')
        if debt_to_fcf:
            print(f"Debt to Free Cash Flow: {debt_to_fcf:.2f}x")
    
    # Save full data to JSON file
    with open(f'{ticker_symbol}_debt_analysis.json', 'w') as f:
        json.dump(debt_data, f, indent=2)
    print(f"Full data saved to '{ticker_symbol}_debt_analysis.json'")

def example_debt_profitability_ratios():
    """Example of getting debt-to-earnings and debt-to-profitability ratios"""
    print("\n=== DEBT-TO-PROFITABILITY ANALYSIS EXAMPLE ===")
    
    # Companies with different debt profiles
    companies = {
        "T": "AT&T (Telecom - High Debt)",
        "MSFT": "Microsoft (Tech - Low Debt)",
        "XOM": "Exxon Mobil (Energy)"
    }
    
    # Choose one company to analyze in detail
    ticker_symbol = "T"  # AT&T
    
    # Get detailed analysis for chosen company
    ratio_data = get_debt_profitability_ratios(ticker_symbol)
    
    # Print company info
    print(f"Debt-to-Profitability Analysis for {companies.get(ticker_symbol, ticker_symbol)}")
    
    # Print current profitability metrics
    print("\nProfitability Metrics (TTM):")
    profit_metrics = ratio_data.get('profitability_metrics', {})
    
    # Format large numbers in billions
    for metric in ['revenue', 'ebitda', 'ebit', 'net_income', 'free_cash_flow']:
        value = profit_metrics.get(metric)
        if value:
            print(f"{metric.replace('_', ' ').title()}: ${value/1e9:.2f} billion")
    
    # Print margin percentages
    print("\nMargin Percentages:")
    for metric in ['operating_margin', 'profit_margin', 'fcf_margin']:
        value = profit_metrics.get(metric)
        if value:
            print(f"{metric.replace('_', ' ').title()}: {value*100:.2f}%")
    
    # Print debt metrics
    print("\nDebt Position:")
    debt_metrics = ratio_data.get('debt_metrics', {})
    
    total_debt = debt_metrics.get('total_debt')
    if total_debt:
        print(f"Total Debt: ${total_debt/1e9:.2f} billion")
    
    net_debt = debt_metrics.get('net_debt')
    if net_debt:
        print(f"Net Debt: ${net_debt/1e9:.2f} billion")
    
    # Print key debt-to-profitability ratios
    print("\nDebt-to-Profitability Ratios:")
    ratios = ratio_data.get('current_ratios', {})
    
    important_ratios = [
        ('debt_to_ebitda', 'Debt to EBITDA'),
        ('debt_to_net_income', 'Debt to Net Income'),
        ('debt_to_free_cash_flow', 'Debt to Free Cash Flow'),
        ('debt_to_revenue', 'Debt to Revenue'),
        ('interest_coverage_ratio', 'Interest Coverage Ratio'),
        ('net_debt_to_ebitda', 'Net Debt to EBITDA')
    ]
    
    for key, label in important_ratios:
        value = ratios.get(key)
        if value:
            if 'coverage' in key:
                print(f"{label}: {value:.2f}x")
            else:
                print(f"{label}: {value:.2f}x")
    
    # Historical trend analysis
    print("\nHistorical Trend Analysis:")
    hist_data = ratio_data.get('historical_ratios', {}).get('annual', [])
    
    if hist_data:
        # Print debt to EBITDA trend
        print("\nDebt to EBITDA Ratio Trend:")
        for period in hist_data:
            if "ratios" in period and "debt_to_ebitda" in period["ratios"]:
                print(f"{period['date']}: {period['ratios']['debt_to_ebitda']:.2f}x")
        
        # Print interest coverage trend
        print("\nInterest Coverage Ratio Trend:")
        for period in hist_data:
            if "ratios" in period and "interest_coverage_ratio" in period["ratios"]:
                print(f"{period['date']}: {period['ratios']['interest_coverage_ratio']:.2f}x")
    
    # Compare with other companies
    print("\nComparison Across Companies:")
    print(f"{'Company':<10} {'Debt/EBITDA':<14} {'Int. Coverage':<14} {'Debt/FCF':<14}")
    print("-" * 55)
    
    for symbol, name in companies.items():
        # Get basic data for comparison
        company_data = get_debt_profitability_ratios(symbol)
        ratios = company_data.get('current_ratios', {})
        
        debt_to_ebitda = ratios.get('debt_to_ebitda', 'N/A')
        if debt_to_ebitda != 'N/A':
            debt_to_ebitda = f"{debt_to_ebitda:.2f}x"
        
        interest_coverage = ratios.get('interest_coverage_ratio', 'N/A')
        if interest_coverage != 'N/A':
            interest_coverage = f"{interest_coverage:.2f}x"
        
        debt_to_fcf = ratios.get('debt_to_free_cash_flow', 'N/A')
        if debt_to_fcf != 'N/A':
            debt_to_fcf = f"{debt_to_fcf:.2f}x"
        
        print(f"{symbol:<10} {debt_to_ebitda:<14} {interest_coverage:<14} {debt_to_fcf:<14}")
    
    # Save full data to JSON file
    with open(f'{ticker_symbol}_debt_profitability.json', 'w') as f:
        json.dump(ratio_data, f, indent=2)
    print(f"Full data saved to '{ticker_symbol}_debt_profitability.json'")

def run_all_examples():
    """Run all the example functions"""
    example_basic_info()
    example_historical_data()
    example_financial_statements()
    example_options_data()
    example_ownership_data()
    example_news_and_recommendations()
    example_pe_ratios()
    example_debt_analysis()
    example_debt_profitability_ratios()
    example_multi_ticker_analysis()
    example_advanced_queries()

if __name__ == "__main__":
    run_all_examples()
    # Or run a specific example:
    # example_basic_info()
