import yfinance as yf
import pandas as pd
import requests
import io
from concurrent.futures import ThreadPoolExecutor, as_completed

def get_top_companies_by_sector(sector_or_industry, 
                               num_companies=10, 
                               criteria="market_cap",
                               ascending=False,
                               max_tickers_to_process=200):
    """
    Returns a list of top companies in a given sector or industry based on specified criteria.
    
    Args:
        sector_or_industry (str): The sector or industry to filter companies by
        num_companies (int): Number of top companies to return
        criteria (str): Criteria for ranking. Options include:
                       "market_cap", "pe_ratio", "forward_pe", "price_to_book",
                       "debt_to_equity", "return_on_equity", "dividend_yield",
                       "profit_margin", "operating_margin", "beta", "peg_ratio",
                       "free_cash_flow", "debt_to_ebitda", "revenue_growth"
        ascending (bool): Sort order - True for ascending (smaller values first),
                         False for descending (larger values first)
        max_tickers_to_process (int): Maximum number of tickers to process to avoid rate limits
        
    Returns:
        pandas.DataFrame: DataFrame with the top companies and their data
    """
    
    # Step 1: Define valid criteria mappings (criteria name -> info dict key)
    criteria_mapping = {
        "market_cap": {"key": "marketCap", "ascending": False},
        "pe_ratio": {"key": "trailingPE", "ascending": True},
        "forward_pe": {"key": "forwardPE", "ascending": True},
        "price_to_book": {"key": "priceToBook", "ascending": True},
        "debt_to_equity": {"key": "debtToEquity", "ascending": True},
        "return_on_equity": {"key": "returnOnEquity", "ascending": False},
        "dividend_yield": {"key": "dividendYield", "ascending": False},
        "profit_margin": {"key": "profitMargin", "ascending": False},
        "operating_margin": {"key": "operatingMargins", "ascending": False},
        "beta": {"key": "beta", "ascending": True},
        "peg_ratio": {"key": "pegRatio", "ascending": True},
        "free_cash_flow": {"key": "freeCashflow", "ascending": False},
        "revenue_growth": {"key": "revenueGrowth", "ascending": False},
        "ebitda_margins": {"key": "ebitdaMargins", "ascending": False},
        "debt_to_ebitda": {"key": "calculated_ratio", "ascending": True},  # Will be calculated
        "lowest_debt": {"key": "totalDebt", "ascending": True}
    }
    
    # Validate criteria
    if criteria not in criteria_mapping:
        valid_criteria = list(criteria_mapping.keys())
        raise ValueError(f"Invalid criteria '{criteria}'. Valid options are: {valid_criteria}")
    
    # Step 2: Get list of tickers by sector/industry
    tickers = get_tickers_by_sector(sector_or_industry)
    
    if not tickers:
        print(f"No tickers found for sector/industry: {sector_or_industry}")
        print("Available sectors include: Technology, Financial, Healthcare, Consumer Cyclical, etc.")
        return pd.DataFrame()
    
    print(f"Found {len(tickers)} companies in {sector_or_industry}. Processing...")
    
    # Limit number of tickers to process to avoid rate limits
    tickers = tickers[:max_tickers_to_process]
    
    # Step 3: Get data for each ticker in parallel
    companies_data = get_companies_data(tickers, criteria_mapping[criteria]["key"])
    
    # Step 4: Filter out entries with missing data for the criteria
    if criteria == "debt_to_ebitda":
        # Special handling for calculated ratio
        filtered_data = [
            company for company in companies_data 
            if company.get("totalDebt") and company.get("ebitda") and company.get("ebitda") != 0
        ]
        
        # Calculate debt to EBITDA ratio
        for company in filtered_data:
            company["debt_to_ebitda"] = company["totalDebt"] / company["ebitda"]
        
        # Sort by the calculated ratio
        sort_key = "debt_to_ebitda"
    else:
        filtered_data = [
            company for company in companies_data 
            if company.get(criteria_mapping[criteria]["key"]) is not None
        ]
        
        # Use the key from criteria mapping
        sort_key = criteria_mapping[criteria]["key"]
    
    # Step 5: Sort by criteria 
    # If ascending wasn't specified, use the default for this criteria
    if ascending is None:
        use_ascending = criteria_mapping[criteria]["ascending"]
    else:
        use_ascending = ascending
        
    sorted_data = sorted(filtered_data, key=lambda x: x.get(sort_key, 0), reverse=not use_ascending)
    
    # Step 6: Return the top N companies
    top_companies = sorted_data[:num_companies]
    
    # Convert to DataFrame for better display
    df_data = []
    for company in top_companies:
        row = {
            "Symbol": company.get("symbol"),
            "Name": company.get("shortName"),
            "Sector": company.get("sector"),
            "Industry": company.get("industry"),
        }
        
        # Add the criteria column
        criteria_display = criteria.replace("_", " ").title()
        
        if criteria == "market_cap":
            # Format market cap in billions
            value = company.get(sort_key)
            row[criteria_display] = f"${value/1e9:.2f}B" if value else "N/A"
        elif criteria in ["dividend_yield", "profit_margin", "operating_margin", "return_on_equity", "revenue_growth"]:
            # Format percentages
            value = company.get(sort_key)
            row[criteria_display] = f"{value*100:.2f}%" if value else "N/A"
        else:
            # Standard format for other metrics
            row[criteria_display] = company.get(sort_key)
        
        # Add additional key metrics
        row["Market Cap"] = f"${company.get('marketCap')/1e9:.2f}B" if company.get('marketCap') else "N/A"
        row["P/E"] = company.get('trailingPE')
        
        if criteria != "debt_to_equity":
            row["Debt/Equity"] = company.get('debtToEquity')
        
        df_data.append(row)
    
    # Create DataFrame
    result_df = pd.DataFrame(df_data)
    
    # Add a note about the criteria used
    print(f"\nTop {num_companies} companies in {sector_or_industry} by {criteria.replace('_', ' ')}:")
    print(f"Sort order: {'Ascending' if use_ascending else 'Descending'} ({'smaller' if use_ascending else 'larger'} values first)")
    
    return result_df


def get_tickers_by_sector(sector_or_industry):
    """
    Get list of tickers that belong to a specific sector or industry.
    This function downloads current listings from NYSE, NASDAQ, and AMEX.
    
    Args:
        sector_or_industry (str): Sector or industry name to filter by
    
    Returns:
        list: List of ticker symbols in the specified sector/industry
    """
    # We'll download company listings from various sources
    # First, try to get a comprehensive listing
    try:
        # Try to download from nasdaq.com
        nasdaq_url = "https://www.nasdaq.com/screening/companies-by-name.aspx?letter=0&exchange=nasdaq&render=download"
        nyse_url = "https://www.nasdaq.com/screening/companies-by-name.aspx?letter=0&exchange=nyse&render=download"
        amex_url = "https://www.nasdaq.com/screening/companies-by-name.aspx?letter=0&exchange=amex&render=download"
        
        # This might fail if NASDAQ changes their URL structure
        nasdaq_data = pd.read_csv(nasdaq_url)
        nyse_data = pd.read_csv(nyse_url)
        amex_data = pd.read_csv(amex_url)
        
        listings = pd.concat([nasdaq_data, nyse_data, amex_data])
        listings_available = True
    except:
        listings_available = False
    
    # If the NASDAQ download fails, try another source or use a fallback list
    if not listings_available:
        try:
            # Try to get S&P 500 components as a fallback
            sp500_url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
            sp500 = pd.read_html(sp500_url)[0]
            sp500.columns = [col.replace(" ", "_").lower() for col in sp500.columns]
            
            # Look for sector/industry in the S&P 500 listing
            search_term = sector_or_industry.lower()
            filtered_sp500 = sp500[
                sp500['gics_sector'].str.lower().str.contains(search_term) | 
                sp500['gics_sub-industry'].str.lower().str.contains(search_term)
            ]
            
            # Return the tickers
            if not filtered_sp500.empty:
                return filtered_sp500['symbol'].tolist()
            else:
                # If no matches, get sector info for each ticker instead
                return use_predefined_ticker_list(sector_or_industry)
        except:
            # If all else fails, use a predefined list
            return use_predefined_ticker_list(sector_or_industry)
    else:
        # If listings are available, filter by sector/industry
        # Since we don't have sector info in the downloaded list,
        # we'll need to get that info for each ticker
        return use_ticker_list_with_filtering(listings['Symbol'].tolist(), sector_or_industry)


def use_predefined_ticker_list(sector_or_industry):
    """
    Use a predefined list of tickers and filter by sector/industry.
    This is a fallback when we can't download ticker lists.
    
    Args:
        sector_or_industry (str): Sector or industry to filter by
    
    Returns:
        list: List of tickers in the sector/industry
    """
    # Define major indices to get a reasonable set of tickers
    major_indices = [
        "^GSPC",  # S&P 500
        "^DJI",   # Dow Jones
        "^IXIC",  # NASDAQ
        "^RUT"    # Russell 2000
    ]
    
    # Get all components of each index
    all_tickers = []
    for index in major_indices:
        try:
            idx = yf.Ticker(index)
            components = idx.components if hasattr(idx, 'components') else []
            all_tickers.extend(components or [])
        except:
            pass
    
    # If we couldn't get index components, use a hardcoded list of large caps
    if not all_tickers:
        # A selection of major tickers across sectors
        all_tickers = [
            # Technology
            "AAPL", "MSFT", "GOOGL", "AMZN", "META", "NVDA", "TSLA", "AMD", "INTC", "IBM", "ORCL", "CRM", "ADBE", "CSCO",
            # Financial
            "JPM", "BAC", "WFC", "C", "GS", "MS", "V", "MA", "AXP", "BLK", "BRK-B", "PNC", "USB", "TFC",
            # Healthcare
            "JNJ", "PFE", "MRK", "ABT", "UNH", "ABBV", "LLY", "BMY", "TMO", "AMGN", "MDT", "GILD", "ISRG", "CVS",
            # Consumer Cyclical 
            "HD", "WMT", "NKE", "SBUX", "MCD", "DIS", "COST", "LOW", "TGT", "BKNG", "F", "GM", "MAR", "CMG",
            # Consumer Defensive
            "PG", "KO", "PEP", "PM", "MO", "CL", "GIS", "K", "SYY", "KHC", "EL", "CLX", "KMB", "HSY",
            # Energy
            "XOM", "CVX", "COP", "EOG", "SLB", "PSX", "VLO", "OXY", "MPC", "KMI", "WMB", "HES", "DVN", "PXD",
            # Industrials
            "BA", "GE", "HON", "UNP", "UPS", "CAT", "LMT", "RTX", "DE", "FDX", "MMM", "CSX", "ETN", "EMR",
            # Utilities
            "NEE", "DUK", "SO", "D", "AEP", "EXC", "SRE", "XEL", "ED", "ES", "WEC", "PCG", "FE", "AEE",
            # Real Estate
            "AMT", "PLD", "CCI", "EQIX", "PSA", "SPG", "O", "WELL", "AVB", "EQR", "DLR", "SBAC", "VTR", "BXP",
            # Basic Materials
            "LIN", "APD", "ECL", "SHW", "DD", "FCX", "NEM", "NUE", "CTVA", "DOW", "PPG", "VMC", "MLM", "ALB",
            # Communication Services
            "T", "VZ", "CMCSA", "NFLX", "CHTR", "TMUS", "ATVI", "EA", "TTWO", "OMC", "IPG", "DISH", "LYV", "LUMN"
        ]
    
    # Remove duplicates
    all_tickers = list(set(all_tickers))
    
    # Filter by sector/industry
    return filter_tickers_by_sector(all_tickers, sector_or_industry)


def use_ticker_list_with_filtering(ticker_list, sector_or_industry):
    """
    Filter a list of tickers by sector/industry.
    
    Args:
        ticker_list (list): List of ticker symbols
        sector_or_industry (str): Sector or industry to filter by
    
    Returns:
        list: Filtered list of tickers
    """
    return filter_tickers_by_sector(ticker_list, sector_or_industry)


def filter_tickers_by_sector(ticker_list, sector_or_industry):
    """
    Filter a list of tickers by sector/industry.
    
    Args:
        ticker_list (list): List of ticker symbols
        sector_or_industry (str): Sector or industry to filter by
    
    Returns:
        list: Filtered list of tickers
    """
    # Sample a subset of tickers to determine which ones are in the specified sector
    # This is to avoid hitting API rate limits
    sample_size = min(100, len(ticker_list))
    sample_tickers = ticker_list[:sample_size]
    
    # Get info for the sample
    sector_tickers = []
    search_term = sector_or_industry.lower()
    
    # Use threading to speed up the process
    with ThreadPoolExecutor(max_workers=10) as executor:
        future_to_ticker = {
            executor.submit(get_ticker_info, ticker): ticker for ticker in sample_tickers
        }
        
        for future in as_completed(future_to_ticker):
            ticker = future_to_ticker[future]
            try:
                info = future.result()
                if info:
                    ticker_sector = info.get('sector', '').lower()
                    ticker_industry = info.get('industry', '').lower()
                    
                    if (search_term in ticker_sector) or (search_term in ticker_industry):
                        sector_tickers.append(ticker)
            except Exception as e:
                print(f"Error processing {ticker}: {e}")
                
    return sector_tickers


def get_ticker_info(ticker):
    """
    Get info for a ticker.
    
    Args:
        ticker (str): Ticker symbol
    
    Returns:
        dict: Ticker info
    """
    try:
        t = yf.Ticker(ticker)
        return t.info
    except:
        return None


def get_companies_data(tickers, criteria_key):
    """
    Get data for a list of tickers, focused on the criteria needed.
    
    Args:
        tickers (list): List of ticker symbols
        criteria_key (str): Key for the sorting criteria
    
    Returns:
        list: List of company data dicts
    """
    companies_data = []
    
    # Use threading to speed up the process
    with ThreadPoolExecutor(max_workers=10) as executor:
        future_to_ticker = {
            executor.submit(get_extended_ticker_info, ticker, criteria_key): ticker for ticker in tickers
        }
        
        for future in as_completed(future_to_ticker):
            ticker = future_to_ticker[future]
            try:
                info = future.result()
                if info:
                    info['symbol'] = ticker  # Make sure symbol is included
                    companies_data.append(info)
            except Exception as e:
                print(f"Error processing {ticker}: {e}")
                
    return companies_data


def get_extended_ticker_info(ticker, criteria_key):
    """
    Get extended info for a ticker, including any calculated fields.
    
    Args:
        ticker (str): Ticker symbol
        criteria_key (str): Key for the sorting criteria
    
    Returns:
        dict: Extended ticker info
    """
    try:
        t = yf.Ticker(ticker)
        info = t.info
        
        # If we need debt-to-EBITDA ratio data, get additional fields
        if criteria_key == "calculated_ratio":
            # Make sure we have totalDebt and EBITDA for the calculation
            if 'totalDebt' not in info or info['totalDebt'] is None:
                # Try to get from financial statements
                try:
                    balance_sheet = t.balance_sheet
                    if not balance_sheet.empty:
                        for index in balance_sheet.index:
                            if "total debt" in str(index).lower():
                                info['totalDebt'] = balance_sheet.loc[index].iloc[0]
                                break
                except:
                    pass
            
            if 'ebitda' not in info or info['ebitda'] is None:
                # Try to get from financial statements
                try:
                    income = t.income_stmt
                    if not income.empty:
                        for index in income.index:
                            if "ebitda" in str(index).lower():
                                info['ebitda'] = income.loc[index].iloc[0]
                                break
                except:
                    pass
        
        return info
    except:
        return None


# Example usage
if __name__ == "__main__":
    # Example 1: Get top technology companies by market cap
    top_tech = get_top_companies_by_sector("Technology", num_companies=10, criteria="market_cap")
    print(top_tech)
    print("\n" + "="*80 + "\n")
    
    # Example 2: Get energy companies with lowest debt-to-equity ratio
    low_debt_energy = get_top_companies_by_sector("Energy", num_companies=5, criteria="debt_to_equity", ascending=True)
    print(low_debt_energy)
    print("\n" + "="*80 + "\n")
    
    # Example 3: Get financial companies with highest return on equity
    high_roe_finance = get_top_companies_by_sector("Financial", num_companies=5, criteria="return_on_equity")
    print(high_roe_finance)
