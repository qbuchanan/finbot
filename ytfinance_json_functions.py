"""
YFinance JSON Functions
Each function returns a specific set of data from yfinance as a JSON-serializable dictionary
"""

import yfinance as yf
import pandas as pd
import json
from datetime import datetime, timedelta

def get_basic_info(ticker_symbol="AAPL"):
    """
    Get basic information about a ticker
    
    Args:
        ticker_symbol (str): Stock ticker symbol
        
    Returns:
        dict: JSON-serializable dictionary with basic company information
    """
    try:
        ticker = yf.Ticker(ticker_symbol)
        info = ticker.info
        
        # Extract key data
        result = {
            "symbol": ticker_symbol,
            "company_info": {
                "name": info.get("shortName"),
                "sector": info.get("sector"),
                "industry": info.get("industry"),
                "website": info.get("website"),
                "business_summary": info.get("longBusinessSummary")
            },
            "market_data": {
                "current_price": info.get("currentPrice"),
                "previous_close": info.get("previousClose"),
                "open": info.get("open"),
                "day_low": info.get("dayLow"),
                "day_high": info.get("dayHigh"),
                "market_cap": info.get("marketCap"),
                "volume": info.get("volume"),
                "avg_volume": info.get("averageVolume")
            },
            "trading_range": {
                "fifty_two_week_low": info.get("fiftyTwoWeekLow"),
                "fifty_two_week_high": info.get("fiftyTwoWeekHigh")
            },
            "key_metrics": {
                "pe_ratio": info.get("trailingPE"),
                "forward_pe": info.get("forwardPE"),
                "eps_ttm": info.get("trailingEps"),
                "peg_ratio": info.get("pegRatio"),
                "price_to_book": info.get("priceToBook"),
                "beta": info.get("beta"),
                "dividend_rate": info.get("dividendRate"),
                "dividend_yield": info.get("dividendYield"),
                "five_year_avg_dividend_yield": info.get("fiveYearAvgDividendYield"),
                "payout_ratio": info.get("payoutRatio")
            }
        }
        return result
    except Exception as e:
        return {"error": str(e), "symbol": ticker_symbol}

def get_historical_data(ticker_symbol="MSFT", period="1y", interval="1d"):
    """
    Get historical price data and calculated metrics
    
    Args:
        ticker_symbol (str): Stock ticker symbol
        period (str): Time period (1d, 5d, 1mo, 3mo, 6mo, 1y, 2y, 5y, 10y, ytd, max)
        interval (str): Data interval (1m, 2m, 5m, 15m, 30m, 60m, 90m, 1h, 1d, 5d, 1wk, 1mo, 3mo)
        
    Returns:
        dict: JSON-serializable dictionary with historical data and metrics
    """
    try:
        ticker = yf.Ticker(ticker_symbol)
        hist = ticker.history(period=period, interval=interval)
        
        # Calculate metrics
        if not hist.empty and len(hist) > 20:
            hist['MA20'] = hist['Close'].rolling(window=20).mean()
            hist['MA50'] = hist['Close'].rolling(window=50).mean()
            hist['Daily_Return'] = hist['Close'].pct_change() * 100
            
            # Calculate drawdown
            cumulative_returns = (1 + hist['Daily_Return']/100).cumprod()
            max_return = cumulative_returns.cummax()
            drawdown = ((cumulative_returns / max_return) - 1) * 100
            
            # Convert timestamps to string format for JSON serialization
            historical_prices = []
            for date, row in hist.iterrows():
                historical_prices.append({
                    "date": date.strftime('%Y-%m-%d'),
                    "open": row['Open'],
                    "high": row['High'],
                    "low": row['Low'],
                    "close": row['Close'],
                    "volume": row['Volume'],
                    "ma20": row['MA20'],
                    "ma50": row['MA50'],
                    "daily_return": row['Daily_Return']
                })
            
            # Calculate summary metrics
            total_return = ((hist['Close'].iloc[-1] / hist['Close'].iloc[0]) - 1) * 100
            
            result = {
                "symbol": ticker_symbol,
                "period": period,
                "interval": interval,
                "price_data": historical_prices,
                "metrics": {
                    "start_date": hist.index[0].strftime('%Y-%m-%d'),
                    "end_date": hist.index[-1].strftime('%Y-%m-%d'),
                    "start_price": hist['Close'].iloc[0],
                    "end_price": hist['Close'].iloc[-1],
                    "total_return_pct": total_return,
                    "avg_daily_return": hist['Daily_Return'].mean(),
                    "return_volatility": hist['Daily_Return'].std(),
                    "max_drawdown": drawdown.min(),
                    "avg_volume": hist['Volume'].mean()
                }
            }
        else:
            result = {
                "symbol": ticker_symbol,
                "period": period,
                "interval": interval,
                "price_data": [],
                "error": "Insufficient data"
            }
            
        return result
    except Exception as e:
        return {"error": str(e), "symbol": ticker_symbol}

def get_financial_statements(ticker_symbol="GOOGL"):
    """
    Get financial statement data (income statement, balance sheet, cash flow)
    
    Args:
        ticker_symbol (str): Stock ticker symbol
        
    Returns:
        dict: JSON-serializable dictionary with financial statement data
    """
    try:
        ticker = yf.Ticker(ticker_symbol)
        
        # Helper function to convert financial dataframes to JSON-compatible format
        def df_to_dict(df):
            if df.empty:
                return {}
            
            result = {}
            for col in df.columns:
                col_date = col.strftime('%Y-%m-%d') if hasattr(col, 'strftime') else str(col)
                result[col_date] = {}
                for idx in df.index:
                    result[col_date][idx] = df.loc[idx, col]
            return result
        
        # Get annual financial statements
        income_annual = df_to_dict(ticker.income_stmt)
        balance_annual = df_to_dict(ticker.balance_sheet)
        cashflow_annual = df_to_dict(ticker.cashflow)
        
        # Get quarterly financial statements
        income_quarterly = df_to_dict(ticker.quarterly_income_stmt)
        balance_quarterly = df_to_dict(ticker.quarterly_balance_sheet)
        cashflow_quarterly = df_to_dict(ticker.quarterly_cashflow)
        
        result = {
            "symbol": ticker_symbol,
            "annual_financials": {
                "income_statement": income_annual,
                "balance_sheet": balance_annual,
                "cash_flow": cashflow_annual
            },
            "quarterly_financials": {
                "income_statement": income_quarterly,
                "balance_sheet": balance_quarterly,
                "cash_flow": cashflow_quarterly
            }
        }
        
        return result
    except Exception as e:
        return {"error": str(e), "symbol": ticker_symbol}

def get_options_data(ticker_symbol="TSLA"):
    """
    Get options chain data and analysis
    
    Args:
        ticker_symbol (str): Stock ticker symbol
        
    Returns:
        dict: JSON-serializable dictionary with options data
    """
    try:
        ticker = yf.Ticker(ticker_symbol)
        current_price = ticker.info.get('currentPrice', 0)
        
        # Get available expiration dates
        expirations = ticker.options
        
        if not expirations:
            return {
                "symbol": ticker_symbol,
                "error": "No options data available"
            }
        
        result = {
            "symbol": ticker_symbol,
            "current_price": current_price,
            "expiration_dates": expirations,
            "options_chains": {}
        }
        
        # Get data for up to 3 expiration dates
        for exp in expirations[:3]:
            option_chain = ticker.option_chain(exp)
            
            # Process calls
            calls_data = []
            for _, row in option_chain.calls.iterrows():
                calls_data.append({
                    "strike": row['strike'],
                    "last_price": row['lastPrice'],
                    "bid": row['bid'],
                    "ask": row['ask'],
                    "change": row['change'],
                    "percent_change": row['percentChange'],
                    "volume": row['volume'],
                    "open_interest": row['openInterest'],
                    "implied_volatility": row['impliedVolatility']
                })
            
            # Process puts
            puts_data = []
            for _, row in option_chain.puts.iterrows():
                puts_data.append({
                    "strike": row['strike'],
                    "last_price": row['lastPrice'],
                    "bid": row['bid'],
                    "ask": row['ask'],
                    "change": row['change'],
                    "percent_change": row['percentChange'],
                    "volume": row['volume'],
                    "open_interest": row['openInterest'],
                    "implied_volatility": row['impliedVolatility']
                })
            
            # Calculate put/call ratio based on open interest
            total_call_oi = sum(row['open_interest'] for row in calls_data if row['open_interest'] is not None)
            total_put_oi = sum(row['open_interest'] for row in puts_data if row['open_interest'] is not None)
            put_call_ratio = total_put_oi / total_call_oi if total_call_oi > 0 else None
            
            # Find ATM options
            atm_calls = []
            atm_puts = []
            
            if current_price > 0:
                option_chain.calls['price_diff'] = abs(option_chain.calls['strike'] - current_price)
                option_chain.puts['price_diff'] = abs(option_chain.puts['strike'] - current_price)
                
                # Get top 3 closest to ATM
                atm_call_indices = option_chain.calls['price_diff'].nsmallest(3).index
                atm_put_indices = option_chain.puts['price_diff'].nsmallest(3).index
                
                for idx in atm_call_indices:
                    row = option_chain.calls.loc[idx]
                    atm_calls.append({
                        "strike": row['strike'],
                        "last_price": row['lastPrice'],
                        "implied_volatility": row['impliedVolatility']
                    })
                
                for idx in atm_put_indices:
                    row = option_chain.puts.loc[idx]
                    atm_puts.append({
                        "strike": row['strike'],
                        "last_price": row['lastPrice'],
                        "implied_volatility": row['impliedVolatility']
                    })
            
            result["options_chains"][exp] = {
                "calls": calls_data,
                "puts": puts_data,
                "analysis": {
                    "put_call_ratio": put_call_ratio,
                    "atm_calls": atm_calls,
                    "atm_puts": atm_puts
                }
            }
        
        return result
    except Exception as e:
        return {"error": str(e), "symbol": ticker_symbol}

def get_ownership_data(ticker_symbol="NVDA"):
    """
    Get institutional ownership and insider transaction data
    
    Args:
        ticker_symbol (str): Stock ticker symbol
        
    Returns:
        dict: JSON-serializable dictionary with ownership data
    """
    try:
        ticker = yf.Ticker(ticker_symbol)
        
        # Process major holders
        major_holders_data = []
        major_holders = ticker.major_holders
        if not major_holders.empty:
            for _, row in major_holders.iterrows():
                if len(row) >= 2:  # Ensure there are at least 2 columns
                    major_holders_data.append({
                        "value": row[0],
                        "description": row[1]
                    })
        
        # Process institutional holders
        institutional_data = []
        inst_holders = ticker.institutional_holders
        if not inst_holders.empty:
            for _, row in inst_holders.iterrows():
                holder_data = {
                    "holder": row.get('Holder', None),
                    "shares": row.get('Shares', None),
                    "date_reported": row.get('Date Reported', None).strftime('%Y-%m-%d') if pd.notnull(row.get('Date Reported', None)) else None,
                    "percent_out": row.get('% Out', None),
                    "value": row.get('Value', None)
                }
                institutional_data.append(holder_data)
        
        # Process mutual fund holders
        fund_data = []
        fund_holders = ticker.mutualfund_holders
        if not fund_holders.empty:
            for _, row in fund_holders.iterrows():
                holder_data = {
                    "holder": row.get('Holder', None),
                    "shares": row.get('Shares', None),
                    "date_reported": row.get('Date Reported', None).strftime('%Y-%m-%d') if pd.notnull(row.get('Date Reported', None)) else None,
                    "percent_out": row.get('% Out', None),
                    "value": row.get('Value', None)
                }
                fund_data.append(holder_data)
        
        # Process insider transactions
        insider_data = []
        insider = ticker.insider_transactions
        if not insider.empty:
            for _, row in insider.iterrows():
                transaction = {
                    "insider": row.get('Insider', None),
                    "relation": row.get('Relation', None),
                    "transaction": row.get('Transaction', None),
                    "date": row.get('Date', None).strftime('%Y-%m-%d') if pd.notnull(row.get('Date', None)) else None,
                    "shares": row.get('Shares', None),
                    "value": row.get('Value', None),
                    "shares_total": row.get('Shares Total', None),
                    "sec_form": row.get('SEC Form 4', None)
                }
                insider_data.append(transaction)
        
        result = {
            "symbol": ticker_symbol,
            "major_holders": major_holders_data,
            "institutional_holders": institutional_data,
            "mutual_fund_holders": fund_data,
            "insider_transactions": insider_data
        }
        
        return result
    except Exception as e:
        return {"error": str(e), "symbol": ticker_symbol}

def get_news_and_recommendations(ticker_symbol="JPM"):
    """
    Get news articles and analyst recommendations
    
    Args:
        ticker_symbol (str): Stock ticker symbol
        
    Returns:
        dict: JSON-serializable dictionary with news and analyst recommendations
    """
    try:
        ticker = yf.Ticker(ticker_symbol)
        
        # Process news
        news_data = []
        news = ticker.news
        if news:
            for item in news:
                news_item = {
                    "title": item.get('title'),
                    "publisher": item.get('publisher'),
                    "link": item.get('link'),
                    "publish_time": datetime.fromtimestamp(item.get('providerPublishTime')).strftime('%Y-%m-%d %H:%M:%S') if item.get('providerPublishTime') else None,
                    "type": item.get('type'),
                    "related_tickers": item.get('relatedTickers')
                }
                news_data.append(news_item)
        
        # Process recommendations
        recommendation_data = []
        recommendations = ticker.recommendations
        if not recommendations.empty:
            for date, row in recommendations.iterrows():
                recommendation = {
                    "date": date.strftime('%Y-%m-%d'),
                    "firm": row.get('Firm', None),
                    "to_grade": row.get('To Grade', None),
                    "from_grade": row.get('From Grade', None),
                    "action": row.get('Action', None)
                }
                recommendation_data.append(recommendation)
        
        # Get recommendation trends if available
        recommendation_trend = {}
        try:
            rec_trend = ticker.recommendations_summary
            if not rec_trend.empty:
                for period in rec_trend.columns:
                    recommendation_trend[str(period)] = {
                        "strongBuy": rec_trend.loc['strongBuy', period],
                        "buy": rec_trend.loc['buy', period],
                        "hold": rec_trend.loc['hold', period],
                        "sell": rec_trend.loc['sell', period],
                        "strongSell": rec_trend.loc['strongSell', period]
                    }
        except:
            pass  # Recommendation trends might not be available
        
        result = {
            "symbol": ticker_symbol,
            "news": news_data,
            "recommendations": recommendation_data,
            "recommendation_trends": recommendation_trend
        }
        
        return result
    except Exception as e:
        return {"error": str(e), "symbol": ticker_symbol}

def get_multi_ticker_analysis(ticker_symbols=["AAPL", "MSFT", "GOOGL", "AMZN", "META"]):
    """
    Compare multiple tickers and provide analysis
    
    Args:
        ticker_symbols (list): List of stock ticker symbols
        
    Returns:
        dict: JSON-serializable dictionary with multi-ticker comparison
    """
    try:
        ticker_data = {}
        
        # Method 1: Get basic info for each ticker
        for symbol in ticker_symbols:
            ticker = yf.Ticker(symbol)
            info = ticker.info
            
            ticker_data[symbol] = {
                "company_name": info.get('shortName'),
                "sector": info.get('sector'),
                "industry": info.get('industry'),
                "market_cap": info.get('marketCap'),
                "pe_ratio": info.get('trailingPE'),
                "forward_pe": info.get('forwardPE'),
                "price_to_book": info.get('priceToBook'),
                "beta": info.get('beta'),
                "dividend_yield": info.get('dividendYield'),
                "current_price": info.get('currentPrice')
            }
        
        # Method 2: Get historical price data for all tickers at once
        end_date = datetime.now()
        start_date = end_date - timedelta(days=30)
        
        # Download price data for all tickers at once
        data = yf.download(ticker_symbols, start=start_date, end=end_date)
        
        # Calculate returns
        returns_data = {}
        correlation_matrix = {}
        
        if 'Adj Close' in data:
            returns = data['Adj Close'].pct_change().dropna()
            
            # Calculate returns for each ticker
            for symbol in ticker_symbols:
                if symbol in returns:
                    symbol_returns = returns[symbol]
                    total_return = ((symbol_returns + 1).cumprod().iloc[-1] - 1) * 100
                    returns_data[symbol] = {
                        "total_return_30d": total_return,
                        "avg_daily_return": symbol_returns.mean() * 100,
                        "volatility": symbol_returns.std() * 100
                    }
            
            # Calculate correlation matrix
            corr_matrix = returns.corr()
            
            # Convert correlation matrix to dict
            for symbol1 in ticker_symbols:
                if symbol1 in corr_matrix:
                    correlation_matrix[symbol1] = {}
                    for symbol2 in ticker_symbols:
                        if symbol2 in corr_matrix:
                            correlation_matrix[symbol1][symbol2] = corr_matrix.loc[symbol1, symbol2]
        
        result = {
            "tickers": ticker_symbols,
            "ticker_data": ticker_data,
            "performance": {
                "returns_30d": returns_data,
                "correlation_matrix": correlation_matrix
            }
        }
        
        return result
    except Exception as e:
        return {"error": str(e), "tickers": ticker_symbols}

def get_advanced_queries(ticker_symbol=None, query_type="dividends"):
    """
    Get advanced data for specific analysis types
    
    Args:
        ticker_symbol (str): Stock ticker symbol
        query_type (str): Type of query (dividends, splits, etf_holdings, sustainability)
        
    Returns:
        dict: JSON-serializable dictionary with requested data
    """
    result = {"query_type": query_type}
    
    try:
        if query_type == "dividends":
            ticker_symbol = ticker_symbol or "JNJ"
            ticker = yf.Ticker(ticker_symbol)
            dividends = ticker.dividends
            
            dividend_data = []
            annual_dividends = {}
            
            if not dividends.empty:
                # Individual dividends
                for date, value in dividends.items():
                    dividend_data.append({
                        "date": date.strftime('%Y-%m-%d'),
                        "amount": value
                    })
                
                # Annual dividends
                annual_div = dividends.resample('YE').sum()
                for date, value in annual_div.items():
                    year = date.year
                    annual_dividends[year] = value
                
                # Calculate growth rates
                growth_rates = {}
                div_growth = annual_div.pct_change() * 100
                for date, value in div_growth.items():
                    year = date.year
                    growth_rates[year] = value
                
                result.update({
                    "symbol": ticker_symbol,
                    "dividend_data": dividend_data,
                    "annual_dividends": annual_dividends,
                    "dividend_growth_rates": growth_rates,
                    "avg_5yr_growth": div_growth.tail(5).mean() if len(div_growth) >= 5 else None
                })
            else:
                result.update({
                    "symbol": ticker_symbol,
                    "dividend_data": [],
                    "message": "No dividend data available"
                })
                
        elif query_type == "splits":
            ticker_symbol = ticker_symbol or "TSLA"
            ticker = yf.Ticker(ticker_symbol)
            splits = ticker.splits
            
            split_data = []
            
            if not splits.empty:
                for date, ratio in splits.items():
                    split_data.append({
                        "date": date.strftime('%Y-%m-%d'),
                        "ratio": ratio
                    })
            
            result.update({
                "symbol": ticker_symbol,
                "split_data": split_data
            })
            
        elif query_type == "etf_holdings":
            ticker_symbol = ticker_symbol or "XLK"
            ticker = yf.Ticker(ticker_symbol)
            holdings = ticker.holdings
            
            holdings_data = []
            
            if not holdings.empty:
                for symbol, row in holdings.iterrows():
                    holding = {
                        "symbol": symbol,
                        "company": row.get('Company', None),
                        "percent": row.get('% Assets', None),
                        "shares": row.get('Shares', None),
                        "value": row.get('Value', None)
                    }
                    holdings_data.append(holding)
            
            result.update({
                "symbol": ticker_symbol,
                "holdings": holdings_data
            })
            
        elif query_type == "sustainability":
            ticker_symbol = ticker_symbol or "MSFT"
            ticker = yf.Ticker(ticker_symbol)
            sustainability = ticker.sustainability
            
            esg_data = {}
            
            if not sustainability.empty:
                for category, value in sustainability.iloc[:, 0].items():
                    esg_data[category] = value
            
            result.update({
                "symbol": ticker_symbol,
                "esg_data": esg_data
            })
            
        else:
            result.update({
                "error": f"Unknown query type: {query_type}"
            })
            
        return result
    except Exception as e:
        result.update({
            "symbol": ticker_symbol,
            "error": str(e)
        })
        return result

def get_pe_ratios(ticker_symbol="AAPL", peer_symbols=None):
    """
    Get P/E ratio analysis and comparison with industry peers
    
    Args:
        ticker_symbol (str): Stock ticker symbol
        peer_symbols (list): List of peer companies for comparison, or None to auto-detect
        
    Returns:
        dict: JSON-serializable dictionary with P/E ratio analysis
    """
    try:
        ticker = yf.Ticker(ticker_symbol)
        info = ticker.info
        
        # Get main ticker P/E data
        result = {
            "symbol": ticker_symbol,
            "company_name": info.get("shortName"),
            "sector": info.get("sector"),
            "industry": info.get("industry"),
            "valuation_metrics": {
                "current_price": info.get("currentPrice"),
                "trailing_pe": info.get("trailingPE"),
                "forward_pe": info.get("forwardPE"),
                "peg_ratio": info.get("pegRatio"),
                "price_to_book": info.get("priceToBook"),
                "price_to_sales": info.get("priceToSales"),
                "enterprise_value_to_revenue": info.get("enterpriseToRevenue"),
                "enterprise_value_to_ebitda": info.get("enterpriseToEbitda"),
                "eps_ttm": info.get("trailingEps"),
                "eps_forward": info.get("forwardEps"),
                "earnings_growth": info.get("earningsGrowth"),
                "revenue_growth": info.get("revenueGrowth")
            }
        }
        
        # Get historical P/E data (if earnings history is available)
        try:
            earnings_history = ticker.earnings
            if not earnings_history.empty:
                # Calculate historical P/E based on quarterly earnings
                historical_pe = []
                for date, row in earnings_history.iterrows():
                    year = date.year if hasattr(date, 'year') else date
                    historical_pe.append({
                        "year": year,
                        "earnings_per_share": row.get("Earnings"),
                        "estimated_pe": None  # Will calculate below if possible
                    })
                
                # Get historical price data to estimate P/E at earnings time
                try:
                    hist_data = ticker.history(period="5y")
                    if not hist_data.empty:
                        for pe_data in historical_pe:
                            # Find closest price data to the earnings year
                            year_prices = hist_data[hist_data.index.year == pe_data["year"]]
                            if not year_prices.empty:
                                avg_price = year_prices["Close"].mean()
                                if pe_data["earnings_per_share"] and pe_data["earnings_per_share"] > 0:
                                    pe_data["estimated_pe"] = avg_price / pe_data["earnings_per_share"]
                except:
                    pass  # Skip if historical price data is unavailable
                
                result["historical_pe"] = historical_pe
        except:
            pass  # Skip if earnings history is unavailable
        
        # Get peer comparison data
        if peer_symbols is None:
            # If no peers provided, try to get from same industry based on info
            industry = info.get("industry")
            if industry:
                try:
                    # This is a simplified approach - in a real app you might use a more
                    # sophisticated way to find peers
                    industry_tickers = []
                    # Example peer groups for common industries
                    peer_groups = {
                        "Technology": ["AAPL", "MSFT", "GOOGL", "META", "AMZN"],
                        "Semiconductors": ["NVDA", "AMD", "INTC", "TSM", "AVGO"],
                        "Banking": ["JPM", "BAC", "WFC", "C", "GS"],
                        "Retail": ["WMT", "TGT", "COST", "AMZN", "HD"],
                        "Healthcare": ["JNJ", "PFE", "MRK", "UNH", "ABBV"],
                        "Automotive": ["TSLA", "F", "GM", "TM", "STLA"]
                    }
                    
                    # Find the best matching peer group
                    for group_name, group_tickers in peer_groups.items():
                        if group_name.lower() in industry.lower() or any(word.lower() in industry.lower() for word in group_name.split()):
                            industry_tickers = group_tickers
                            break
                    
                    # If we found peers, use them
                    if industry_tickers:
                        peer_symbols = [t for t in industry_tickers if t != ticker_symbol]
                except:
                    peer_symbols = []
            
            # If still no peers, use a default set
            if not peer_symbols:
                if ticker_symbol in ["AAPL", "MSFT", "GOOGL", "META", "AMZN"]:
                    others = ["AAPL", "MSFT", "GOOGL", "META", "AMZN"]
                    peer_symbols = [t for t in others if t != ticker_symbol]
                else:
                    # Default to big tech if we can't find relevant peers
                    peer_symbols = ["AAPL", "MSFT", "GOOGL", "META", "AMZN"]
        
        # Get data for peer comparison
        peers_data = {}
        for peer in peer_symbols:
            try:
                peer_ticker = yf.Ticker(peer)
                peer_info = peer_ticker.info
                
                peers_data[peer] = {
                    "company_name": peer_info.get("shortName"),
                    "current_price": peer_info.get("currentPrice"),
                    "trailing_pe": peer_info.get("trailingPE"),
                    "forward_pe": peer_info.get("forwardPE"),
                    "peg_ratio": peer_info.get("pegRatio"),
                    "price_to_book": peer_info.get("priceToBook"),
                    "eps_ttm": peer_info.get("trailingEps"),
                    "market_cap": peer_info.get("marketCap")
                }
            except:
                peers_data[peer] = {"error": "Could not retrieve data"}
        
        # Calculate industry averages
        if peers_data:
            # Include the main ticker in calculations
            all_tickers = list(peers_data.keys()) + [ticker_symbol]
            all_data = peers_data.copy()
            all_data[ticker_symbol] = result["valuation_metrics"]
            
            # Calculate averages for key metrics
            metrics = ["trailing_pe", "forward_pe", "peg_ratio", "price_to_book"]
            industry_avgs = {}
            
            for metric in metrics:
                values = [data.get(metric) for ticker, data in all_data.items() 
                          if data.get(metric) is not None and data.get(metric) > 0]
                if values:
                    industry_avgs[metric] = sum(values) / len(values)
            
            result["peers"] = peers_data
            result["industry_averages"] = industry_avgs
        
        return result
    except Exception as e:
        return {"error": str(e), "symbol": ticker_symbol}

def get_debt_analysis(ticker_symbol="AAPL", period_years=5):
    """
    Get comprehensive debt analysis for a company
    
    Args:
        ticker_symbol (str): Stock ticker symbol
        period_years (int): Number of years of historical data to analyze
        
    Returns:
        dict: JSON-serializable dictionary with debt analysis
    """
    try:
        ticker = yf.Ticker(ticker_symbol)
        info = ticker.info
        
        # Basic debt information
        debt_info = {
            "symbol": ticker_symbol,
            "company_name": info.get("shortName"),
            "current_debt_metrics": {
                "total_debt": info.get("totalDebt"),
                "cash_and_equivalents": info.get("totalCash"),
                "net_debt": None,  # Will calculate if data available
                "debt_to_equity": info.get("debtToEquity"),
                "current_ratio": info.get("currentRatio"),
                "quick_ratio": info.get("quickRatio"),
                "interest_coverage": None,  # Will try to calculate from financial statements
                "total_debt_to_ebitda": None,  # Will try to calculate
            },
            "debt_components": {
                "short_term_debt": info.get("shortLongTermDebt"),
                "current_portion_of_long_term_debt": None,  # Will try to get from balance sheet
                "long_term_debt": info.get("longTermDebt"),
                "capital_lease_obligations": None,  # Will try to get from balance sheet
            },
            "credit_profile": {
                # Credit ratings aren't directly available in yfinance
                # but we can include other financial health metrics
                "current_enterprise_value": info.get("enterpriseValue"),
                "enterprise_value_to_revenue": info.get("enterpriseToRevenue"),
                "enterprise_value_to_ebitda": info.get("enterpriseToEbitda"),
                "beta": info.get("beta"),
                "52week_change": info.get("52WeekChange"),
                "free_cash_flow": info.get("freeCashflow")
            },
            "historical_debt": {
                "annual": {},
                "quarterly": {}
            }
        }
        
        # Calculate net debt if totalDebt and totalCash are available
        if info.get("totalDebt") is not None and info.get("totalCash") is not None:
            debt_info["current_debt_metrics"]["net_debt"] = info.get("totalDebt") - info.get("totalCash")
        
        # Get financial statements to extract more detailed debt information
        # Balance sheet analysis
        try:
            balance_sheet = ticker.balance_sheet
            if not balance_sheet.empty:
                # Extract debt-related items from most recent balance sheet
                latest_period = balance_sheet.columns[0]  # Most recent period
                
                # Try to find debt components (terminology varies between companies)
                debt_components = {}
                for row_name in balance_sheet.index:
                    row_lower = row_name.lower()
                    # Map common debt-related line items
                    if "short term debt" in row_lower or "short-term debt" in row_lower:
                        debt_components["short_term_debt"] = balance_sheet.loc[row_name, latest_period]
                    elif "current portion" in row_lower and "long term debt" in row_lower:
                        debt_components["current_portion_of_long_term_debt"] = balance_sheet.loc[row_name, latest_period]
                    elif "long term debt" in row_lower and "current portion" not in row_lower:
                        debt_components["long_term_debt"] = balance_sheet.loc[row_name, latest_period]
                    elif "capital lease" in row_lower or "lease obligation" in row_lower:
                        debt_components["capital_lease_obligations"] = balance_sheet.loc[row_name, latest_period]
                
                # Update debt_components with values found in balance sheet
                for component, value in debt_components.items():
                    if value is not None:
                        debt_info["debt_components"][component] = value
                
                # Try to find total stockholder equity
                for row_name in balance_sheet.index:
                    row_lower = row_name.lower()
                    if "total stockholder equity" in row_lower or "total shareholders' equity" in row_lower:
                        stockholder_equity = balance_sheet.loc[row_name, latest_period]
                        
                        # Calculate debt to equity if not already available
                        if debt_info["current_debt_metrics"]["debt_to_equity"] is None and stockholder_equity != 0:
                            total_debt = (debt_info["debt_components"]["short_term_debt"] or 0) + (debt_info["debt_components"]["long_term_debt"] or 0)
                            debt_info["current_debt_metrics"]["debt_to_equity"] = total_debt / stockholder_equity
                        break
            
            # Get historical annual balance sheet data for debt trend analysis
            historical_annual_debt = {}
            annual_debt_trend = []
            
            if not balance_sheet.empty:
                max_years = min(period_years, len(balance_sheet.columns))
                for i in range(max_years):
                    if i < len(balance_sheet.columns):
                        period = balance_sheet.columns[i]
                        period_str = period.strftime('%Y-%m-%d') if hasattr(period, 'strftime') else str(period)
                        
                        # Find debt values
                        short_term_debt = None
                        long_term_debt = None
                        total_debt = None
                        
                        for row_name in balance_sheet.index:
                            row_lower = row_name.lower()
                            if "short term debt" in row_lower or "short-term debt" in row_lower:
                                short_term_debt = balance_sheet.loc[row_name, period]
                            elif "long term debt" in row_lower and "current portion" not in row_lower:
                                long_term_debt = balance_sheet.loc[row_name, period]
                            elif "total debt" in row_lower:
                                total_debt = balance_sheet.loc[row_name, period]
                        
                        # If total_debt isn't directly available, calculate it
                        if total_debt is None and short_term_debt is not None and long_term_debt is not None:
                            total_debt = short_term_debt + long_term_debt
                        
                        # Store the debt data for this period
                        historical_annual_debt[period_str] = {
                            "short_term_debt": short_term_debt,
                            "long_term_debt": long_term_debt,
                            "total_debt": total_debt
                        }
                        
                        # Add to trend data for charting
                        if total_debt is not None:
                            annual_debt_trend.append({
                                "date": period_str,
                                "total_debt": total_debt,
                                "short_term_debt": short_term_debt,
                                "long_term_debt": long_term_debt
                            })
            
            debt_info["historical_debt"]["annual"] = historical_annual_debt
            debt_info["historical_debt"]["annual_trend"] = annual_debt_trend
        except Exception as e:
            debt_info["errors"] = {"balance_sheet": str(e)}
        
        # Income statement analysis to calculate interest coverage
        try:
            income_stmt = ticker.income_stmt
            if not income_stmt.empty:
                latest_period = income_stmt.columns[0]  # Most recent period
                
                # Try to find EBIT and interest expense
                ebit = None
                interest_expense = None
                
                for row_name in income_stmt.index:
                    row_lower = row_name.lower()
                    if "ebit" == row_lower:
                        ebit = income_stmt.loc[row_name, latest_period]
                    elif "operating income" == row_lower:
                        ebit = income_stmt.loc[row_name, latest_period]
                    elif "interest expense" in row_lower or "interest paid" in row_lower:
                        interest_expense = abs(income_stmt.loc[row_name, latest_period])  # Convert to positive number
                
                # Calculate interest coverage ratio
                if ebit is not None and interest_expense is not None and interest_expense != 0:
                    debt_info["current_debt_metrics"]["interest_coverage"] = ebit / interest_expense
                
                # Calculate EBITDA
                ebitda = None
                for row_name in income_stmt.index:
                    row_lower = row_name.lower()
                    if "ebitda" == row_lower:
                        ebitda = income_stmt.loc[row_name, latest_period]
                        break
                
                # If EBITDA not found directly, try to calculate it
                if ebitda is None and ebit is not None:
                    # Try to find depreciation and amortization
                    depreciation_amortization = None
                    for row_name in income_stmt.index:
                        row_lower = row_name.lower()
                        if "depreciation" in row_lower and "amortization" in row_lower:
                            depreciation_amortization = income_stmt.loc[row_name, latest_period]
                            break
                    
                    if depreciation_amortization is not None:
                        ebitda = ebit + depreciation_amortization
                
                # Calculate total debt to EBITDA ratio
                if ebitda is not None and ebitda != 0:
                    total_debt = info.get("totalDebt")
                    if total_debt is not None:
                        debt_info["current_debt_metrics"]["total_debt_to_ebitda"] = total_debt / ebitda
        except Exception as e:
            if "errors" not in debt_info:
                debt_info["errors"] = {}
            debt_info["errors"]["income_statement"] = str(e)
        
        # Calculate debt servicing capabilities and sustainability metrics
        try:
            cashflow = ticker.cashflow
            if not cashflow.empty:
                latest_period = cashflow.columns[0]  # Most recent period
                
                # Try to find operating cash flow and free cash flow
                operating_cash_flow = None
                capital_expenditure = None
                
                for row_name in cashflow.index:
                    row_lower = row_name.lower()
                    if "operating cash flow" in row_lower:
                        operating_cash_flow = cashflow.loc[row_name, latest_period]
                    elif "capital expenditure" in row_lower:
                        capital_expenditure = cashflow.loc[row_name, latest_period]
                
                # Calculate debt servicing metrics
                if operating_cash_flow is not None:
                    debt_info["debt_servicing"] = {
                        "operating_cash_flow": operating_cash_flow,
                        "capital_expenditure": capital_expenditure
                    }
                    
                    # Calculate debt to operating cash flow ratio
                    total_debt = info.get("totalDebt")
                    if total_debt is not None and operating_cash_flow != 0:
                        debt_info["debt_servicing"]["debt_to_operating_cash_flow"] = total_debt / operating_cash_flow
                    
                    # Calculate free cash flow if not directly found
                    if "free_cash_flow" not in debt_info["credit_profile"] or debt_info["credit_profile"]["free_cash_flow"] is None:
                        if capital_expenditure is not None:
                            # Free cash flow = Operating cash flow + Capital expenditure (negative number)
                            free_cash_flow = operating_cash_flow + capital_expenditure
                            debt_info["debt_servicing"]["free_cash_flow"] = free_cash_flow
                            
                            # Calculate debt to free cash flow ratio
                            if total_debt is not None and free_cash_flow != 0:
                                debt_info["debt_servicing"]["debt_to_free_cash_flow"] = total_debt / free_cash_flow
        except Exception as e:
            if "errors" not in debt_info:
                debt_info["errors"] = {}
            debt_info["errors"]["cash_flow"] = str(e)
        
        # Add debt maturity schedule if available (rarely provided via yfinance)
        # This would typically require specialized data sources or SEC filings analysis
        
        return debt_info
    except Exception as e:
        return {"error": str(e), "symbol": ticker_symbol}

def get_debt_profitability_ratios(ticker_symbol="AAPL", period_years=3):
    """
    Get debt-to-earnings and debt-to-profitability ratios with historical trends
    
    Args:
        ticker_symbol (str): Stock ticker symbol
        period_years (int): Number of years of historical data to analyze
        
    Returns:
        dict: JSON-serializable dictionary with debt-to-profitability metrics
    """
    try:
        ticker = yf.Ticker(ticker_symbol)
        info = ticker.info
        
        # Initialize result structure
        result = {
            "symbol": ticker_symbol,
            "company_name": info.get("shortName"),
            "sector": info.get("sector"),
            "industry": info.get("industry"),
            "current_ratios": {
                # Current year debt-to-profitability ratios
                "debt_to_ebitda": None,
                "debt_to_ebit": None,
                "debt_to_net_income": None,
                "debt_to_operating_income": None,
                "debt_to_free_cash_flow": None,
                "debt_to_revenue": None,
                "interest_coverage_ratio": None,
                "debt_service_coverage_ratio": None,
                "net_debt_to_ebitda": None,
                "net_debt_to_fcf": None
            },
            "profitability_metrics": {
                # Current year earnings and profitability metrics
                "revenue": None,
                "gross_profit": None,
                "operating_income": None,
                "ebitda": None,
                "ebit": None,
                "net_income": None,
                "free_cash_flow": None,
                "interest_expense": None,
                "operating_margin": None,
                "profit_margin": None,
                "fcf_margin": None
            },
            "debt_metrics": {
                # Current year debt metrics
                "total_debt": info.get("totalDebt"),
                "short_term_debt": info.get("shortLongTermDebt"),
                "long_term_debt": info.get("longTermDebt"),
                "cash_and_equivalents": info.get("totalCash"),
                "net_debt": None
            },
            "historical_ratios": {
                # Year-by-year debt-to-profitability ratios
                "annual": [],
                "quarterly": []
            }
        }
        
        # Calculate net debt
        if result["debt_metrics"]["total_debt"] is not None and result["debt_metrics"]["cash_and_equivalents"] is not None:
            result["debt_metrics"]["net_debt"] = result["debt_metrics"]["total_debt"] - result["debt_metrics"]["cash_and_equivalents"]
        
        # Get most recent financial data from statements
        try:
            # Income statement for earnings metrics
            income_stmt = ticker.income_stmt
            if not income_stmt.empty and len(income_stmt.columns) > 0:
                latest_period = income_stmt.columns[0]  # Most recent period
                
                # Extract profitability metrics from income statement
                for row_name in income_stmt.index:
                    row_lower = row_name.lower()
                    
                    if "total revenue" in row_lower or row_lower == "revenue":
                        result["profitability_metrics"]["revenue"] = income_stmt.loc[row_name, latest_period]
                    elif "gross profit" in row_lower:
                        result["profitability_metrics"]["gross_profit"] = income_stmt.loc[row_name, latest_period]
                    elif "operating income" in row_lower:
                        result["profitability_metrics"]["operating_income"] = income_stmt.loc[row_name, latest_period]
                        # EBIT is typically the same as operating income
                        result["profitability_metrics"]["ebit"] = income_stmt.loc[row_name, latest_period]
                    elif row_lower == "ebitda":
                        result["profitability_metrics"]["ebitda"] = income_stmt.loc[row_name, latest_period]
                    elif "net income" in row_lower:
                        result["profitability_metrics"]["net_income"] = income_stmt.loc[row_name, latest_period]
                    elif "interest expense" in row_lower:
                        # Store as positive number for ratio calculations
                        result["profitability_metrics"]["interest_expense"] = abs(income_stmt.loc[row_name, latest_period])
                
                # If EBITDA not directly found, calculate it by adding D&A to EBIT
                if result["profitability_metrics"]["ebitda"] is None and result["profitability_metrics"]["ebit"] is not None:
                    for row_name in income_stmt.index:
                        row_lower = row_name.lower()
                        if "depreciation" in row_lower and "amortization" in row_lower:
                            depreciation_amortization = income_stmt.loc[row_name, latest_period]
                            result["profitability_metrics"]["ebitda"] = result["profitability_metrics"]["ebit"] + abs(depreciation_amortization)
                            break
                
                # Calculate profitability margins
                if result["profitability_metrics"]["revenue"] is not None and result["profitability_metrics"]["revenue"] != 0:
                    if result["profitability_metrics"]["operating_income"] is not None:
                        result["profitability_metrics"]["operating_margin"] = (
                            result["profitability_metrics"]["operating_income"] / result["profitability_metrics"]["revenue"]
                        )
                    
                    if result["profitability_metrics"]["net_income"] is not None:
                        result["profitability_metrics"]["profit_margin"] = (
                            result["profitability_metrics"]["net_income"] / result["profitability_metrics"]["revenue"]
                        )
            
            # Cash flow statement for free cash flow
            cashflow = ticker.cashflow
            if not cashflow.empty and len(cashflow.columns) > 0:
                latest_period = cashflow.columns[0]  # Most recent period
                
                # Try to find FCF components
                operating_cash_flow = None
                capital_expenditure = None
                
                for row_name in cashflow.index:
                    row_lower = row_name.lower()
                    if "operating cash flow" in row_lower:
                        operating_cash_flow = cashflow.loc[row_name, latest_period]
                    elif "capital expenditure" in row_lower:
                        capital_expenditure = cashflow.loc[row_name, latest_period]
                    elif "free cash flow" in row_lower:
                        result["profitability_metrics"]["free_cash_flow"] = cashflow.loc[row_name, latest_period]
                
                # Calculate FCF if not directly found
                if result["profitability_metrics"]["free_cash_flow"] is None and operating_cash_flow is not None and capital_expenditure is not None:
                    result["profitability_metrics"]["free_cash_flow"] = operating_cash_flow + capital_expenditure  # CapEx is usually negative
                
                # Calculate FCF margin
                if result["profitability_metrics"]["free_cash_flow"] is not None and result["profitability_metrics"]["revenue"] is not None and result["profitability_metrics"]["revenue"] != 0:
                    result["profitability_metrics"]["fcf_margin"] = (
                        result["profitability_metrics"]["free_cash_flow"] / result["profitability_metrics"]["revenue"]
                    )
            
            # Now calculate debt-to-profitability ratios
            total_debt = result["debt_metrics"]["total_debt"]
            net_debt = result["debt_metrics"]["net_debt"]
            
            if total_debt is not None:
                # Debt to EBITDA
                if result["profitability_metrics"]["ebitda"] is not None and result["profitability_metrics"]["ebitda"] > 0:
                    result["current_ratios"]["debt_to_ebitda"] = total_debt / result["profitability_metrics"]["ebitda"]
                
                # Debt to EBIT
                if result["profitability_metrics"]["ebit"] is not None and result["profitability_metrics"]["ebit"] > 0:
                    result["current_ratios"]["debt_to_ebit"] = total_debt / result["profitability_metrics"]["ebit"]
                
                # Debt to Net Income
                if result["profitability_metrics"]["net_income"] is not None and result["profitability_metrics"]["net_income"] > 0:
                    result["current_ratios"]["debt_to_net_income"] = total_debt / result["profitability_metrics"]["net_income"]
                
                # Debt to Operating Income
                if result["profitability_metrics"]["operating_income"] is not None and result["profitability_metrics"]["operating_income"] > 0:
                    result["current_ratios"]["debt_to_operating_income"] = total_debt / result["profitability_metrics"]["operating_income"]
                
                # Debt to Free Cash Flow
                if result["profitability_metrics"]["free_cash_flow"] is not None and result["profitability_metrics"]["free_cash_flow"] > 0:
                    result["current_ratios"]["debt_to_free_cash_flow"] = total_debt / result["profitability_metrics"]["free_cash_flow"]
                
                # Debt to Revenue
                if result["profitability_metrics"]["revenue"] is not None and result["profitability_metrics"]["revenue"] > 0:
                    result["current_ratios"]["debt_to_revenue"] = total_debt / result["profitability_metrics"]["revenue"]
            
            # Interest Coverage Ratio
            if result["profitability_metrics"]["ebit"] is not None and result["profitability_metrics"]["interest_expense"] is not None and result["profitability_metrics"]["interest_expense"] > 0:
                result["current_ratios"]["interest_coverage_ratio"] = result["profitability_metrics"]["ebit"] / result["profitability_metrics"]["interest_expense"]
            
            # Debt Service Coverage Ratio (EBITDA / (Interest + Principal Payments))
            # Note: Principal payments are not directly available in yfinance data
            if result["profitability_metrics"]["ebitda"] is not None and result["profitability_metrics"]["interest_expense"] is not None and result["profitability_metrics"]["interest_expense"] > 0:
                # This is a simplified version using just interest expense
                result["current_ratios"]["debt_service_coverage_ratio"] = result["profitability_metrics"]["ebitda"] / result["profitability_metrics"]["interest_expense"]
            
            # Net Debt ratios
            if net_debt is not None:
                # Net Debt to EBITDA
                if result["profitability_metrics"]["ebitda"] is not None and result["profitability_metrics"]["ebitda"] > 0:
                    result["current_ratios"]["net_debt_to_ebitda"] = net_debt / result["profitability_metrics"]["ebitda"]
                
                # Net Debt to FCF
                if result["profitability_metrics"]["free_cash_flow"] is not None and result["profitability_metrics"]["free_cash_flow"] > 0:
                    result["current_ratios"]["net_debt_to_fcf"] = net_debt / result["profitability_metrics"]["free_cash_flow"]
            
            # Historical analysis - get annual trends for key debt-to-profitability ratios
            annual_ratios = []
            
            # Process annual income statements
            max_years = min(period_years, len(income_stmt.columns))
            
            for i in range(max_years):
                if i < len(income_stmt.columns):
                    period = income_stmt.columns[i]
                    period_str = period.strftime('%Y-%m-%d') if hasattr(period, 'strftime') else str(period)
                    
                    # Create structure for this period
                    period_data = {
                        "date": period_str,
                        "profitability": {},
                        "debt": {},
                        "ratios": {}
                    }
                    
                    # Extract profitability metrics for this period
                    for row_name in income_stmt.index:
                        row_lower = row_name.lower()
                        
                        if "total revenue" in row_lower or row_lower == "revenue":
                            period_data["profitability"]["revenue"] = income_stmt.loc[row_name, period]
                        elif "operating income" in row_lower:
                            period_data["profitability"]["operating_income"] = income_stmt.loc[row_name, period]
                            period_data["profitability"]["ebit"] = income_stmt.loc[row_name, period]
                        elif row_lower == "ebitda":
                            period_data["profitability"]["ebitda"] = income_stmt.loc[row_name, period]
                        elif "net income" in row_lower:
                            period_data["profitability"]["net_income"] = income_stmt.loc[row_name, period]
                        elif "interest expense" in row_lower:
                            period_data["profitability"]["interest_expense"] = abs(income_stmt.loc[row_name, period])
                    
                    # Get cash flow data for this period if available
                    if i < len(cashflow.columns):
                        cashflow_period = cashflow.columns[i]
                        # Only process if the periods match
                        if str(cashflow_period).split(" ")[0] == str(period).split(" ")[0]:
                            for row_name in cashflow.index:
                                row_lower = row_name.lower()
                                if "free cash flow" in row_lower:
                                    period_data["profitability"]["free_cash_flow"] = cashflow.loc[row_name, cashflow_period]
                    
                    # Try to get debt data for this period
                    # This would typically come from the balance sheet, but we need to match the periods
                    balance_sheet = ticker.balance_sheet
                    if not balance_sheet.empty and i < len(balance_sheet.columns):
                        bs_period = balance_sheet.columns[i]
                        # Only process if the periods match approximately (year-to-year)
                        if str(bs_period).split("-")[0] == str(period).split("-")[0]:
                            for row_name in balance_sheet.index:
                                row_lower = row_name.lower()
                                
                                if "total debt" in row_lower:
                                    period_data["debt"]["total_debt"] = balance_sheet.loc[row_name, bs_period]
                                elif "long term debt" in row_lower and "current" not in row_lower:
                                    period_data["debt"]["long_term_debt"] = balance_sheet.loc[row_name, bs_period]
                                elif "short term debt" in row_lower or "short-term debt" in row_lower:
                                    period_data["debt"]["short_term_debt"] = balance_sheet.loc[row_name, bs_period]
                                elif "cash and cash equivalents" in row_lower:
                                    period_data["debt"]["cash_and_equivalents"] = balance_sheet.loc[row_name, bs_period]
                    
                    # Calculate total debt if not found directly
                    if "total_debt" not in period_data["debt"] and "long_term_debt" in period_data["debt"] and "short_term_debt" in period_data["debt"]:
                        period_data["debt"]["total_debt"] = period_data["debt"]["long_term_debt"] + period_data["debt"]["short_term_debt"]
                    
                    # Calculate net debt if possible
                    if "total_debt" in period_data["debt"] and "cash_and_equivalents" in period_data["debt"]:
                        period_data["debt"]["net_debt"] = period_data["debt"]["total_debt"] - period_data["debt"]["cash_and_equivalents"]
                    
                    # Calculate key ratios for this period
                    if "total_debt" in period_data["debt"]:
                        total_debt = period_data["debt"]["total_debt"]
                        
                        # Debt to EBITDA
                        if "ebitda" in period_data["profitability"] and period_data["profitability"]["ebitda"] > 0:
                            period_data["ratios"]["debt_to_ebitda"] = total_debt / period_data["profitability"]["ebitda"]
                        
                        # Debt to Net Income
                        if "net_income" in period_data["profitability"] and period_data["profitability"]["net_income"] > 0:
                            period_data["ratios"]["debt_to_net_income"] = total_debt / period_data["profitability"]["net_income"]
                        
                        # Debt to FCF
                        if "free_cash_flow" in period_data["profitability"] and period_data["profitability"]["free_cash_flow"] > 0:
                            period_data["ratios"]["debt_to_free_cash_flow"] = total_debt / period_data["profitability"]["free_cash_flow"]
                    
                    # Calculate interest coverage for this period
                    if "ebit" in period_data["profitability"] and "interest_expense" in period_data["profitability"] and period_data["profitability"]["interest_expense"] > 0:
                        period_data["ratios"]["interest_coverage_ratio"] = period_data["profitability"]["ebit"] / period_data["profitability"]["interest_expense"]
                    
                    # Add to historical data
                    annual_ratios.append(period_data)
            
            # Add historical data to result
            result["historical_ratios"]["annual"] = annual_ratios
            
            # Add industry average comparison if available
            try:
                # This would typically require a database of industry averages
                # For demonstration, we'll just note that this would be calculated here
                pass
            except:
                pass
            
        except Exception as e:
            if "errors" not in result:
                result["errors"] = {}
            result["errors"]["financial_statements"] = str(e)
        
        return result
    except Exception as e:
        return {"error": str(e), "symbol": ticker_symbol}

def demo_all_functions():
    """
    Demonstrate all functions with sample output
    
    Returns:
        dict: JSON-serializable dictionary with sample outputs from all functions
    """
    results = {
        "basic_info": get_basic_info("AAPL"),
        "historical_data": get_historical_data("MSFT", period="1mo"),
        "financial_statements": get_financial_statements("GOOGL"),
        "options_data": get_options_data("TSLA"),
        "ownership_data": get_ownership_data("NVDA"),
        "news_and_recommendations": get_news_and_recommendations("JPM"),
        "pe_ratios": get_pe_ratios("AAPL", ["MSFT", "GOOGL", "META"]),
        "debt_analysis": get_debt_analysis("JNJ", period_years=3),
        "debt_profitability": get_debt_profitability_ratios("T", period_years=3),
        "multi_ticker_analysis": get_multi_ticker_analysis(["AAPL", "MSFT", "GOOGL"]),
        "advanced_queries": {
            "dividends": get_advanced_queries(query_type="dividends"),
            "splits": get_advanced_queries(query_type="splits"),
            "etf_holdings": get_advanced_queries(query_type="etf_holdings"),
            "sustainability": get_advanced_queries(query_type="sustainability")
        }
    }
    
    return results

# Example usage
if __name__ == "__main__":
    # Get data for a specific function
    aapl_info = get_basic_info("AAPL")
    print(json.dumps(aapl_info, indent=2))
    
    # Or get all data
    # all_data = demo_all_functions()
    # Save to file
    # with open('yfinance_data.json', 'w') as f:
    #     json.dump(all_data, f, indent=2)
