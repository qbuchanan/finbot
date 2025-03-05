import yfinance as yf
from datetime import datetime, timedelta

def calculate_investment_value(ticker_symbol, purchase_date, num_shares, end_date=None):
    """
    Calculate the current value of a stock investment made in the past,
    accounting for stock splits.
    
    Args:
        ticker_symbol (str): Stock ticker symbol
        purchase_date (str): Date of purchase in format 'YYYY-MM-DD'
        num_shares (float): Number of shares purchased
        end_date (str, optional): End date for calculation in format 'YYYY-MM-DD'.
                                 If None, uses current date.
    
    Returns:
        dict: Dictionary with details about the investment including:
              - original investment details
              - current investment value
              - splits that occurred
              - total return
    """
    try:
        # Initialize ticker object
        ticker = yf.Ticker(ticker_symbol)
        
        # Parse purchase date
        purchase_date = datetime.strptime(purchase_date, '%Y-%m-%d')
        
        # Set end date (either provided or current date)
        if end_date:
            end_date = datetime.strptime(end_date, '%Y-%m-%d')
        else:
            end_date = datetime.now()
        
        # take care of timezone info so we can do date arthmetic

        if end_date.tzinfo is not None:
            end_date = end_date.replace(tzinfo=None)
        # Get historical data for purchase date
        # Add a small window to ensure we get data (in case of weekends/holidays)
        start_window = purchase_date - timedelta(days=7)  # One week before
        purchase_window = ticker.history(start=start_window.strftime('%Y-%m-%d'), 
                                        end=(purchase_date + timedelta(days=7)).strftime('%Y-%m-%d'))
        
        if purchase_window.empty:
            return {"error": f"No historical data available for {ticker_symbol} around {purchase_date.strftime('%Y-%m-%d')}"}
        
        # Find the closest trading day on or after the purchase date
        purchase_price = None
        actual_purchase_date = None
        
        for date, row in purchase_window.iterrows():
            if date.date() >= purchase_date.date():
                purchase_price = row['Close']
                actual_purchase_date = date
                break
        
        # If no trading day on or after purchase date in our window, use the last available date
        if purchase_price is None:
            actual_purchase_date = purchase_window.index[-1]
            purchase_price = purchase_window.loc[actual_purchase_date, 'Close']
        
        # Remove timezone info from both dates
        if actual_purchase_date.tzinfo is not None:
            actual_purchase_date = actual_purchase_date.replace(tzinfo=None)

        # Get all splits that occurred after the purchase
        splits = ticker.splits
        post_purchase_splits = {}
        
        total_split_factor = 1.0
        for split_date, split_ratio in splits.items():
            if split_date.date() > actual_purchase_date.date():
                post_purchase_splits[split_date.strftime('%Y-%m-%d')] = split_ratio
                total_split_factor *= split_ratio

        # Current adjusted shares
        current_shares = num_shares * total_split_factor
        
        # Get current stock price
        try:
            current_data = ticker.history(period='1d')
            if current_data.empty:
                return {"error": f"Could not retrieve current price for {ticker_symbol}"}
            
            current_price = current_data['Close'].iloc[-1]
        except Exception as e:
            return {"error": f"Failed to get current price: {str(e)}"}
        
        # Calculate values
        original_investment = purchase_price * num_shares
        current_value = current_price * current_shares
        total_return_pct = ((current_value / original_investment) - 1) * 100
        
        # Get company info
        try:
            info = ticker.info
            company_name = info.get('shortName', ticker_symbol)
        except:
            company_name = ticker_symbol
        
        # Prepare result
        result = {
            "symbol": ticker_symbol,
            "company_name": company_name,
            "purchase_details": {
                "date": actual_purchase_date.strftime('%Y-%m-%d'),
                "price_per_share": purchase_price,
                "shares_purchased": num_shares,
                "total_investment": original_investment
            },
            "split_history": {
                "splits": post_purchase_splits,
                "total_split_factor": total_split_factor,
                "split_adjusted_shares": current_shares
            },
            "current_details": {
                "date": end_date.strftime('%Y-%m-%d'),
                "price_per_share": current_price,
                "adjusted_shares": current_shares,
                "current_value": current_value
            },
            "performance": {
                "total_return_dollars": current_value - original_investment,
                "total_return_percent": total_return_pct,
                "annualized_return": None  # Will calculate below
            }
        }
        
        # Calculate annualized return if more than a year has passed
        years_held = (end_date - actual_purchase_date).days / 365.25
        result["performance"]["years_held"] = years_held
        
        if years_held > 1:
            annualized_return = ((current_value / original_investment) ** (1 / years_held) - 1) * 100
            result["performance"]["annualized_return"] = annualized_return
        
        return result
    
    except Exception as e:
        return {"error": f"An error occurred: {str(e)}"}

def print_investment_summary(investment_data):
    """
    Print a summary of the investment data in a readable format.
    
    Args:
        investment_data (dict): Investment data returned by calculate_investment_value()
    """
    if "error" in investment_data:
        print(f"Error: {investment_data['error']}")
        return
    
    print(f"\n=== INVESTMENT SUMMARY FOR {investment_data['company_name']} ({investment_data['symbol']}) ===\n")
    
    # Purchase details
    purchase = investment_data['purchase_details']
    print(f"PURCHASE DETAILS:")
    print(f"Date: {purchase['date']}")
    print(f"Price per share: ${purchase['price_per_share']:.2f}")
    print(f"Shares purchased: {purchase['shares_purchased']:.2f}")
    print(f"Initial investment: ${purchase['total_investment']:.2f}")
    
    # Split history
    splits = investment_data['split_history']
    print(f"\nSTOCK SPLITS:")
    if splits['splits']:
        for date, ratio in splits['splits'].items():
            print(f"{date}: {ratio}-for-1 split")
        print(f"Total split factor: {splits['total_split_factor']:.2f}")
        print(f"Split-adjusted shares: {splits['split_adjusted_shares']:.2f}")
    else:
        print("No splits have occurred since purchase")
    
    # Current details
    current = investment_data['current_details']
    print(f"\nCURRENT DETAILS:")
    print(f"Date: {current['date']}")
    print(f"Current price: ${current['price_per_share']:.2f}")
    print(f"Current shares: {current['adjusted_shares']:.2f}")
    print(f"Current value: ${current['current_value']:.2f}")
    
    # Performance
    perf = investment_data['performance']
    print(f"\nPERFORMANCE:")
    print(f"Total return: ${perf['total_return_dollars']:.2f} ({perf['total_return_percent']:.2f}%)")
    print(f"Holding period: {perf['years_held']:.2f} years")
    
    if perf['annualized_return'] is not None:
        print(f"Annualized return: {perf['annualized_return']:.2f}%")
    
    print("\n" + "=" * 60)

# Example usage 
if __name__ == "__main__":
    # Example 1: Apple shares purchased in 2005
    apple_investment = calculate_investment_value("AAPL", "2005-01-03", 100)
    print_investment_summary(apple_investment)
    
    # Example 2: Tesla shares purchased just before its big run
    tesla_investment = calculate_investment_value("TSLA", "2019-01-02", 50)
    print_investment_summary(tesla_investment)
    
    # Example 3: Microsoft shares from the 90s
    microsoft_investment = calculate_investment_value("MSFT", "1995-01-03", 200)
    print_investment_summary(microsoft_investment)
