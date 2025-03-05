import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import yfinance as yf
from typing import List, Dict, Callable, Union, Optional, Tuple


class StockIndexCalculator:
    """
    A class for calculating custom stock market indices similar to DOW Jones, S&P 500, etc.
    Supports different calculation methodologies and time periods.
    """
    
    # Supported calculation methods
    CALCULATION_METHODS = {
        'price_weighted': 'Calculate index based on stock prices (like DOW)',
        'market_cap_weighted': 'Calculate index based on market capitalization (like S&P 500)',
        'equal_weighted': 'Calculate index where each stock has equal weight',
        'float_adjusted': 'Calculate index based on float-adjusted market cap',
        'fundamental_weighted': 'Calculate index based on fundamental metrics (e.g., earnings, book value)'
    }
    
    # Supported time steps
    TIME_STEPS = ['daily', 'weekly', 'monthly', 'quarterly', 'yearly']
    
    def __init__(self, base_value: float = 100.0, base_date: str = None):
        """
        Initialize the index calculator.
        
        Args:
            base_value: The starting value for the index
            base_date: The starting date for the index in 'YYYY-MM-DD' format
        """
        self.base_value = base_value
        self.base_date = base_date if base_date else datetime.now().strftime('%Y-%m-%d')
        self.stocks_data = {}
        self.divisor = None
    
    def add_stocks(self, symbols: List[str]) -> None:
        """
        Add a list of stock symbols to the index.
        
        Args:
            symbols: List of stock ticker symbols
        """
        for symbol in symbols:
            if symbol not in self.stocks_data:
                self.stocks_data[symbol] = None
    
    def remove_stocks(self, symbols: List[str]) -> None:
        """
        Remove stocks from the index.
        
        Args:
            symbols: List of stock ticker symbols to remove
        """
        for symbol in symbols:
            if symbol in self.stocks_data:
                del self.stocks_data[symbol]
    
    def get_stock_data(self, start_date: str, end_date: str = None, interval: str = 'daily') -> pd.DataFrame:
        """
        Fetch historical stock data for all stocks in the index.
        
        Args:
            start_date: Start date in 'YYYY-MM-DD' format
            end_date: End date in 'YYYY-MM-DD' format (defaults to today)
            interval: Time interval ('daily', 'weekly', 'monthly', etc.)
            
        Returns:
            DataFrame containing the stock price history
        """
        if not end_date:
            end_date = datetime.now().strftime('%Y-%m-%d')
            
        # Convert interval to yfinance format
        yf_interval = '1d'  # default daily
        if interval == 'weekly':
            yf_interval = '1wk'
        elif interval == 'monthly':
            yf_interval = '1mo'
        
        # Fetch data for all symbols
        symbols = list(self.stocks_data.keys())
        if not symbols:
            raise ValueError("No stocks added to the index")
            
        data = yf.download(symbols, start=start_date, end=end_date, interval=yf_interval)
        
        # If only one stock, yfinance returns a different format
        if len(symbols) == 1:
            data = data.iloc[:, data.columns.get_level_values(0) == 'Close']
            data.columns = [symbols[0]]
        else:
            data = data['Close']
            
        return data
    
    def calculate_divisor(self, prices: pd.Series, weights: pd.Series = None) -> float:
        """
        Calculate the index divisor to set the base value.
        
        Args:
            prices: Series of stock prices
            weights: Series of weights for each stock (optional)
            
        Returns:
            The calculated divisor
        """
        if weights is None:
            # Default to equal weights
            weights = pd.Series(1.0, index=prices.index)
            
        # Calculate weighted sum of prices
        weighted_sum = sum(prices * weights)
        
        # Divisor to achieve the base value
        return weighted_sum / self.base_value
    
    def price_weighted_calculation(self, prices: pd.DataFrame) -> pd.Series:
        """
        Calculate a price-weighted index (like DOW Jones).
        
        Args:
            prices: DataFrame of closing prices
            
        Returns:
            Series of index values
        """
        # Sum prices and divide by divisor
        index_values = prices.sum(axis=1) / self.divisor
        return index_values
    
    def market_cap_weighted_calculation(self, prices: pd.DataFrame, market_caps: pd.DataFrame) -> pd.Series:
        """
        Calculate a market-cap weighted index (like S&P 500).
        
        Args:
            prices: DataFrame of closing prices
            market_caps: DataFrame of market capitalizations
            
        Returns:
            Series of index values
        """
        # Calculate weights for each date
        weights = market_caps.div(market_caps.sum(axis=1), axis=0)
        
        # Multiply prices by weights and sum
        weighted_prices = prices * weights
        index_values = weighted_prices.sum(axis=1) * self.base_value / weighted_prices.iloc[0].sum()
        
        return index_values
    
    def equal_weighted_calculation(self, prices: pd.DataFrame) -> pd.Series:
        """
        Calculate an equal-weighted index.
        
        Args:
            prices: DataFrame of closing prices
            
        Returns:
            Series of index values
        """
        # Normalize to starting price
        normalized = prices.div(prices.iloc[0])
        
        # Average of normalized prices
        index_values = normalized.mean(axis=1) * self.base_value
        
        return index_values
    
    def float_adjusted_calculation(self, prices: pd.DataFrame, float_shares: pd.DataFrame) -> pd.Series:
        """
        Calculate a float-adjusted market cap weighted index.
        
        Args:
            prices: DataFrame of closing prices
            float_shares: DataFrame of float-adjusted outstanding shares
            
        Returns:
            Series of index values
        """
        # Calculate float-adjusted market cap
        float_market_cap = prices * float_shares
        
        # Weights based on float-adjusted market cap
        weights = float_market_cap.div(float_market_cap.sum(axis=1), axis=0)
        
        # Weighted sum of prices
        weighted_prices = prices * weights
        index_values = weighted_prices.sum(axis=1) * self.base_value / weighted_prices.iloc[0].sum()
        
        return index_values
    
    def fundamental_weighted_calculation(self, prices: pd.DataFrame, fundamental_data: pd.DataFrame) -> pd.Series:
        """
        Calculate a fundamental-weighted index (e.g., based on earnings, book value).
        
        Args:
            prices: DataFrame of closing prices
            fundamental_data: DataFrame of fundamental metrics (e.g., earnings)
            
        Returns:
            Series of index values
        """
        # Weights based on fundamental data
        weights = fundamental_data.div(fundamental_data.sum(axis=1), axis=0)
        
        # Weighted sum of prices
        weighted_prices = prices * weights
        index_values = weighted_prices.sum(axis=1) * self.base_value / weighted_prices.iloc[0].sum()
        
        return index_values
    
    def calculate_index(self, 
                        method: str = 'price_weighted',
                        start_date: str = None,
                        end_date: str = None,
                        interval: str = 'daily',
                        **kwargs) -> pd.DataFrame:
        """
        Calculate the index value over a specified period.
        
        Args:
            method: Calculation method (price_weighted, market_cap_weighted, etc.)
            start_date: Start date in 'YYYY-MM-DD' format
            end_date: End date in 'YYYY-MM-DD' format (defaults to today)
            interval: Time interval (daily, weekly, monthly, etc.)
            **kwargs: Additional parameters for specific calculation methods
            
        Returns:
            DataFrame containing index values over time
        """
        if method not in self.CALCULATION_METHODS:
            raise ValueError(f"Invalid calculation method: {method}. Available methods: {list(self.CALCULATION_METHODS.keys())}")
            
        if interval not in self.TIME_STEPS:
            raise ValueError(f"Invalid time step: {interval}. Available steps: {self.TIME_STEPS}")
            
        # Set start date to base_date if not provided
        if not start_date:
            start_date = self.base_date
            
        # Get stock price data
        prices_df = self.get_stock_data(start_date, end_date, interval)
        
        # Calculate index based on selected method
        if method == 'price_weighted':
            # Calculate divisor using the first day's prices
            if self.divisor is None:
                self.divisor = self.calculate_divisor(prices_df.iloc[0])
            index_values = self.price_weighted_calculation(prices_df)
            
        elif method == 'market_cap_weighted':
            # We need market cap data
            if 'market_caps' not in kwargs:
                # Try to fetch market cap data
                market_caps = self._get_market_cap_data(prices_df.index, list(self.stocks_data.keys()))
            else:
                market_caps = kwargs['market_caps']
                
            index_values = self.market_cap_weighted_calculation(prices_df, market_caps)
            
        elif method == 'equal_weighted':
            index_values = self.equal_weighted_calculation(prices_df)
            
        elif method == 'float_adjusted':
            # We need float-adjusted shares data
            if 'float_shares' not in kwargs:
                raise ValueError("float_shares DataFrame is required for float_adjusted calculation")
            
            index_values = self.float_adjusted_calculation(prices_df, kwargs['float_shares'])
            
        elif method == 'fundamental_weighted':
            # We need fundamental data
            if 'fundamental_data' not in kwargs:
                raise ValueError("fundamental_data DataFrame is required for fundamental_weighted calculation")
                
            index_values = self.fundamental_weighted_calculation(prices_df, kwargs['fundamental_data'])
            
        # Create a DataFrame with the index values
        index_df = pd.DataFrame({'index_value': index_values})
        
        return index_df
    
    def get_current_index_value(self, 
                               method: str = 'price_weighted',
                               date: str = None,
                               **kwargs) -> float:
        """
        Calculate the index value for a specific date (defaults to latest available).
        
        Args:
            method: Calculation method
            date: Date in 'YYYY-MM-DD' format (defaults to today/latest available)
            **kwargs: Additional parameters for specific calculation methods
            
        Returns:
            The index value for the specified date
        """
        # If no date specified, use current date
        if not date:
            end_date = datetime.now().strftime('%Y-%m-%d')
        else:
            end_date = date
            
        # Calculate for a week prior to ensure we have data
        start_date = (datetime.strptime(end_date, '%Y-%m-%d') - timedelta(days=7)).strftime('%Y-%m-%d')
        
        # Calculate the index
        index_df = self.calculate_index(method=method, start_date=start_date, end_date=end_date, **kwargs)
        
        # Return the last available value
        return index_df.iloc[-1]['index_value']
    
    def _get_market_cap_data(self, dates: pd.DatetimeIndex, symbols: List[str]) -> pd.DataFrame:
        """
        Fetch market capitalization data for the stocks in the index.
        
        Args:
            dates: DatetimeIndex for which to get market cap data
            symbols: List of stock symbols
            
        Returns:
            DataFrame with market cap data
        """
        # This is a simplified implementation
        # In a real application, you would need historical shares outstanding data
        
        # Get current shares outstanding
        shares_outstanding = {}
        for symbol in symbols:
            try:
                stock = yf.Ticker(symbol)
                shares = stock.info.get('sharesOutstanding', None)
                if shares:
                    shares_outstanding[symbol] = shares
                else:
                    # Fallback to a default value for demonstration
                    shares_outstanding[symbol] = 1000000
            except:
                # Fallback to a default value
                shares_outstanding[symbol] = 1000000
        
        # Get historical price data
        prices_df = self.get_stock_data(dates[0].strftime('%Y-%m-%d'), dates[-1].strftime('%Y-%m-%d'))
        
        # Calculate market cap (price * shares outstanding)
        market_caps = pd.DataFrame(index=prices_df.index)
        for symbol in symbols:
            if symbol in prices_df.columns and symbol in shares_outstanding:
                market_caps[symbol] = prices_df[symbol] * shares_outstanding[symbol]
        
        return market_caps


def create_index(
    symbols: List[str],
    method: str = 'price_weighted',
    base_value: float = 100.0,
    base_date: str = None,
    start_date: str = None,
    end_date: str = None,
    interval: str = 'daily',
    **kwargs
) -> Tuple[StockIndexCalculator, pd.DataFrame]:
    """
    Convenience function to create and calculate an index in one step.
    
    Args:
        symbols: List of stock ticker symbols
        method: Calculation method
        base_value: Starting value for the index
        base_date: Starting date for the index
        start_date: Period start date
        end_date: Period end date
        interval: Time step interval
        **kwargs: Additional parameters for specific calculation methods
        
    Returns:
        Tuple of (StockIndexCalculator instance, DataFrame with index values)
    """
    calculator = StockIndexCalculator(base_value=base_value, base_date=base_date)
    calculator.add_stocks(symbols)
    
    index_df = calculator.calculate_index(
        method=method,
        start_date=start_date,
        end_date=end_date,
        interval=interval,
        **kwargs
    )
    
    return calculator, index_df


def get_available_methods() -> Dict[str, str]:
    """
    Return a dictionary of available calculation methods with descriptions.
    
    Returns:
        Dictionary of method names and descriptions
    """
    return StockIndexCalculator.CALCULATION_METHODS


def get_available_time_steps() -> List[str]:
    """
    Return a list of available time step intervals.
    
    Returns:
        List of time step intervals
    """
    return StockIndexCalculator.TIME_STEPS


# Example usage
if __name__ == "__main__":
    # Create a tech index with some popular tech stocks
    tech_stocks = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META']
    
    # Calculate a price-weighted index (like DOW) for the past year
    start_date = (datetime.now() - timedelta(days=365)).strftime('%Y-%m-%d')
    
    # Example 1: Simple price-weighted index
    calculator, tech_index = create_index(
        symbols=tech_stocks,
        method='price_weighted',
        base_value=1000.0,
        start_date=start_date,
        interval='daily'
    )
    
    print(f"Current tech index value: {calculator.get_current_index_value():.2f}")
    
    # Example 2: Market-cap weighted index (like S&P 500)
    calculator2, sp_style_index = create_index(
        symbols=tech_stocks,
        method='market_cap_weighted',
        base_value=1000.0,
        start_date=start_date,
        interval='weekly'
    )
    
    print(f"Current market-cap weighted index value: {calculator2.get_current_index_value(method='market_cap_weighted'):.2f}")
    
    # Plot the indices
    import matplotlib.pyplot as plt
    
    plt.figure(figsize=(12, 6))
    plt.plot(tech_index.index, tech_index['index_value'], label='Price-Weighted Tech Index')
    plt.title('Custom Tech Stock Index')
    plt.xlabel('Date')
    plt.ylabel('Index Value')
    plt.legend()
    plt.grid(True)
    plt.show()
