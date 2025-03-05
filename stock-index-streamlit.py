import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import yfinance as yf
from typing import List, Dict, Tuple
import time
import base64
from io import BytesIO

# Import the StockIndexCalculator class and helper functions
# For deployment, you would import these from your module
# from stock_index_calculator import StockIndexCalculator, create_index, get_available_methods, get_available_time_steps

# For demonstration, I'm including the required code directly
# ------------------------------------------------------------
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
            
        data = yf.download(symbols, start=start_date, end=end_date, interval=yf_interval, progress=False)
        
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
        
        # Also store the individual component prices
        index_df = pd.concat([index_df, prices_df], axis=1)
        
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

# ------------------------------------------------------------

# Define some stock collections for easy selection
STOCK_COLLECTIONS = {
    "Tech Giants": ["AAPL", "MSFT", "GOOGL", "AMZN", "META"],
    "Financial Sector": ["JPM", "BAC", "WFC", "C", "GS"],
    "Healthcare": ["JNJ", "PFE", "MRK", "UNH", "ABBV"],
    "Consumer Discretionary": ["AMZN", "HD", "NKE", "SBUX", "MCD"],
    "Energy": ["XOM", "CVX", "COP", "SLB", "EOG"],
    "Dow Jones 10": ["AAPL", "MSFT", "JNJ", "V", "PG", "HD", "UNH", "JPM", "MA", "DIS"],
    "S&P 10": ["AAPL", "MSFT", "AMZN", "GOOGL", "META", "NVDA", "BRK-B", "UNH", "JNJ", "JPM"]
}

# Set page configuration
st.set_page_config(
    page_title="Custom Stock Index Builder",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Define functions for the Streamlit UI
def validate_stock_symbols(symbols):
    """Validate stock symbols by trying to fetch their data"""
    valid_symbols = []
    invalid_symbols = []
    
    with st.spinner(f"Validating {len(symbols)} stock symbols..."):
        progress_bar = st.progress(0)
        for i, symbol in enumerate(symbols):
            try:
                # Try to fetch some minimal data to validate the symbol
                data = yf.download(symbol, period="1d", progress=False)
                if not data.empty:
                    valid_symbols.append(symbol)
                else:
                    invalid_symbols.append(symbol)
            except:
                invalid_symbols.append(symbol)
            progress_bar.progress((i + 1) / len(symbols))
    
    return valid_symbols, invalid_symbols

def format_large_number(number):
    """Format large numbers with K, M, B suffixes"""
    if number >= 1_000_000_000:
        return f"{number / 1_000_000_000:.2f}B"
    elif number >= 1_000_000:
        return f"{number / 1_000_000:.2f}M"
    elif number >= 1_000:
        return f"{number / 1_000:.2f}K"
    else:
        return f"{number:.2f}"

def plot_index_performance(index_df, title, compare_with=None):
    """Create a plotly figure for the index performance"""
    fig = go.Figure()
    
    # Plot the custom index
    fig.add_trace(
        go.Scatter(
            x=index_df.index,
            y=index_df['index_value'],
            mode='lines',
            name='Custom Index',
            line=dict(color='royalblue', width=3)
        )
    )
    
    # Plot comparison indices if provided
    if compare_with:
        for symbol, data in compare_with.items():
            # Normalize to match starting point of custom index
            normalized = data / data.iloc[0] * index_df['index_value'].iloc[0]
            fig.add_trace(
                go.Scatter(
                    x=data.index,
                    y=normalized,
                    mode='lines',
                    name=symbol,
                    line=dict(width=2, dash='dash')
                )
            )
    
    # Customize layout
    fig.update_layout(
        title=title,
        xaxis_title='Date',
        yaxis_title='Index Value',
        template='plotly_white',
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        ),
        margin=dict(l=20, r=20, t=60, b=20),
        height=500
    )
    
    return fig

def plot_component_weights(index_df, method, title):
    """Create a plotly figure showing the weights of index components"""
    if method == 'price_weighted':
        # For price-weighted, weight is proportional to price
        latest_data = index_df.iloc[-1].drop('index_value')
        weights = latest_data / latest_data.sum()
    elif method == 'equal_weighted':
        # For equal-weighted, all weights are the same
        symbols = index_df.columns.drop('index_value')
        weights = pd.Series([1/len(symbols)] * len(symbols), index=symbols)
    else:
        # For other methods, use approximate weights
        latest_data = index_df.iloc[-1].drop('index_value')
        
        if method == 'market_cap_weighted':
            # Get approximate market caps
            market_caps = {}
            for symbol in latest_data.index:
                try:
                    stock = yf.Ticker(symbol)
                    market_cap = stock.info.get('marketCap', 0)
                    if market_cap:
                        market_caps[symbol] = market_cap
                    else:
                        # Fallback: use price as a proxy (not accurate but visual only)
                        market_caps[symbol] = latest_data[symbol] * 1000000
                except:
                    # Fallback
                    market_caps[symbol] = latest_data[symbol] * 1000000
            
            weights_series = pd.Series(market_caps)
            weights = weights_series / weights_series.sum()
        else:
            # Default to price-weighted for visual
            weights = latest_data / latest_data.sum()
    
    # Create the pie chart
    fig = px.pie(
        values=weights.values,
        names=weights.index,
        title=title,
        hole=0.4,
    )
    
    fig.update_layout(
        margin=dict(l=20, r=20, t=60, b=20),
        height=400
    )
    
    return fig

def plot_component_performance(index_df, title):
    """Create a plotly figure for individual component performance"""
    # Normalize all components to starting value of 100
    components = index_df.drop('index_value', axis=1)
    normalized = components.div(components.iloc[0]) * 100
    
    fig = go.Figure()
    
    # Add a trace for each component
    for column in normalized.columns:
        fig.add_trace(
            go.Scatter(
                x=normalized.index,
                y=normalized[column],
                mode='lines',
                name=column
            )
        )
    
    # Customize layout
    fig.update_layout(
        title=title,
        xaxis_title='Date',
        yaxis_title='Normalized Value (Base 100)',
        template='plotly_white',
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        ),
        margin=dict(l=20, r=20, t=60, b=20),
        height=500
    )
    
    return fig

def export_to_excel(index_df, method, index_name):
    """Export index data to Excel"""
    # Create a BytesIO object
    output = BytesIO()
    
    # Create a Pandas Excel writer using the BytesIO object
    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        # Write the index data
        index_df.to_excel(writer, sheet_name='Index Values')
        
        # Create a summary sheet
        summary_data = {
            'Index Name': [index_name],
            'Calculation Method': [method],
            'Start Date': [index_df.index[0].strftime('%Y-%m-%d')],
            'End Date': [index_df.index[-1].strftime('%Y-%m-%d')],
            'Starting Value': [index_df['index_value'].iloc[0]],
            'Ending Value': [index_df['index_value'].iloc[-1]],
            'Change': [f"{((index_df['index_value'].iloc[-1] / index_df['index_value'].iloc[0]) - 1) * 100:.2f}%"],
            'Components': [', '.join(index_df.columns.drop('index_value'))]
        }
        pd.DataFrame(summary_data).to_excel(writer, sheet_name='Summary')
        
        # Get the workbook and add a format
        workbook = writer.book
        percent_format = workbook.add_format({'num_format': '0.00%'})
        
        # Add a components sheet
        components = index_df.drop('index_value', axis=1)
        components.to_excel(writer, sheet_name='Component Prices')
        
        # Create a performance sheet
        normalized = components.div(components.iloc[0]) - 1  # Convert to percent change
        normalized.to_excel(writer, sheet_name='Component Performance')
        
        # Format the percent sheet
        worksheet = writer.sheets['Component Performance']
        for col_num, column in enumerate(normalized.columns):
            worksheet.set_column(col_num + 1, col_num + 1, None, percent_format)
    
    # Get the value of the BytesIO object
    excel_data = output.getvalue()
    
    return excel_data


# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1E3A8A;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.8rem;
        font-weight: bold;
        color: #2563EB;
        margin-top: 2rem;
        margin-bottom: 1rem;
    }
    .card {
        padding: 1.5rem;
        border-radius: 0.5rem;
        background-color: #F9FAFB;
        border: 1px solid #E5E7EB;
        margin-bottom: 1rem;
    }
    .metric-label {
        font-size: 1rem;
        color: #6B7280;
        margin-bottom: 0.25rem;
    }
    .metric-value {
        font-size: 2rem;
        font-weight: bold;
        color: #1E3A8A;
    }
    .metric-change-positive {
        font-size: 1rem;
        color: #059669;
    }
    .metric-change-negative {
        font-size: 1rem;
        color: #DC2626;
    }
    .tooltip {
        position: relative;
        display: inline-block;
        cursor: help;
    }
    .info-icon {
        color: #6B7280;
        font-size: 1rem;
        margin-left: 0.5rem;
    }
</style>
""", unsafe_allow_html=True)

# App title
st.markdown("<div class='main-header'>🏛️ Custom Stock Index Builder</div>", unsafe_allow_html=True)
st.markdown("Build, analyze, and compare custom stock market indices using different methodologies.")

# Sidebar for inputs
with st.sidebar:
    st.markdown("### Index Configuration")
    
    # Index name
    index_name = st.text_input("Index Name", "My Custom Index")
    
    # Stock selection method
    selection_method = st.radio(
        "Stock Selection Method",
        ["Pre-defined Collections", "Manual Entry"]
    )
    
    if selection_method == "Pre-defined Collections":
        # Pre-defined stock collections
        collection = st.selectbox(
            "Select Stock Collection",
            list(STOCK_COLLECTIONS.keys())
        )
        selected_stocks = STOCK_COLLECTIONS[collection]
        
        # Show selected stocks
        st.markdown(f"**Selected Stocks ({len(selected_stocks)}):**")
        st.text(", ".join(selected_stocks))
        
    else:  # Manual entry
        # Stock symbols input
        stock_input = st.text_area(
            "Enter Stock Symbols (comma or space separated)",
            "AAPL, MSFT, GOOGL, AMZN, JPM"
        )
        
        # Parse and validate stock symbols
        raw_symbols = [s.strip().upper() for s in stock_input.replace(",", " ").split() if s.strip()]
        selected_stocks = list(dict.fromkeys(raw_symbols))  # Remove duplicates while preserving order
    
    # Method selection
    available_methods = get_available_methods()
    method = st.selectbox(
        "Calculation Method",
        list(available_methods.keys()),
        format_func=lambda x: x.replace("_", " ").title()
    )
    
    # Show method description
    st.info(available_methods[method])
    
    # Base value
    base_value = st.number_input("Base Value", min_value=1.0, value=100.0, step=10.0)
    
    # Time parameters
    st.markdown("### Time Parameters")
    
    # Date range selection
    today = datetime.now().date()
    
    # Default to 1 year look-back
    default_start = today - timedelta(days=365)
    
    date_range = st.date_input(
        "Date Range",
        [default_start, today],
        min_value=datetime(2000, 1, 1).date(),
        max_value=today
    )
    
    if len(date_range) == 2:
        start_date = date_range[0].strftime('%Y-%m-%d')
        end_date = date_range[1].strftime('%Y-%m-%d')
    else:
        start_date = default_start.strftime('%Y-%m-%d')
        end_date = today.strftime('%Y-%m-%d')
    
    # Time interval
    available_steps = get_available_time_steps()
    interval = st.selectbox(
        "Time Interval",
        available_steps,
        index=0,  # Default to daily
        format_func=lambda x: x.capitalize()
    )
    
    # Comparison options
    st.markdown("### Comparison Options")
    
    # Benchmark indices
    compare_options = {
        "None": [],
        "S&P 500 (^GSPC)": ["^GSPC"],
        "Dow Jones (^DJI)": ["^DJI"],
        "NASDAQ (^IXIC)": ["^IXIC"],
        "S&P 500 & Dow Jones": ["^GSPC", "^DJI"],
        "All Major Indices": ["^GSPC", "^DJI", "^IXIC"]
    }
    
    comparison = st.selectbox(
        "Compare With",
        list(compare_options.keys())
    )
    
    comparison_indices = compare_options[comparison]
    
    # Calculate button
    calculate = st.button("🔢 Calculate Index", type="primary", use_container_width=True)

# Main content
if calculate:
    # Validate the symbols first
    with st.spinner("Validating stock symbols..."):
        valid_symbols, invalid_symbols = validate_stock_symbols(selected_stocks)
    
    if invalid_symbols:
        st.warning(f"The following symbols could not be validated: {', '.join(invalid_symbols)}")
        selected_stocks = valid_symbols
    
    