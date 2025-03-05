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

# Import the StockIndexCalculator class and helper functions from your module
from stock_index_calculator import StockIndexCalculator, create_index, get_available_methods, get_available_time_steps

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
    
    if not valid_symbols:
        st.error("No valid stock symbols found. Please enter valid stock symbols and try again.")
    else:
        # Calculate the index
        with st.spinner(f"Calculating {method.replace('_', ' ')} index for {len(valid_symbols)} stocks..."):
            try:
                # Create the index calculator and calculate the index
                calculator, index_df = create_index(
                    symbols=valid_symbols,
                    method=method,
                    base_value=base_value,
                    start_date=start_date,
                    end_date=end_date,
                    interval=interval
                )
                
                # Get comparison data if requested
                comparison_data = {}
                if comparison_indices:
                    for idx in comparison_indices:
                        try:
                            comp_data = yf.download(idx, start=start_date, end=end_date, interval='1d' if interval == 'daily' else '1wk', progress=False)
                            comparison_data[idx] = comp_data['Close']
                        except Exception as e:
                            st.warning(f"Could not fetch comparison data for {idx}: {str(e)}")
                
                # Display results
                st.markdown(f"<div class='sub-header'>📊 {index_name} Performance</div>", unsafe_allow_html=True)
                
                # Key metrics
                start_value = index_df['index_value'].iloc[0]
                end_value = index_df['index_value'].iloc[-1]
                percent_change = ((end_value / start_value) - 1) * 100
                
                # Create three columns for metrics
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.markdown("<div class='card'>", unsafe_allow_html=True)
                    st.markdown("<div class='metric-label'>Starting Value</div>", unsafe_allow_html=True)
                    st.markdown(f"<div class='metric-value'>{start_value:.2f}</div>", unsafe_allow_html=True)
                    st.markdown("</div>", unsafe_allow_html=True)
                
                with col2:
                    st.markdown("<div class='card'>", unsafe_allow_html=True)
                    st.markdown("<div class='metric-label'>Current Value</div>", unsafe_allow_html=True)
                    st.markdown(f"<div class='metric-value'>{end_value:.2f}</div>", unsafe_allow_html=True)
                    st.markdown("</div>", unsafe_allow_html=True)
                
                with col3:
                    st.markdown("<div class='card'>", unsafe_allow_html=True)
                    st.markdown("<div class='metric-label'>Total Return</div>", unsafe_allow_html=True)
                    
                    if percent_change >= 0:
                        st.markdown(f"<div class='metric-value'><span class='metric-change-positive'>+{percent_change:.2f}%</span></div>", unsafe_allow_html=True)
                    else:
                        st.markdown(f"<div class='metric-value'><span class='metric-change-negative'>{percent_change:.2f}%</span></div>", unsafe_allow_html=True)
                    
                    st.markdown("</div>", unsafe_allow_html=True)
                
                # Index performance chart
                st.plotly_chart(
                    plot_index_performance(
                        index_df, 
                        f"{index_name} Performance", 
                        comparison_data
                    ),
                    use_container_width=True
                )
                
                # Component weights and performance
                col1, col2 = st.columns(2)
                
                with col1:
                    st.plotly_chart(
                        plot_component_weights(
                            index_df, 
                            method, 
                            f"{index_name} Component Weights"
                        ),
                        use_container_width=True
                    )
                
                with col2:
                    st.plotly_chart(
                        plot_component_performance(
                            index_df, 
                            f"{index_name} Component Performance"
                        ),
                        use_container_width=True
                    )
                
                # Data table
                st.markdown("<div class='sub-header'>📋 Index Data</div>", unsafe_allow_html=True)
                
                # Toggle for data display
                show_data = st.checkbox("Show Raw Data Table")
                if show_data:
                    st.dataframe(index_df, use_container_width=True)
                
                # Export options
                st.markdown("<div class='sub-header'>📥 Export Options</div>", unsafe_allow_html=True)
                
                # Create Excel export
                excel_data = export_to_excel(index_df, method, index_name)
                
                # Create download button
                st.download_button(
                    label="📥 Download Excel Report",
                    data=excel_data,
                    file_name=f"{index_name.replace(' ', '_')}_{datetime.now().strftime('%Y%m%d')}.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                )
                
            except Exception as e:
                st.error(f"Error calculating index: {str(e)}")
                st.exception(e)

# Documentation section
with st.expander("📚 How to Use This Tool"):
    st.markdown("""
    ### Custom Stock Index Builder
    
    This tool allows you to create and analyze custom stock indices similar to well-known indices like the S&P 500 or Dow Jones Industrial Average.
    
    #### Steps to create your index:
    
    1. **Configure your index**:
       - Choose a name for your index
       - Select stocks either from pre-defined collections or by entering symbols manually
       - Choose a calculation method (price-weighted, market-cap weighted, etc.)
       - Set a base value for your index
    
    2. **Set time parameters**:
       - Choose the date range for analysis
       - Select the time interval (daily, weekly, monthly, etc.)
    
    3. **Choose comparison options**:
       - Optionally compare your index with standard market indices
    
    4. **Calculate and analyze**:
       - Press the "Calculate Index" button
       - View performance charts, component weights, and detailed data
       - Export results to Excel for further analysis
    
    #### Calculation Methods:
    
    - **Price-weighted (like Dow Jones)**: Stocks with higher prices have more influence
    - **Market-cap weighted (like S&P 500)**: Larger companies have more influence
    - **Equal-weighted**: All stocks have the same influence regardless of price or size
    - **Float-adjusted**: Based on market cap but considers only publicly available shares
    - **Fundamental-weighted**: Based on fundamental metrics (earnings, book value, etc.)
    """)

# Footer
st.markdown("---")
st.markdown("Built with Streamlit • Data provided by Yahoo Finance")
