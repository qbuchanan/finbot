Sector Scan retrieves top companies in a specified sector based on customizable ranking criteria. Here's what the function offers:

### Main Features:

1. **Flexible Sector/Industry Selection**
   - Works with any sector or industry name (Technology, Financial, Healthcare, etc.)
   - Handles fuzzy matching for sector/industry names

2. **Customizable Ranking Criteria**
   - `market_cap` - Largest companies by market capitalization
   - `debt_to_equity` - Companies with lowest/highest debt-to-equity ratios
   - `return_on_equity` - Highest/lowest return on equity
   - `dividend_yield` - Highest/lowest dividend yield
   - `profit_margin` - Most/least profitable companies
   - `pe_ratio` - Lowest/highest price-to-earnings ratio
   - `debt_to_ebitda` - Companies with best/worst debt levels relative to earnings
   - Plus many more metrics (16 total options)

3. **Adjustable Result Size**
   - Get top 5, 10, 20, or any number of companies

4. **Customizable Sort Order**
   - Ascending (small values first) or descending (large values first)
   - Default sort direction is appropriate for each metric (e.g., low is better for debt ratios)

### How to Use:

```python
# Get top 10 tech companies by market cap
top_tech = get_top_companies_by_sector("Technology", num_companies=10, criteria="market_cap")

# Get 5 energy companies with lowest debt-to-equity ratio
low_debt_energy = get_top_companies_by_sector("Energy", num_companies=5, 
                                             criteria="debt_to_equity", ascending=True)

# Get 5 financial companies with highest return on equity
high_roe_finance = get_top_companies_by_sector("Financial", num_companies=5, 
                                              criteria="return_on_equity")
```

### Behind the Scenes:

The function uses several strategies to overcome limitations in the yfinance API:

1. Attempts to download company listings from NASDAQ
2. Falls back to S&P 500 components from Wikipedia if needed
3. Uses predefined ticker lists for major indices as a last resort
4. Employs multithreading for efficient data collection
5. Handles missing data gracefully

The returned data is formatted as a clean pandas DataFrame with relevant financial metrics included for easy analysis.