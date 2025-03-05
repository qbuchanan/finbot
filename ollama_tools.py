import streamlit as st
import pandas as pd
import numpy as np
import ollama
import yfinance as yf
from typing import List, Dict, Any, Callable, Optional
from dataclasses import dataclass
from json import dumps
import datetime


def get_ticker(company_name):
    try:
        stock = yf.Ticker(company_name)
        return stock.ticker  # Returns the exact ticker if found
    except Exception as e:
        return f"Error: {e}"

def get_stock_price(symbol: str) -> float:
    """
    Get the current stock price of a given stock symbol using the yfinance library
    Args:
        symbol (str): The stock symbol to get the price for
    Returns:    
        float: The current stock price
    """
    ticker = yf.Ticker(symbol)
    temp =  yf.Ticker(symbol).history(period='1d').Close.iloc[0]
    return ticker.info.get('regularMarketPrice') or ticker.fast_info.last_price

def get_current_time() -> str:
    """
    Get the current time
    Returns:
        str: The current time
    """
    return datetime.now().strftime("%H:%M:%S")

def get_current_date() -> str:
    """
    Get the current date            
    Returns:
        str: The current date
    """   
    return datetime.now().strftime("%Y-%m-%d")


available_functions:Dict[str, Callable] = { 
    "get_ticker": get_ticker,
    "get_stock_price": get_stock_price,
    "get_current_time": get_current_time,
    "get_current_date": get_current_date
}   



def call_tools(response: Dict[str, Any]) -> str:
    results = []
    if response.message.tool_calls:
        for tool_call in response.message.tool_calls:
            if function_to_call := available_functions.get(tool_call.function.name):
                result = function_to_call(**tool_call.function.arguments)
                results.append({  # turn this into a data class
                    'role' : 'tool',
                    'name': tool_call.function.name,
                    'arguments': tool_call.function.arguments,
                    'content': f"{result}"
                })
                print(f'Function {tool_call.function.name} called with args {tool_call.function.arguments} and result {result}')    
            else:
                tool_call.result = f"Function {tool_call.function_name} not found"
                print(f"Function {tool_call.function_name} not found")
    return results


def test_prompt():
    prompt = f'What is the current stock price of {st.session_state.company}'
    messages=[{'role': 'user', 'content': prompt}]
    response = ollama.chat(
        'llama3.2',
        messages,
        tools=[get_stock_price, get_ticker, get_current_time, get_current_date]
    )
    if response.message.tool_calls:
        tool_results =  call_tools(response)
        messages.append({
            'role': 'user',
            'content': f"Please use the following tool results in your answer"
        })
        messages.extend(tool_results)
        response = ollama.chat(
            'llama3.2',
            messages)
    return response  
    


st.write('ollama tools test')
st.sidebar.title("Stocks")
st.session_state.company = st.sidebar.text_input("Enter a stock ticker", key="ticker")
if st.sidebar.button("Get stock price"):
    response = test_prompt()
    st.write(response)
