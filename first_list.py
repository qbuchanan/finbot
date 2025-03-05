"""
# My first streamlit app
Here's our first attempt at using data to create a table:
"""

import streamlit as st
import pandas as pd
import numpy as np

st.write("""
# My first streamlit app
Here's our first attempt at using data to create a table:
""")

if st.sidebar.checkbox('Show table'):
    df = pd.DataFrame({
        "first column": [1, 2, 3, 4],
        "second column": [10, 20, 30, 40]
    })

    st.table(df.style.highlight_max(axis=0))

if st.sidebar.checkbox('Show dataframe'):
    chart_data = pd.DataFrame(
        np.random.randn(20, 3),
        columns=['a', 'b', 'c'])

    "# next chart"

    st.line_chart(chart_data)

"# Next up a map"
map_data = pd.DataFrame(
    np.random.randn(1000, 2) / [50, 50] + [37.76, -122.4],
    columns=['lat', 'lon'])

st.map(map_data)

x = st.sidebar.slider('x')
st.write(x, 'squared is', x * x)    

"whats your name"
st.sidebar.text_input("Your name", key="name")

# You can access the value at any point with:
st.session_state.name

left_column, right_column = st.columns(2)
# You can use a column just like st.sidebar:
left_column.button('Press me!')

# Or even better, call Streamlit functions inside a "with" block:
with right_column:
    chosen = st.radio(
        'Sorting hat',
        ("Gryffindor", "Ravenclaw", "Hufflepuff", "Slytherin"))
    st.write(f"You are in {chosen} house!")