"""
Utility Functions Module

This module contains helper functions for formatting, data manipulation,
and other utility operations used throughout the application.
"""

from babel.numbers import format_decimal
import pandas as pd
from typing import Union


def indian_commas(x: Union[float, int, None], decimals: int = 2) -> str:
    """
    Format numbers with Indian comma notation.
    
    Args:
        x: Number to format (can be None)
        decimals: Number of decimal places
        
    Returns:
        Formatted string with Indian comma notation
    """
    if x is None or x == "":
        return " "
    pattern = f"#,##,##0.{'0'*decimals}"
    return format_decimal(x, format=pattern, locale='en_IN')


def format_breakdown_dataframe(breakdown_df: pd.DataFrame) -> pd.DataFrame:
    """
    Format breakdown dataframe for display with Indian commas and proper decimals.
    
    Args:
        breakdown_df: Raw breakdown dataframe
        
    Returns:
        Formatted dataframe for display
    """
    display_df = breakdown_df.copy()
    
    display_df['Expense (INR)'] = display_df['Expense (INR)'].apply(
        lambda x: f"₹{indian_commas(x, 2)}")
    
    display_df['BTC Price (USD)'] = display_df['BTC Price (USD)'].apply(
        lambda x: f"${x:,.2f}")
    
    display_df['BTC Price (INR)'] = display_df['BTC Price (INR)'].apply(
        lambda x: f"₹{indian_commas(x, 2)}")
    
    display_df['USD/INR Rate'] = display_df['USD/INR Rate'].apply(
        lambda x: f"₹{indian_commas(x, 2)}")
    
    display_df['BTC Needed'] = display_df['BTC Needed'].apply(
        lambda x: f"{x:.6f}")
    
    return display_df