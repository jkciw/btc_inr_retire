"""
Display Components Module

This module contains functions for creating charts, displaying comparisons,
and generating visualizations for the Streamlit app.
"""

import streamlit as st
import pandas as pd
import plotly.express as px
from datetime import datetime
from typing import Dict, Any
from utils import indian_commas
from market_data import get_market_price


def display_scenario_comparison(scenario_results: Dict[str, Any], retirement_year: int):
    """
    Display scenario comparison in Streamlit.

    Args:
        scenario_results: Results from all scenarios
        retirement_year: Target retirement year
    """
    st.markdown("### Scenario Comparison")

    # Get current market rate as base, fallback to 86 if API fails
    try:
        _, current_market_rate = get_market_price()
        base_rate = current_market_rate if current_market_rate > 0 else 86.0
    except:
        base_rate = 86.0  # Conservative fallback

    # Create comparison dataframe
    comparison_data = []
    for scenario_name, results in scenario_results.items():
        params = results['params']
        # Calculate USD/INR at retirement year for reference
        years_to_retirement = retirement_year - datetime.now().year
        usd_inr_retirement = base_rate * \
            (1 + params['usd_appreciation_rate']) ** years_to_retirement
        annual_expenditure_at_retirement = results['annual_expenditure_at_retirement']

        comparison_data.append({
            'Scenario': scenario_name,
            'Inflation Rate': f"{params['inflation_rate']*100:.1f}%",
            'USD/INR Appreciation Rate': f"{params['usd_appreciation_rate']*100:.1f}%",
            'BTC Needed': f"{results['total_bitcoin_needed']:.4f}",
            'USD/INR at retirement': f"₹{usd_inr_retirement:.0f}",
            'Annual Expenditure at retirement (₹)': f"₹{indian_commas(annual_expenditure_at_retirement, 0)}",
            'Retirement Corpus Needed (₹)': f"₹{indian_commas(results['total_inr_needed'], 0)}",
        })

    comparison_df = pd.DataFrame(comparison_data)
    comparison_df.columns = [
        "Scenario",
        "Assumed <br> Inflation<br>Rate",
        "Assumed <br> USD/INR Appreciation<br>Rate",
        "Calculated <br> BTC Needed",
        "Calculated <br> USD/INR at retirement",
        "Calculated <br> Annual Expenditure<br>at retirement (₹)",
        "Calculated <br>Retirement Corpus<br>Needed (₹)"
    ]
    # st.table(display_df)
    styler = comparison_df.style.hide(axis="index")
    styler = styler.set_table_attributes(
        'style="text-align: center; margin: auto;"')
    styled_html = styler.to_html(escape=False)
    st.markdown(styled_html, unsafe_allow_html=True)
