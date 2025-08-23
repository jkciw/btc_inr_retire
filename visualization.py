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
            (1 + params['depreciation_rate']) ** years_to_retirement
        annual_expenditure_at_retirement = results['annual_expenditure_at_retirement']

        comparison_data.append({
            'Scenario': scenario_name,
            'BTC Needed': f"{results['total_bitcoin_needed']:.4f}",
            'Retirement Corpus Needed (₹)': f"₹{indian_commas(results['total_inr_needed'], 0)}",
            'Inflation Rate': f"{params['inflation_rate']*100:.1f}%",
            'Annual Expenditure at retirement (₹)': f"₹{indian_commas(annual_expenditure_at_retirement, 0)}",
            'USD Depreciation Rate': f"{params['depreciation_rate']*100:.1f}%",
            'USD/INR at retirement': f"₹{usd_inr_retirement:.0f}",
        })

    comparison_df = pd.DataFrame(comparison_data)
    st.dataframe(comparison_df, use_container_width=True, hide_index=True)

    # Recommendations
    conservative_btc = scenario_results['conservative']['total_bitcoin_needed']
    extreme_btc = scenario_results['extreme']['total_bitcoin_needed']

    st.markdown("### Recommendations")
    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(
            label="Primary Target",
            value=f"{conservative_btc:.4f} BTC",
            help="Conservative scenario - recommended planning baseline"
        )

    with col2:
        st.metric(
            label="Stress Test",
            value=f"{extreme_btc:.4f} BTC",
            help="Extreme scenario - worst-case protection"
        )

    with col3:
        st.metric(
            label="Extra Buffer",
            value=f"{extreme_btc - conservative_btc:.4f} BTC",
            help="Additional BTC needed for extreme scenario protection"
        )
