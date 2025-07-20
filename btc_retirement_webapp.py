import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import streamlit as st
import plotly.express as px
from babel.numbers import format_decimal

def bitcoin_power_law_price(days_since_genesis):
    """
    Calculate Bitcoin price using power law model
    Price = 10^-17 × (days since genesis)^5.8
    Returns the 2.5th percentile (conservative estimate) - 24% of trendline
    """
    trendline_price = 10**(-17) * (days_since_genesis ** 5.8)
    # 2.5th percentile is approximately 24% of trendline price
    percentile_2_5_price = trendline_price * 0.24
    return trendline_price, percentile_2_5_price

def calculate_usd_inr_rate(years_from_2025, base_rate=83.0, depreciation_rate=0.027):
    """
    Calculate future USD/INR exchange rate assuming 2.7% annual depreciation
    """
    return base_rate * (1 + depreciation_rate) ** years_from_2025

def run_calculator(current_age, annual_expenditure_inr, avg_inflation, btc_price, retirement_year, cap_gain):
    avg_inf_decimal = avg_inflation / 100.0
    cap_gain_decimal = cap_gain / 100.0
    results = calculate_retirement_bitcoin_needs(current_age, annual_expenditure_inr, avg_inf_decimal, btc_price, retirement_year, cap_gain_decimal)
    breakdown = pd.DataFrame(results['breakdown'])
    breakdown = breakdown.rename(columns={
        'year': 'Year',
        'age': 'Age',
        'expense_inr': 'Expected Annual Expenditure (INR)',
        'btc_price_usd': 'Projected Bitcoin Price (USD)',
        'btc_price_inr': 'Projected Bitcoin Price (INR)',
        'tax_inr': 'Capital Gains Tax Paid (INR)',
        'btc_needed': 'Bitcoin Needed (BTC)',
        'usd_inr_rate': 'USD/INR Rate',
    })
    display_cols = [
        'Year',
        'Age',
        'Expected Annual Expenditure (INR)',
        'Projected Bitcoin Price (USD)',
        'Projected Bitcoin Price (INR)',
        'Capital Gains Tax Paid (INR)',
        'Bitcoin Needed (BTC)',
        'USD/INR Rate'
    ]
    breakdown_view = breakdown[display_cols]
    fmt = {col: indian_commas for col in breakdown_view.columns if col not in ("Year", "Age", "Bitcoin Needed (BTC)")}
    styled = (
    breakdown_view
      .style
      .format(fmt)                                 # apply dec/commas
      .set_table_styles(
          [{"selector": "th", "props": [("white-space", "normal")]}]
      )
    )
    return results, breakdown_view, styled 
# Helper function to format numbers with Indian commas
def indian_commas(x, decimals=2):
    if x is None or x =="":
        return " "
    pattern = f"#,##,##0.{ '0'*decimals }" 
    return format_decimal(x, format=pattern, locale='en_IN')

def calculate_retirement_bitcoin_needs(current_age, annual_expenditure_inr, avg_inflation, btc_purchase_price_usd, retirement_year, cap_gain):
    """
    Calculate Bitcoin requirements for retirement planning till age 90
    """
    # Calculate current year and retirement age
    current_year = datetime.today().year
    years_to_retirement = retirement_year - current_year
    retirement_age = current_age + years_to_retirement

    # Calculate years in retirement (till age 90)
    years_in_retirement = max(0, 90 - retirement_age+1)

    # Bitcoin genesis date: January 3, 2009
    genesis_date = datetime(2009, 1, 3)

    # Calculate exchange rate for retirement year (accounting for INR depreciation)
    years_from_2025 = retirement_year - 2025
    retirement_usd_inr = calculate_usd_inr_rate(years_from_2025)

    # Calculate annual expenditure at retirement (during accumulation phase, use both inflations)
    annual_expenditure_at_retirement = annual_expenditure_inr * (1 + avg_inflation) ** years_to_retirement

    # Convert to USD for retirement year
    annual_expenditure_usd_year_1 = annual_expenditure_at_retirement / retirement_usd_inr

    # Initialize total Bitcoin needed
    total_bitcoin_needed = 0.0
    breakdown = []

    # Calculate Bitcoin needed for each year of retirement
    for year in range(years_in_retirement):
        current_retirement_year = retirement_year + year
        current_date = datetime(current_retirement_year, 1, 1)
        days_since_genesis = (current_date - genesis_date).days

        # Get Bitcoin price for THIS specific year (2.5th percentile)
        _, btc_price_2_5_year = bitcoin_power_law_price(days_since_genesis)

        # Calculate exchange rate for this specific year (INR continues to depreciate)
        years_from_retirement = year
        current_usd_inr = retirement_usd_inr * (1 + 0.027) ** years_from_retirement

        # Calculate annual expenditure for this year (grows with inflation)
        year_expense_usd = annual_expenditure_usd_year_1 * (1 + avg_inflation) ** year

        # Adjust for capital gains tax
        # btc_needed_this_year = (year_expense_usd / 0.7) / btc_price_2_5_year
        if cap_gain > 0:
            y = 1-cap_gain
            btc_needed_this_year = year_expense_usd / (y * btc_price_2_5_year + cap_gain * btc_purchase_price_usd)
        else:
            btc_needed_this_year = year_expense_usd / btc_price_2_5_year
        total_bitcoin_needed += btc_needed_this_year

        sale_proceeds = btc_needed_this_year * btc_price_2_5_year
        cost_basis = btc_needed_this_year * btc_purchase_price_usd
        capital_gain = sale_proceeds - cost_basis
        tax_paid = cap_gain * capital_gain
        net_proceeds = sale_proceeds - tax_paid

        # Store breakdown for analysis
        breakdown.append({
            'year': current_retirement_year,
            'age': retirement_age + year,
            'expense_usd': year_expense_usd,
            'btc_price_usd': btc_price_2_5_year,
            'btc_price_inr': btc_price_2_5_year * current_usd_inr,
            'btc_needed': btc_needed_this_year,
            'usd_inr_rate': current_usd_inr,
            'expense_inr': year_expense_usd * current_usd_inr,
            'sale proceeds': sale_proceeds,
            'cost_basis': cost_basis,
            'capital_gain': capital_gain,
            'tax_usd': tax_paid,
            'tax_inr': tax_paid * current_usd_inr,
            'net_proceeds': net_proceeds
        })

    # Calculate some key metrics for display
    retirement_date = datetime(retirement_year, 1, 1)
    days_since_genesis_retirement = (retirement_date - genesis_date).days
    trendline_price, btc_price_2_5_percentile_retirement = bitcoin_power_law_price(days_since_genesis_retirement)

    # Create detailed CSV export
    # df = pd.DataFrame(breakdown)
    # filename = f'bitcoin_retirement_breakdown_{retirement_year}_corrected.csv'
    # df.to_csv(filename, index=False)
    # print(f"\nDetailed breakdown saved as: {filename}")

    return {
        'current_age': current_age,
        'retirement_age': retirement_age,
        'years_in_retirement': years_in_retirement,
        'total_bitcoin_needed': total_bitcoin_needed,
        'btc_price_2_5_percentile': btc_price_2_5_percentile_retirement,
        'annual_expenditure_at_retirement': annual_expenditure_at_retirement,
        'years_to_retirement': years_to_retirement,
        'retirement_usd_inr': retirement_usd_inr,
        'breakdown': breakdown
    }

st.set_page_config(page_title="How many Bitcoin's do you need to retire ?", layout="wide")  
st.title("How many bitcoins do you need to retire ?")
st.caption("Using Bitcoin Power Law price model")
#--- Get Inputs ---
with st.sidebar: 
    st.header("Configure the following details")
    age = st.slider("Current Age", min_value=18, max_value=89, value=30, step=1)
    exp = st.slider("Current Annual Expenditure (INR)", min_value=100000, max_value=5000000, value=500000, step=50000)
    inf = st.slider("Average Inflation Rate(%)", min_value = 1.0, max_value=20.0, value=8.0, step=0.5)
    btc_price = st.slider("Average purchase price of BTC in retirement stack (USD)", min_value = 1000, max_value=500000, value=70000, step=1000)
    ret_year = st.slider("Retirement year", min_value=2026, max_value = 2060, value = 2040, step = 1)
    cap_gain = st.slider("Capital Gains Tax on BTC sale (%)", min_value=0, max_value=50, value=30, step=1)
    run = st.button("Calculate")
# -- Run the Calculator -- #
if run:
    summary, breakdown, styled = run_calculator(age, exp, inf, btc_price, ret_year, cap_gain)

    #Show BTC required
    st.subheader("BTC required for retirement")
    st.markdown(f"<h1 style='font-size: 28px; font-weight: bold; color: #1f77b4;'>{summary['total_bitcoin_needed']:.4f} BTC</h1>", unsafe_allow_html=True)
    # Display Table
    st.subheader("Detailed Yearly Breakdown")
    st.dataframe(styled, use_container_width=True)
    
