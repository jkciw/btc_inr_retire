import numpy as np
import pandas as pd
from datetime import datetime, time
import streamlit as st
import plotly.express as px
from babel.numbers import format_decimal
from requests.exceptions import HTTPError
import requests


# Function to calculate Bitcoin price using power law model
def bitcoin_power_law_price(days_since_genesis):
    """
    Price = 10^-17 × (days since genesis)^5.8
    Returns the 2.5th percentile (conservative estimate) - 24% of trendline
    """
    trendline_price = 10**(-17) * (days_since_genesis ** 5.8)
    # 2.5th percentile is approximately 24% of trendline price
    percentile_2_5_price = trendline_price * 0.24
    return trendline_price, percentile_2_5_price

# Function to fetch current Bitcoin market price and USD/INR exchange rate


def get_market_price_coinpaprika():
    try:
        # CoinPaprika has very generous free limits
        btc_url = 'https://api.coinpaprika.com/v1/tickers/btc-bitcoin'
        response = requests.get(btc_url, timeout=10)
        response.raise_for_status()
        btc_data = response.json()

        btc_price = btc_data.get('quotes', {}).get('USD', {}).get('price', 0.0)

        # Fetch the latest USD/INR exchange rate
        url = 'https://cdn.jsdelivr.net/npm/@fawazahmed0/currency-api@latest/v1/currencies/usd.json'
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        data = response.json()
        if 'usd' not in data or 'inr' not in data['usd']:
            st.warning("Couldn't fetch USD/INR exchange rate")
            usd_inr = 0.0
        else:
            usd_inr = data['usd']['inr']

        return float(btc_price), float(usd_inr)

    except Exception as e:
        st.error(f"Error fetching market data: {e}")
        return 0.0, 0.0


def get_market_price_binance():
    try:
        # Binance allows 1200 calls/minute without API key
        btc_url = 'https://api.binance.com/api/v3/ticker/price'
        btc_params = {'symbol': 'BTCUSDT'}
        response = requests.get(btc_url, params=btc_params, timeout=10)
        response.raise_for_status()
        btc_data = response.json()

        btc_price = float(btc_data.get('price', 0.0))

        # Fetch the latest USD/INR exchange rate
        url = 'https://cdn.jsdelivr.net/npm/@fawazahmed0/currency-api@latest/v1/currencies/usd.json'
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        data = response.json()
        if 'usd' not in data or 'inr' not in data['usd']:
            st.warning("Couldn't fetch USD/INR exchange rate")
            usd_inr = 0.0
        else:
            usd_inr = data['usd']['inr']

        return btc_price, usd_inr

    except Exception as e:
        return 0.0, 0.0


@st.cache_data(ttl=600)  # Cache for 10 minutes
def get_cached_market_price():
    return get_market_price_with_retry()


def get_market_price_with_retry():
    def fetch_with_retry(url, params=None, max_retries=5):
        retries = 0
        while retries < max_retries:
            try:
                response = requests.get(url, params=params, timeout=10)
                response.raise_for_status()
                return response.json()
            except HTTPError:
                if response.status_code == 429:
                    # Exponential backoff: 1,2,4,8,16 seconds
                    wait_time = (2 ** retries) * 1
                    st.warning(
                        f"Rate limit hit. Waiting {wait_time} seconds...")
                    time.sleep(wait_time)
                    retries += 1
                    continue
                else:
                    raise
            except Exception:
                raise
        raise Exception(
            f"Failed after {max_retries} retries due to rate limiting.")

    try:
        # Fetch Bitcoin price with retry logic
        btc_url = 'https://api.coingecko.com/api/v3/simple/price'
        btc_params = {'ids': 'bitcoin', 'vs_currencies': 'usd'}
        btc_data = fetch_with_retry(btc_url, btc_params)

        btc_price = btc_data.get('bitcoin', {}).get('usd', 0.0)

        # Add delay between requests
        time.sleep(2)
        # Fetch the latest USD/INR exchange rate
        url = 'https://cdn.jsdelivr.net/npm/@fawazahmed0/currency-api@latest/v1/currencies/usd.json'
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        data = response.json()
        if 'usd' not in data or 'inr' not in data['usd']:
            st.warning("Couldn't fetch USD/INR exchange rate")
            usd_inr = 0.0
        else:
            usd_inr = data['usd']['inr']
        return float(btc_price), float(usd_inr)
    except Exception as e:
        st.error(f"Error fetching market data: {e}")
        return 0.0, 0.0


def get_market_price():
    """Try multiple free APIs in order of preference"""

    # API sources in order of preference
    apis = [
        ('CoinPaprika', get_market_price_coinpaprika),
        ('Binance', get_market_price_binance),
        ('CoinGecko', get_market_price_with_retry)  # CoinGecko as fallback
    ]

    for api_name, api_func in apis:
        try:
            btc_price, usd_inr = api_func()
            if btc_price > 0 and usd_inr > 0:
                st.success(f"✅ Market data fetched from {api_name}")
                return btc_price, usd_inr
        except Exception as e:
            st.warning(f"⚠️ {api_name} failed: {str(e)}")
            continue

    # If all APIs fail, use conservative fallbacks
    st.error("❌ All APIs failed. Using fallback values.")
    return 100000.0, 87.0  # Conservative fallback values
# Function to calculate future USD/INR exchange rate based on depreciation


def calculate_usd_inr_rate(years_from_current, base_rate=83.0, depreciation_rate=0.045):
    """
    Calculate future USD/INR exchange rate assuming by default 4.5% annual depreciation
    """
    return base_rate * (1 + depreciation_rate) ** years_from_current

# Helper function to format numbers with Indian commas


def indian_commas(x, decimals=2):
    if x is None or x == "":
        return " "
    pattern = f"#,##,##0.{'0'*decimals}"
    return format_decimal(x, format=pattern, locale='en_IN')

### Function to calculate Bitcoin requirements for retirement based on three scenarios ###
    # Optimistic : Inflation 6.5% and USD/INR Depreciation 3.0%
    # Conservative : Inflation 8.0% and USD/INR Depreciation 4.5%
    # Extreme : Inflation 10.0% and USD/INR Depreciation 6.0%


def calculate_retirement_bitcoin_needs_scenarios(current_age, annual_expenditure_inr, retirement_year):
    """
    Calculate Bitcoin requirements for all three scenarios
    """
    # Calculate basic parameters
    current_year = datetime.now().year
    years_to_retirement = retirement_year - current_year
    retirement_age = current_age + years_to_retirement
    years_in_retirement = max(0, 90 - retirement_age)

    if years_in_retirement <= 0:
        return None

    # Bitcoin genesis date
    genesis_date = datetime(2009, 1, 3)

    # Calculate for all three scenarios
    scenario_results = {}

    for scenario_name in ['optimistic', 'conservative', 'extreme']:
        scenario_params = get_scenario_parameters(scenario_name)
        depreciation_rate = scenario_params['depreciation_rate']
        inflation_rate = scenario_params['inflation_rate']

        # Calculate scenario-specific results
        result = calculate_single_scenario(
            current_age, retirement_year, years_to_retirement, years_in_retirement,
            annual_expenditure_inr, depreciation_rate, inflation_rate, genesis_date
        )

        scenario_results[scenario_name] = {
            **result,
            'params': scenario_params
        }

    return scenario_results

# Function to provide parameters for different scenarios


def get_scenario_parameters(scenario_type):
    """
    Returns depreciation and inflation parameters for different scenarios
    """
    scenarios = {
        'optimistic': {
            'depreciation_rate': 0.030,
            'inflation_rate': 0.065,
            'description': 'Best-case economic conditions',
            'color': '#28a745'  # Green
        },
        'conservative': {
            'depreciation_rate': 0.045,
            'inflation_rate': 0.080,
            'description': 'Prudent retirement planning baseline',
            'color': '#ffc107'  # Yellow
        },
        'extreme': {
            'depreciation_rate': 0.060,
            'inflation_rate': 0.100,
            'description': 'Worst-case stress testing',
            'color': '#dc3545'  # Red
        }
    }
    return scenarios.get(scenario_type)


def calculate_single_scenario(current_age, retirement_year, years_to_retirement,
                              years_in_retirement, annual_expenditure_inr,
                              depreciation_rate, inflation_rate, genesis_date):
    """
    Calculate Bitcoin needs for a single scenario
    """
    # Calculate exchange rate for retirement year
    years_from_current = retirement_year - datetime.now().year
    retirement_usd_inr = calculate_usd_inr_rate(
        years_from_current, depreciation_rate=depreciation_rate)

    # Calculate annual expenditure at retirement
    annual_expenditure_at_retirement = annual_expenditure_inr * \
        (1 + inflation_rate) ** years_to_retirement
    annual_expenditure_usd_year_1 = annual_expenditure_at_retirement / retirement_usd_inr

    # Calculate Bitcoin needed for each year
    total_bitcoin_needed = 0.0
    breakdown = []

    for year in range(years_in_retirement):
        current_retirement_year = retirement_year + year
        current_date = datetime(current_retirement_year, 1, 1)
        days_since_genesis = (current_date - genesis_date).days

        # Get Bitcoin price for this year (2.5th percentile)
        _, btc_price_2_5_usd = bitcoin_power_law_price(days_since_genesis)

        # Calculate exchange rate for this year (continues depreciating)
        current_usd_inr = retirement_usd_inr * (1 + depreciation_rate) ** year

        # Calculate annual expenditure with inflation
        year_expense_usd = annual_expenditure_usd_year_1 * \
            (1 + inflation_rate) ** year
        year_expense_inr = year_expense_usd * current_usd_inr

        # Calculate Bitcoin needed
        btc_price_2_5_inr = btc_price_2_5_usd * current_usd_inr
        btc_needed_this_year = year_expense_inr / btc_price_2_5_inr
        total_bitcoin_needed += btc_needed_this_year

        # Store breakdown
        breakdown.append({
            'Year': current_retirement_year,
            'Age': current_age + years_to_retirement + year,
            'Expense (INR)': year_expense_inr,
            'BTC Price (USD)': btc_price_2_5_usd,
            'BTC Price (INR)': btc_price_2_5_inr,
            'BTC Needed': btc_needed_this_year,
            'USD/INR Rate': current_usd_inr
        })

    return {
        'total_bitcoin_needed': total_bitcoin_needed,
        'retirement_usd_inr': retirement_usd_inr,
        'annual_expenditure_at_retirement': annual_expenditure_at_retirement,
        'retirement_age': current_age + years_to_retirement,
        'years_in_retirement': years_in_retirement,
        'breakdown': pd.DataFrame(breakdown)
    }
# Helper function to format data in the breakdown dataframe for display


def format_breakdown_dataframe(breakdown_df):
    """Format breakdown dataframe for display with Indian commas and proper decimals"""
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


def display_scenario_comparison(scenario_results, retirement_year):
    """
    Display scenario comparison in Streamlit
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

        comparison_data.append({
            'Scenario': scenario_name,
            'BTC Needed': f"{results['total_bitcoin_needed']:.4f}",
            'Depreciation Rate': f"{params['depreciation_rate']*100:.1f}%",
            'Inflation Rate': f"{params['inflation_rate']*100:.1f}%",
            'USD/INR at retirement': f"₹{usd_inr_retirement:.0f}",
            'Description': params['description']
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
            label="🎯 Primary Target",
            value=f"{conservative_btc:.4f} BTC",
            help="Conservative scenario - recommended planning baseline"
        )

    with col2:
        st.metric(
            label="🛡️ Stress Test",
            value=f"{extreme_btc:.4f} BTC",
            help="Extreme scenario - worst-case protection"
        )

    with col3:
        st.metric(
            label="📈 Extra Buffer",
            value=f"{extreme_btc - conservative_btc:.4f} BTC",
            help="Additional BTC needed for extreme scenario protection"
        )


def create_scenario_charts(scenario_results, current_age, retirement_year):
    """
    Create visualization charts for scenarios
    """
    st.markdown("### Scenario Visualization")

    # Prepare data for all scenarios
    expense_data = []

    for scenario_name, results in scenario_results.items():
        breakdown_df = results['breakdown'].copy()
        # Add scenario column for grouping and convert to lakhs
        breakdown_df['Scenario'] = scenario_name.capitalize()
        breakdown_df['Expense (Lakhs)'] = breakdown_df['Expense (INR)'] / 100000
        expense_data.append(
            breakdown_df[['Year', 'Expense (Lakhs)', 'Scenario', 'Expense (INR)']])

    # Combine all scenario data
    combined_expenses_df = pd.concat(expense_data, ignore_index=True)

    # Create the combined line chart
    fig_expenses = px.line(
        combined_expenses_df,
        x='Year',
        y='Expense (Lakhs)',
        color='Scenario',
        title="Annual Retirement Expenses (Inflation-Adjusted) - All Scenarios",
        markers=True,
        color_discrete_map={
            'Optimistic': '#28a745',
            'Conservative': '#ffc107',
            'Extreme': '#dc3545'
        }
    )

    fig_expenses.update_yaxes(range=[0, None])
    # Enhance the chart appearance
    fig_expenses.update_traces(
        line=dict(width=3),
        marker=dict(size=6)
    )

    # Update layout with Indian formatting
    fig_expenses.update_layout(
        yaxis_title="Annual Expense (₹ Lakhs)",
        xaxis_title="Retirement Year",
        yaxis=dict(tickformat='.1f'),
        hovermode='x unified',
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="right",
            x=0.99
        )
    )

    # Custom hover template using indian_commas
    fig_expenses.update_traces(
        hovertemplate='<b>%{fullData.name}</b><br>' +
                      'Year: %{x}<br>' +
                      'Expense: ₹%{customdata} Lakhs<extra></extra>',
        customdata=[indian_commas(val/100000, 1)
                    for val in combined_expenses_df['Expense (INR)']]
    )

    st.plotly_chart(fig_expenses, use_container_width=True)

    # Add insights with indian_commas formatting
    st.markdown("**💡 Key Insights:**")
    col1, col2, col3 = st.columns(3)

    # Calculate final year expenses for each scenario
    final_year_expenses = {}
    for scenario_name, results in scenario_results.items():
        final_expense = results['breakdown']['Expense (INR)'].iloc[-1]
        final_year_expenses[scenario_name] = final_expense

    with col1:
        st.metric(
            "Final Year Expense - Optimistic",
            f"₹{indian_commas(final_year_expenses['optimistic'], 0)}",
            help="Annual expense in final retirement year under optimistic inflation"
        )

    with col2:
        conservative_vs_optimistic = (
            (final_year_expenses['conservative']/final_year_expenses['optimistic'])-1)*100
        st.metric(
            "Final Year Expense - Conservative",
            f"₹{indian_commas(final_year_expenses['conservative'], 0)}",
            f"+{conservative_vs_optimistic:.1f}% vs Optimistic"
        )

    with col3:
        extreme_vs_optimistic = (
            (final_year_expenses['extreme']/final_year_expenses['optimistic'])-1)*100
        st.metric(
            "Final Year Expense - Extreme",
            f"₹{indian_commas(final_year_expenses['extreme'], 0)}",
            f"+{extreme_vs_optimistic:.1f}% vs Optimistic"
        )

    # Exchange Rate Projection Chart
    # Calculate user's actual timeline
    # Get current market rate as base, fallback to 86 if API fails
    try:
        _, current_market_rate = get_market_price()
        base_rate = current_market_rate if current_market_rate > 0 else 86.0
    except:
        base_rate = 86.0  # Conservative fallback

    current_year = datetime.now().year
    years_to_90 = 90 - current_age
    retirement_age = current_age + (retirement_year - current_year)
    end_of_retirement_year = current_year + years_to_90

    # Chart years: from current year to when user turns 90
    start_year = current_year
    end_year = max(end_of_retirement_year, retirement_year +
                   10)  # At least 10 years post retirement
    years = list(range(start_year, end_year + 1))

    chart_data = []
    for year in years:
        years_from_current = year - current_year
        user_age_in_year = current_age + years_from_current
        if user_age_in_year <= 90:
            for scenario_name, results in scenario_results.items():
                depreciation_rate = results['params']['depreciation_rate']
                usd_inr_rate = base_rate * \
                    (1 + depreciation_rate) ** years_from_current
                chart_data.append({
                    'Year': year,
                    'USD/INR Rate': usd_inr_rate,
                    'Scenario': scenario_name.capitalize(),
                    'User Age': user_age_in_year
                })

    projection_df = pd.DataFrame(chart_data)
    if chart_data:  # Only create chart if we have data
        projection_df = pd.DataFrame(chart_data)

        fig_projection = px.line(
            projection_df,
            x='Year',
            y='USD/INR Rate',
            color='Scenario',
            title=f"USD/INR Exchange Rate Projections (Age {current_age} to 90)",
            color_discrete_map={
                'Optimistic': '#28a745',
                'Conservative': '#ffc107',
                'Extreme': '#dc3545'
            },
            markers=True,
            hover_data=['User Age']
        )
        fig_projection.update_traces(
            line=dict(width=3), marker=dict(size=6))
        # Add vertical line for retirement year
        fig_projection.add_vline(
            x=retirement_year,
            line_dash="dash",
            line_color="gray",
            annotation_text=f"Retirement (Age {retirement_age})",
            annotation_position="top right"
        )

        # Format y-axis to show rupees
        fig_projection.update_yaxes(title="USD/INR Exchange Rate (₹)")
        fig_projection.update_xaxes(title="Year")

        # Add grid for better readability
        fig_projection.update_layout(
            hovermode='x unified',
            legend=dict(
                yanchor="top",
                y=0.99,
                xanchor="left",
                x=0.01
            )
        )

        st.plotly_chart(fig_projection, use_container_width=True)

        # Show key projection values
        col1, col2, col3 = st.columns(3)

        with col1:
            optimistic_rate = base_rate * \
                (1 + 0.030) ** (retirement_year - current_year)
            st.metric(
                "Optimistic @ Retirement",
                f"₹{optimistic_rate:.1f}",
                f"+{((optimistic_rate/base_rate)-1)*100:.1f}%"
            )

        with col2:
            conservative_rate = base_rate * \
                (1 + 0.045) ** (retirement_year - current_year)
            st.metric(
                "Conservative @ Retirement",
                f"₹{conservative_rate:.1f}",
                f"+{((conservative_rate/base_rate)-1)*100:.1f}%"
            )

        with col3:
            extreme_rate = base_rate * \
                (1 + 0.060) ** (retirement_year - current_year)
            st.metric(
                "Extreme @ Retirement",
                f"₹{extreme_rate:.1f}",
                f"+{((extreme_rate/base_rate)-1)*100:.1f}%"
            )

    # Display timeline info
    st.info(f"""
    **📅 Your Timeline:**
    - **Current Age**: {current_age} years (Year {current_year})
    - **Retirement Age**: {retirement_age} years (Year {retirement_year})
    - **Years to Retirement**: {retirement_year - current_year} years
    - **Chart Coverage**: Until age 90 (Year {end_of_retirement_year})
    - **Base Exchange Rate**: ₹{base_rate:.1f}/USD (current market rate)
    """)

    # Additional insights
    years_to_retirement = retirement_year - current_year
    if years_to_retirement <= 5:
        st.warning("⚠️ **Short Retirement Timeline**: With less than 5 years to retirement, consider the Conservative or Extreme scenarios for better security.")
    elif years_to_retirement >= 30:
        st.success("✅ **Long Retirement Timeline**: You have time for Bitcoin appreciation. The Optimistic scenario may be reasonable, but Conservative is still recommended.")
    else:
        st.info("ℹ️ **Moderate Timeline**: The Conservative scenario provides a good balance of growth assumptions and safety margin.")


def main():
    st.set_page_config(
        page_title="Bitcoin Retirement Calculator",
        page_icon="₿",
        layout="wide"
    )

    st.title("₿ Bitcoin Retirement Calculator")
    st.markdown(
        "**Powered by the 2.5th percentile price of the Bitcoin power law model**")

    # Sidebar inputs
    st.sidebar.header("Input Parameters")

    current_age = st.sidebar.number_input(
        "Current Age",
        min_value=18, max_value=80, value=30, step=1
    )

    annual_expenditure_inr = st.sidebar.number_input(
        "Annual Expenditure (INR)",
        min_value=100000, max_value=50000000, value=1000000, step=50000
    )

    retirement_year = st.sidebar.number_input(
        "Retirement Year",
        min_value=2026, max_value=2070, value=2045, step=1
    )

    # Analysis type selection
    analysis_type = st.sidebar.radio(
        "Analysis Type",
        ["Scenario Comparison", "Single Scenario"],
        help="Choose between comparing all scenarios or analyzing one scenario in detail"
    )

    if analysis_type == "Scenario Comparison":
        # Run scenario analysis
        if st.sidebar.button("Calculate All Scenarios"):
            with st.spinner("Calculating scenarios..."):
                scenario_results = calculate_retirement_bitcoin_needs_scenarios(
                    current_age, annual_expenditure_inr, retirement_year
                )

                if scenario_results:
                    # Display results
                    display_scenario_comparison(
                        scenario_results, retirement_year)
                    create_scenario_charts(
                        scenario_results, current_age, retirement_year)

                    # Show detailed breakdown for each scenario
                    st.markdown("### Detailed Breakdown")

                    scenario_tab1, scenario_tab2, scenario_tab3 = st.tabs(
                        ["Optimistic", "Conservative", "Extreme"])

                    with scenario_tab1:
                        st.markdown("#### Optimistic Scenario")
                        st.dataframe(format_breakdown_dataframe(scenario_results['optimistic']['breakdown']),
                                     use_container_width=True, hide_index=True)

                    with scenario_tab2:
                        st.markdown("#### Conservative Scenario")
                        st.dataframe(format_breakdown_dataframe(scenario_results['conservative']['breakdown']),
                                     use_container_width=True, hide_index=True)

                    with scenario_tab3:
                        st.markdown("#### Extreme Scenario")
                        st.dataframe(format_breakdown_dataframe(scenario_results['extreme']['breakdown']),
                                     use_container_width=True, hide_index=True)

                else:
                    st.error(
                        "Error calculating scenarios. Please check your inputs.")

    else:
        # Single scenario analysis
        selected_scenario = st.sidebar.selectbox(
            "Select Scenario",
            ["optimistic", "conservative", "extreme"],
            index=1
        )

        scenario_params = get_scenario_parameters(selected_scenario)
        st.sidebar.markdown(f"**{scenario_params['description']}**")
        st.sidebar.markdown(
            f"- Depreciation: {scenario_params['depreciation_rate']*100:.1f}%")
        st.sidebar.markdown(
            f"- Inflation: {scenario_params['inflation_rate']*100:.1f}%")

        if st.sidebar.button("🚀 Calculate Single Scenario"):
            with st.spinner(f"Calculating {selected_scenario} scenario..."):
                try:
                    # Get scenario parameters
                    scenario_params = get_scenario_parameters(
                        selected_scenario)
                    depreciation_rate = scenario_params['depreciation_rate']
                    inflation_rate = scenario_params['inflation_rate']
                    description = scenario_params['description']
                    color = scenario_params['color']

                    # Calculate basic timeline parameters
                    current_year = datetime.now().year
                    years_to_retirement = retirement_year - current_year
                    retirement_age = current_age + years_to_retirement
                    years_in_retirement = max(0, 90 - retirement_age)

                    if years_in_retirement <= 0:
                        st.error(
                            "❌ Invalid retirement timeline. Retirement age would be 90 or older.")
                    else:
                        # Bitcoin genesis date
                        genesis_date = datetime(2009, 1, 3)

                        # Calculate single scenario results
                        results = calculate_single_scenario(
                            current_age, retirement_year, years_to_retirement, years_in_retirement,
                            annual_expenditure_inr, depreciation_rate, inflation_rate, genesis_date
                        )

                        # Display scenario header
                        st.markdown(
                            f"## {selected_scenario.capitalize()} Scenario Analysis")
                        st.markdown(f"**{description}**")

                        # Display key metrics
                        col1, col2, col3, col4 = st.columns(4)

                        with col1:
                            st.metric(
                                label="💰 Total Bitcoin Needed",
                                value=f"{results['total_bitcoin_needed']:.4f} BTC"
                            )

                        with col2:
                            st.metric(
                                label="USD/INR at Retirement",
                                value=f"₹{results['retirement_usd_inr']:.1f}",
                                delta=f"+{((results['retirement_usd_inr']/87)-1)*100:.1f}% from today"
                            )

                        with col3:
                            st.metric(
                                label="Retirement Age",
                                value=f"{int(results['retirement_age'])} years"
                            )

                        with col4:
                            st.metric(
                                label="Years in Retirement",
                                value=f"{int(results['years_in_retirement'])} years"
                            )

                        # Current Bitcoin value
                        try:
                            current_btc_price, current_usd_inr = get_market_price()
                            if current_btc_price > 0 and current_usd_inr > 0:
                                current_value_usd = results['total_bitcoin_needed'] * \
                                    current_btc_price
                                current_value_inr = current_value_usd * current_usd_inr

                                st.markdown(
                                    "### 💵 Current Value of Required Bitcoin")
                                col1, col2 = st.columns(2)

                                with col1:
                                    st.metric(
                                        label="Current Value (USD)",
                                        value=f"${current_value_usd:,.0f}"
                                    )

                                with col2:
                                    st.metric(
                                        label="Current Value (INR)",
                                        value=f"₹{indian_commas(current_value_inr, 0)}"
                                    )
                        except:
                            pass

                        # Scenario parameters display
                        st.markdown("### ⚙️ Scenario Parameters")
                        param_col1, param_col2 = st.columns(2)

                        with param_col1:
                            st.metric(
                                label="USD/INR Depreciation Rate",
                                value=f"{depreciation_rate*100:.1f}% annually"
                            )

                        with param_col2:
                            st.metric(
                                label="Inflation Rate",
                                value=f"{inflation_rate*100:.1f}% annually"
                            )

                        # Year-wise breakdown chart
                        st.markdown("### 📈 Bitcoin Requirements Over Time")

                        fig_yearly = px.bar(
                            results['breakdown'],
                            x='Year',
                            y='BTC Needed',
                            title=f"Annual Bitcoin Requirements - {selected_scenario.capitalize()} Scenario",
                            color_discrete_sequence=[color],
                            hover_data=['Age', 'Expense (INR)', 'USD/INR Rate']
                        )

                        fig_yearly.update_layout(
                            xaxis_title="Retirement Year",
                            yaxis_title="Bitcoin Needed",
                            showlegend=False
                        )

                        st.plotly_chart(fig_yearly, use_container_width=True)

                        # Cumulative Bitcoin requirement chart
                        results['breakdown']['Cumulative BTC'] = results['breakdown']['BTC Needed'].cumsum(
                        )

                        fig_cumulative = px.line(
                            results['breakdown'],
                            x='Year',
                            y='Cumulative BTC',
                            title=f"Cumulative Bitcoin Requirements - {selected_scenario.capitalize()} Scenario",
                            color_discrete_sequence=[color],
                            markers=True
                        )

                        fig_cumulative.update_layout(
                            xaxis_title="Retirement Year",
                            yaxis_title="Cumulative Bitcoin Needed"
                        )

                        st.plotly_chart(
                            fig_cumulative, use_container_width=True)

                        # Detailed breakdown table
                        st.markdown("### 📋 Detailed Year-wise Breakdown")

                        # Format the dataframe for better display
                        display_df = results['breakdown'].copy()
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

                        st.dataframe(
                            display_df, use_container_width=True, hide_index=True)

                        # Download option
                        csv = results['breakdown'].to_csv(index=False)
                        st.download_button(
                            label="💾 Download Detailed Breakdown (CSV)",
                            data=csv,
                            file_name=f'bitcoin_retirement_{selected_scenario}_scenario_{current_age}yo_{retirement_year}.csv',
                            mime='text/csv'
                        )

                        # Scenario-specific recommendations
                        st.markdown("### 💡 Recommendations")

                        if selected_scenario == 'optimistic':
                            st.success(
                                "✅ **Optimistic Scenario**: This assumes favorable economic conditions. Consider also reviewing the Conservative scenario for additional security.")
                        elif selected_scenario == 'conservative':
                            st.info(
                                "ℹ️ **Conservative Scenario**: This provides a balanced approach with reasonable safety margins. Recommended for primary retirement planning.")
                        else:  # extreme
                            st.warning(
                                "⚠️ **Extreme Scenario**: This represents worst-case conditions. If you can afford this amount, you'll be well-prepared for any economic situation.")

                        # Compare with other scenarios
                        st.markdown("### 🔄 Quick Scenario Comparison")
                        comparison_data = []

                        for scenario_name in ['optimistic', 'conservative', 'extreme']:
                            temp_params = get_scenario_parameters(
                                scenario_name)
                            temp_result = calculate_single_scenario(
                                current_age, retirement_year, years_to_retirement, years_in_retirement,
                                annual_expenditure_inr, temp_params['depreciation_rate'],
                                temp_params['inflation_rate'], genesis_date
                            )

                            comparison_data.append({
                                'Scenario': scenario_name.capitalize(),
                                'BTC Required': f"{temp_result['total_bitcoin_needed']:.4f}",
                                'Difference': f"{temp_result['total_bitcoin_needed'] - results['total_bitcoin_needed']:+.4f}" if scenario_name != selected_scenario else "Current"
                            })

                        comparison_df = pd.DataFrame(comparison_data)
                        st.dataframe(comparison_df, use_container_width=True)
                except Exception as e:
                    st.error(f"❌ Error in calculation: {str(e)}")
                    st.error("Please check your input values and try again.")


if __name__ == "__main__":
    main()
