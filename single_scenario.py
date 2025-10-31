"""
Single Scenario Analysis Components Module

This module contains functions for detailed single scenario analysis,
including charts, metrics, and breakdowns.
"""

import streamlit as st
import pandas as pd
import plotly.express as px
from datetime import datetime
from typing import Dict, Any
from utils import indian_commas, format_breakdown_dataframe
from market_data import get_market_price
from calculations import calculate_single_scenario, get_scenario_parameters


def display_single_scenario_analysis(selected_scenario: str, current_age: int,
                                     annual_expenditure_inr: float, retirement_year: int):
    """
    Display detailed analysis for a single scenario.

    Args:
        selected_scenario: Selected scenario name
        current_age: Current age of the person
        annual_expenditure_inr: Annual expenditure in INR
        retirement_year: Target retirement year
    """
    with st.spinner(f"Calculating {selected_scenario} scenario..."):
        try:
            # Get scenario parameters
            scenario_params = get_scenario_parameters(selected_scenario)
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
                st.error("Invalid retirement timeline.")
                return

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
            _display_key_metrics(results)

            # Current Bitcoin value
            _display_current_bitcoin_value(results)

            # Scenario parameters display
            _display_scenario_parameters(depreciation_rate, inflation_rate)

            # Year-wise breakdown chart
            _create_yearly_breakdown_chart(results, selected_scenario, color)

            # Annual expense vs BTC price comparison
            _create_expense_vs_price_chart(results, selected_scenario, color)

            # Detailed breakdown table
            _display_detailed_breakdown(results)

            # Download option
            _provide_download_option(
                results, selected_scenario, current_age, retirement_year)

            # Scenario-specific recommendations
            _display_scenario_recommendations(selected_scenario)

            # Compare with other scenarios
            _display_quick_comparison(current_age, retirement_year, years_to_retirement,
                                      years_in_retirement, annual_expenditure_inr,
                                      genesis_date, selected_scenario, results)

        except Exception as e:
            st.error(f"Error in calculation: {str(e)}")
            st.error("Please check your input values and try again.")


def _display_key_metrics(results: Dict[str, Any]):
    """Display key metrics for the scenario."""
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            label="Total Bitcoin Needed",
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


def _display_current_bitcoin_value(results: Dict[str, Any]):
    """Display current value of required Bitcoin."""
    try:
        current_btc_price, current_usd_inr = get_market_price()
        if current_btc_price > 0 and current_usd_inr > 0:
            current_value_usd = results['total_bitcoin_needed'] * \
                current_btc_price
            current_value_inr = current_value_usd * current_usd_inr

            st.markdown("### Current Value of Required Bitcoin")
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


def _display_scenario_parameters(depreciation_rate: float, inflation_rate: float):
    """Display scenario parameters."""
    st.markdown("### Scenario Parameters")
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


def _create_yearly_breakdown_chart(results: Dict[str, Any], scenario_name: str, color: str):
    """Create yearly Bitcoin requirements chart."""
    st.markdown("### Bitcoin Requirements Over Time")

    fig_yearly = px.bar(
        results['breakdown'],
        x='Year',
        y='BTC Needed',
        title=f"Annual Bitcoin Requirements - {scenario_name.capitalize()} Scenario",
        color_discrete_sequence=[color],
    )

    hover_expenses = [
        f"₹{indian_commas(val, 2)}" for val in results['breakdown']['Expense (INR)']]
    hover_rates = [
        f"₹{indian_commas(val, 2)}" for val in results['breakdown']['USD/INR Rate']]

    fig_yearly.update_traces(
        hovertemplate="<b>Year: %{x}</b>" +
        "<br>Age: %{customdata[0]}" +
        "<br>BTC Needed: %{y:.6f}" +
        "<br>Expenses(INR): %{customdata[1]}" +
        "<br>USD/INR Rate: %{customdata[2]}" +
        "<extra></extra>",
        customdata=list(zip(
            results['breakdown']['Age'].values,
            hover_expenses,
            hover_rates
        ))
    )

    fig_yearly.update_layout(
        xaxis_title="Retirement Year",
        yaxis_title="Bitcoin Needed",
        showlegend=False
    )

    st.plotly_chart(fig_yearly, use_container_width=True)


def _create_expense_vs_price_chart(results: Dict[str, Any], scenario_name: str, color: str):
    """Create annual expenses vs Bitcoin price comparison chart."""
    st.markdown("### Annual Expenses vs Bitcoin Price Growth")

    chart_data = results['breakdown'].copy()
    # Convert to lakhs
    chart_data['Expense (Lakhs)'] = chart_data['Expense (INR)'] / 100000
    # Convert to crores
    chart_data['BTC Price (Crores)'] = chart_data['BTC Price (INR)'] / 10000000

    fig_dual = px.line(
        chart_data,
        x='Year',
        y='Expense (Lakhs)',
        title=f"Annual Expenses vs Bitcoin Price Appreciation(2.5th percentile price) - {scenario_name.capitalize()} Scenario",
        color_discrete_sequence=[color],
        markers=True
    )

    fig_dual.update_traces(
        name="Annual Expenses (₹ Lakhs)",
        hovertemplate="<b>Year: %{x}</b>" +
        "<br>Annual Expenses: ₹%{y:.2f} Lakhs" +
        "<extra></extra>",
        showlegend=True
    )

    # Add BTC price on secondary axis (in crores)
    fig_dual.add_scatter(
        x=chart_data['Year'],
        y=chart_data['BTC Price (Crores)'],
        mode='lines+markers',
        name='BTC Price (₹ Crores)',
        yaxis='y2',
        line=dict(color='green', width=3),
        marker=dict(size=6, color='green'),
        hovertemplate="<b>Year: %{x}</b>" +
        "<br>BTC Price: ₹%{y:.2f} Crores" +
        "<extra></extra>",
        showlegend=True
    )

    fig_dual.update_layout(
        xaxis_title="Retirement Year",
        yaxis=dict(
            title="Annual Expenses (₹ Lakhs)",
            side="left",
            type="log",  # Log scale for primary y-axis
            tickformat='.2f'
        ),
        yaxis2=dict(
            title="BTC Price (₹ Crores)",
            side="right",
            overlaying="y",
            type="log",  # Log scale for secondary y-axis
            tickformat='.2f'
        ),
        hovermode='x unified',
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=0.01,
            bgcolor="rgba(255,255,255,0.8)"
        )
    )

    st.plotly_chart(fig_dual, use_container_width=True)


def _display_detailed_breakdown(results: Dict[str, Any]):
    """Display detailed year-wise breakdown table."""
    st.markdown("### Detailed Year-wise Breakdown")

    # Format the dataframe for better display
    display_df = format_breakdown_dataframe(results['breakdown'])
    st.dataframe(display_df, use_container_width=True, hide_index=True)


def _provide_download_option(results: Dict[str, Any], scenario_name: str,
                             current_age: int, retirement_year: int):
    """Provide CSV download option."""
    csv = results['breakdown'].to_csv(index=False)
    st.download_button(
        label="📊 Download Detailed Breakdown (CSV)",
        data=csv,
        file_name=f'bitcoin_retirement_{scenario_name}_scenario_{current_age}yo_{retirement_year}.csv',
        mime='text/csv',
        icon=":material/download:"
    )


def _display_scenario_recommendations(scenario_name: str):
    """Display scenario-specific recommendations."""
    st.markdown("### Recommendations")

    if scenario_name == 'optimistic':
        st.success(
            "**Optimistic Scenario**: This assumes favorable economic conditions. "
            "Consider also reviewing the Conservative scenario for additional security.")
    elif scenario_name == 'conservative':
        st.info(
            "**Conservative Scenario**: This provides a balanced approach with reasonable "
            "safety margins. Recommended for primary retirement planning.")
    else:  # extreme
        st.warning(
            "**Extreme Scenario**: This represents worst-case conditions. If you can "
            "afford this amount, you'll be well-prepared for any economic situation.")


def _display_quick_comparison(current_age: int, retirement_year: int, years_to_retirement: int,
                              years_in_retirement: int, annual_expenditure_inr: float,
                              genesis_date: datetime, selected_scenario: str,
                              current_results: Dict[str, Any]):
    """Display quick comparison with other scenarios."""
    st.markdown("### Quick Scenario Comparison")

    comparison_data = []
    for scenario_name in ['optimistic', 'conservative', 'extreme']:
        temp_params = get_scenario_parameters(scenario_name)
        temp_result = calculate_single_scenario(
            current_age, retirement_year, years_to_retirement, years_in_retirement,
            annual_expenditure_inr, temp_params['depreciation_rate'],
            temp_params['inflation_rate'], genesis_date
        )

        comparison_data.append({
            'Scenario': scenario_name.capitalize(),
            'BTC Required': f"{temp_result['total_bitcoin_needed']:.4f}",
            'Difference': f"{temp_result['total_bitcoin_needed'] - current_results['total_bitcoin_needed']:+.4f}"
            if scenario_name != selected_scenario else "Current"
        })

    comparison_df = pd.DataFrame(comparison_data)
    st.dataframe(comparison_df, use_container_width=True, hide_index=True)
