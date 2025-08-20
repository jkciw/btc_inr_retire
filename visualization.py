"""
Visualization and Display Components Module

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


def create_scenario_charts(scenario_results: Dict[str, Any], current_age: int, retirement_year: int):
    """
    Create visualization charts for scenarios.
    
    Args:
        scenario_results: Results from all scenarios
        current_age: Current age of the person
        retirement_year: Target retirement year
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
        hovertemplate='%{fullData.name}' +
        '<br>Year: %{x}' +
        '<br>Expense: ₹%{customdata} Lakhs',
        customdata=[indian_commas(val/100000, 1)
                   for val in combined_expenses_df['Expense (INR)']]
    )
    
    st.plotly_chart(fig_expenses, use_container_width=True)
    
    # Add insights with indian_commas formatting
    st.markdown("**Key Insights:**")
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
    _create_exchange_rate_chart(scenario_results, current_age, retirement_year)


def _create_exchange_rate_chart(scenario_results: Dict[str, Any], current_age: int, retirement_year: int):
    """
    Create exchange rate projection chart.
    
    Args:
        scenario_results: Results from all scenarios
        current_age: Current age of the person
        retirement_year: Target retirement year
    """
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
    end_year = max(end_of_retirement_year, retirement_year + 10)  # At least 10 years post retirement
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
        _display_exchange_rate_metrics(base_rate, retirement_year)
        
        # Display timeline info
        _display_timeline_info(current_age, retirement_age, retirement_year, 
                              end_of_retirement_year, base_rate)


def _display_exchange_rate_metrics(base_rate: float, retirement_year: int):
    """Display exchange rate projection metrics."""
    current_year = datetime.now().year
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


def _display_timeline_info(current_age: int, retirement_age: int, retirement_year: int,
                          end_of_retirement_year: int, base_rate: float):
    """Display timeline information and insights."""
    current_year = datetime.now().year
    years_to_retirement = retirement_year - current_year
    
    st.info(f"""
    **Your Timeline:**
    - **Current Age**: {current_age} years (Year {current_year})
    - **Retirement Age**: {retirement_age} years (Year {retirement_year})
    - **Years to Retirement**: {years_to_retirement} years
    - **Chart Coverage**: Until age 90 (Year {end_of_retirement_year})
    - **Base Exchange Rate**: ₹{base_rate:.1f}/USD (current market rate)
    """)
    
    # Additional insights
    if years_to_retirement <= 5:
        st.warning("**Short Retirement Timeline**: With less than 5 years to retirement, consider the Conservative or Extreme scenarios for better security.")
    elif years_to_retirement >= 30:
        st.success("**Long Retirement Timeline**: You have time for Bitcoin appreciation. The Optimistic scenario may be reasonable, but Conservative is still recommended.")
    else:
        st.info("ℹ️ **Moderate Timeline**: The Conservative scenario provides a good balance of growth assumptions and safety margin.")