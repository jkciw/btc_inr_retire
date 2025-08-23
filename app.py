"""
Main app to fetch user intputs and calculate bitcoins required for retirement 

Three scenarios are considered: Optimistic, Conservative, Extreme

A information module about Bitcoin's power law model is also included
"""

import streamlit as st
from datetime import datetime
from calculations import (
    calculate_retirement_bitcoin_needs_scenarios,
    bitcoin_power_law_price
)
from utils import format_breakdown_dataframe
from visualization import display_scenario_comparison
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np
from utils import indian_commas
from power_law_chart import display_power_law_chart
from heatmap import display_interactive_heatmap_chart


def main():
    """Main function to run the Streamlit application."""
    st.set_page_config(
        page_title="Bitcoin Retirement Calculator",
        page_icon="₿",
        layout="wide"
    )

    st.title("₿ Bitcoin Retirement Calculator")
    st.markdown(
        "**Powered by the 2.5th percentile price of the Bitcoin power law model**")

    # Optional educational content - collapsed by default
    display_power_law_chart("coinmcap_consolidated.csv")

    # Sidebar inputs
    current_age, annual_expenditure_inr, retirement_year = setup_sidebar()

    # Single calculate button
    if st.sidebar.button("Calculate Bitcoin Requirements", icon=":material/calculate:", type="primary"):
        with st.spinner("Calculating all scenarios..."):
            scenario_results = calculate_retirement_bitcoin_needs_scenarios(
                current_age, annual_expenditure_inr, retirement_year
            )

            if scenario_results:
                # 1. Display scenario comparison table
                display_scenario_comparison(scenario_results, retirement_year)

                # 2. Create stacked bar chart for Bitcoin requirements
                create_bitcoin_requirements_chart(
                    scenario_results, current_age, retirement_year)

                # 3. Create annual expenses chart for all scenarios
                create_annual_expenses_chart(
                    scenario_results, current_age, retirement_year)

                # 4. Display recommendations
                display_recommendations(scenario_results)

                # 5. Show detailed breakdown for each scenario
                display_scenario_breakdowns(scenario_results)

                # 6. Show heatmap for BTC needed
                display_interactive_heatmap_chart(
                    current_age, annual_expenditure_inr, retirement_year)

            else:
                st.error(
                    "Error calculating scenarios. Please check your inputs and ensure retirement timeline is valid.")


def setup_sidebar():
    """Setup sidebar inputs and return user parameters."""
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

    return current_age, annual_expenditure_inr, retirement_year


def create_bitcoin_requirements_chart(scenario_results: dict, current_age: int, retirement_year: int):
    """Create stacked bar chart showing Bitcoin requirements for all scenarios."""
    st.markdown("### Annual Bitcoin Requirements - All Scenarios")

    # Prepare data for stacked bar chart
    all_data = []
    colors = {
        'Optimistic': '#28a745',
        'Conservative': '#ffc107',
        'Extreme': '#dc3545'
    }

    for scenario_name, results in scenario_results.items():
        breakdown = results['breakdown']
        for _, row in breakdown.iterrows():
            all_data.append({
                'Year': row['Year'],
                'BTC Needed': row['BTC Needed'],
                'Scenario': scenario_name.capitalize(),
                'Age': row['Age']
            })

    df_chart = pd.DataFrame(all_data)

    # Create stacked bar chart
    fig = px.bar(
        df_chart,
        x='Year',
        y='BTC Needed',
        color='Scenario',
        color_discrete_map={
            'Optimistic': colors['Optimistic'],
            'Conservative': colors['Conservative'],
            'Extreme': colors['Extreme']
        }
    )
    fig.update_traces(
        hovertemplate="<b>%{fullData.name}</b><br>" +
        "Year: %{x}<br>" +
        "BTC Needed: %{y:.6f}<br>" +
        "<extra></extra>"
    )

    fig.update_layout(
        xaxis_title="Retirement Year",
        yaxis_title="Bitcoin Needed",
        hovermode='x unified',
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="right",
            x=0.99
        ),
        height=500
    )

    st.plotly_chart(fig, use_container_width=True)


def create_annual_expenses_chart(scenario_results: dict, current_age: int, retirement_year: int):
    """Create line chart showing annual expenses for all scenarios."""
    st.markdown("### Annual Expenses Comparison - All Scenarios")

    # Prepare data for line chart
    all_expense_data = []
    colors = {
        'Optimistic': '#28a745',
        'Conservative': '#ffc107',
        'Extreme': '#dc3545'
    }

    for scenario_name, results in scenario_results.items():
        breakdown = results['breakdown']
        for _, row in breakdown.iterrows():
            all_expense_data.append({
                'Year': row['Year'],
                'Expense (Lakhs)': row['Expense (INR)'] / 100000,
                'Scenario': scenario_name.capitalize(),
                'Age': row['Age'],
                'Expense (INR)': row['Expense (INR)']
            })

    df_expenses = pd.DataFrame(all_expense_data)

    # Create line chart
    fig_expenses = px.line(
        df_expenses,
        x='Year',
        y='Expense (Lakhs)',
        color='Scenario',
        markers=True,
        color_discrete_map={
            'Optimistic': colors['Optimistic'],
            'Conservative': colors['Conservative'],
            'Extreme': colors['Extreme']
        }
    )

    # Update traces for better hover info
    fig_expenses.update_traces(
        line=dict(width=3),
        marker=dict(size=6),
        hovertemplate="<b>%{fullData.name}</b><br>" +
        "Year: %{x}<br>" +
        "Age: %{customdata[1]}<br>" +
        "Expenses: ₹%{y:.1f} Lakhs<br>" +
        "<extra></extra>"
    )

    # Add custom data for hover
    for i, scenario in enumerate(['Optimistic', 'Conservative', 'Extreme']):
        scenario_data = df_expenses[df_expenses['Scenario'] == scenario]
        # Pass both INR values and Age as customdata
        fig_expenses.data[i].customdata = list(zip(
            scenario_data['Expense (INR)'].values,
            scenario_data['Age'].values
        ))

    fig_expenses.update_layout(
        xaxis_title="Retirement Year",
        yaxis_title="Annual Expenses (₹ Lakhs)",
        hovermode='x unified',
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="right",
            x=0.99
        ),
        height=500
    )

    st.plotly_chart(fig_expenses, use_container_width=True)


def display_recommendations(scenario_results: dict):
    """Display enhanced recommendations section."""
    st.markdown("### Recommendations")

    conservative_btc = scenario_results['conservative']['total_bitcoin_needed']
    extreme_btc = scenario_results['extreme']['total_bitcoin_needed']
    optimistic_btc = scenario_results['optimistic']['total_bitcoin_needed']

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
            label="Optimistic Case",
            value=f"{optimistic_btc:.4f} BTC",
            help="Best-case economic conditions"
        )

    # Additional insights
    st.info(
        f"""
        **Strategic Recommendations:**
        
        - **Start with Conservative**: Plan for {conservative_btc:.4f} BTC as your baseline target
        - **Consider Extreme Buffer**: An additional {extreme_btc - conservative_btc:.4f} BTC provides maximum security
        - **Minimum Viable**: {optimistic_btc:.4f} BTC works under favorable conditions
        
        **Action Plan**: Focus on accumulating the Conservative scenario amount, with the Extreme scenario as your stretch goal for additional peace of mind.
        """
    )


def display_scenario_breakdowns(scenario_results):
    """Display detailed breakdown for each scenario in tabs."""
    st.markdown("### Detailed Year-wise Breakdown")

    scenario_tab1, scenario_tab2, scenario_tab3 = st.tabs(
        ["🟢 Optimistic", "🟡 Conservative", "🔴 Extreme"])

    with scenario_tab1:
        st.markdown("#### Optimistic Scenario")
        optimistic_df = format_breakdown_dataframe(
            scenario_results['optimistic']['breakdown'])
        st.dataframe(optimistic_df, use_container_width=True, hide_index=True)

        # Download button for optimistic
        csv_opt = scenario_results['optimistic']['breakdown'].to_csv(
            index=False)
        st.download_button(
            label="Download Optimistic Scenario (CSV)",
            data=csv_opt,
            file_name=f'bitcoin_retirement_optimistic_scenario.csv',
            mime='text/csv'
        )

    with scenario_tab2:
        st.markdown("#### Conservative Scenario")
        conservative_df = format_breakdown_dataframe(
            scenario_results['conservative']['breakdown'])
        st.dataframe(conservative_df,
                     use_container_width=True, hide_index=True)

        # Download button for conservative
        csv_cons = scenario_results['conservative']['breakdown'].to_csv(
            index=False)
        st.download_button(
            label="Download Conservative Scenario (CSV)",
            data=csv_cons,
            file_name=f'bitcoin_retirement_conservative_scenario.csv',
            mime='text/csv'
        )

    with scenario_tab3:
        st.markdown("#### Extreme Scenario")
        extreme_df = format_breakdown_dataframe(
            scenario_results['extreme']['breakdown'])
        st.dataframe(extreme_df, use_container_width=True, hide_index=True)

        # Download button for extreme
        csv_ext = scenario_results['extreme']['breakdown'].to_csv(index=False)
        st.download_button(
            label="Download Extreme Scenario (CSV)",
            data=csv_ext,
            file_name=f'bitcoin_retirement_extreme_scenario.csv',
            mime='text/csv'
        )


if __name__ == "__main__":
    main()
