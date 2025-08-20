"""
Main Streamlit Application
Entry point for the Bitcoin Retirement Calculator web application.
"""

import streamlit as st
from datetime import datetime
from calculations import (
    calculate_retirement_bitcoin_needs_scenarios,
    get_scenario_parameters
)
from utils import format_breakdown_dataframe
from visualization import display_scenario_comparison, create_scenario_charts
from single_scenario import display_single_scenario_analysis


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

    # Sidebar inputs
    current_age, annual_expenditure_inr, retirement_year, analysis_type = setup_sidebar()

    if analysis_type == "Scenario Comparison":
        scenario_comparison(
            current_age, annual_expenditure_inr, retirement_year)
    else:
        single_scenario(current_age, annual_expenditure_inr, retirement_year)


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

    # Analysis type selection
    analysis_type = st.sidebar.radio(
        "Analysis Type",
        ["Scenario Comparison", "Single Scenario"],
        help="Choose between comparing all scenarios or analyzing one scenario in detail"
    )

    return current_age, annual_expenditure_inr, retirement_year, analysis_type


def scenario_comparison(current_age: int, annual_expenditure_inr: float, retirement_year: int):
    """Handle scenario comparison analysis."""
    if st.sidebar.button("Calculate All Scenarios", icon=":material/calculate:"):
        with st.spinner("Calculating scenarios..."):
            scenario_results = calculate_retirement_bitcoin_needs_scenarios(
                current_age, annual_expenditure_inr, retirement_year
            )

            if scenario_results:
                # Display results
                display_scenario_comparison(scenario_results, retirement_year)
                create_scenario_charts(
                    scenario_results, current_age, retirement_year)

                # Show detailed breakdown for each scenario
                display_scenario_breakdowns(scenario_results)
            else:
                st.error("Error calculating scenarios. Please check your inputs.")


def single_scenario(current_age: int, annual_expenditure_inr: float, retirement_year: int):
    """Handle single scenario analysis."""
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

    if st.sidebar.button("Calculate Single Scenario", icon=":material/calculate:"):
        display_single_scenario_analysis(
            selected_scenario, current_age, annual_expenditure_inr, retirement_year
        )


def display_scenario_breakdowns(scenario_results):
    """Display detailed breakdown for each scenario in tabs."""
    st.markdown("### Detailed Breakdown")

    scenario_tab1, scenario_tab2, scenario_tab3 = st.tabs(
        ["Optimistic", "Conservative", "Extreme"])

    with scenario_tab1:
        st.markdown("#### Optimistic Scenario")
        st.dataframe(
            format_breakdown_dataframe(
                scenario_results['optimistic']['breakdown']),
            use_container_width=True,
            hide_index=True
        )

    with scenario_tab2:
        st.markdown("#### Conservative Scenario")
        st.dataframe(
            format_breakdown_dataframe(
                scenario_results['conservative']['breakdown']),
            use_container_width=True,
            hide_index=True
        )

    with scenario_tab3:
        st.markdown("#### Extreme Scenario")
        st.dataframe(
            format_breakdown_dataframe(
                scenario_results['extreme']['breakdown']),
            use_container_width=True,
            hide_index=True
        )


if __name__ == "__main__":
    main()
