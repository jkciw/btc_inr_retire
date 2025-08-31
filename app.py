"""
Main app to fetch user intputs and calculate bitcoins required for retirement 

Three scenarios are considered: Optimistic, Conservative, Extreme

A Heatmap is included to show how inflation and currency depreciation impact Bitcoin needs

A information module about Bitcoin's power law model is also included

Analysis of the user's current portfolio vs required Bitcoin is also included
"""

import streamlit as st
from datetime import datetime
from calculations import (
    calculate_retirement_bitcoin_needs_scenarios,
    initialize_percentiles_from_csv
)
from utils import format_breakdown_dataframe
from visualization import display_scenario_comparison
import plotly.express as px
import pandas as pd
from power_law_chart import display_power_law_chart
from heatmap import display_interactive_heatmap_chart
from portfolio_gap_analysis import display_portfolio_gap_analysis
from config import INPUT_LIMITS
from currency_analysis import display_currency_analysis_page


def main():
    """Main function to run the Streamlit application."""
    st.set_page_config(
        page_title="Bitcoin Retirement Calculator",
        page_icon="₿",
        layout="wide"
    )

    # Initialize session state
    if 'calculation_data' not in st.session_state:
        st.session_state.calculation_data = None
    if 'show_results' not in st.session_state:
        st.session_state.show_results = False

    # Initialize percentiles
    if 'percentiles_initialized' not in st.session_state:
        with st.spinner("Loading Bitcoin power law data (one-time setup)..."):
            try:
                initialize_percentiles_from_csv("coinmcap_consolidated.csv")
                st.session_state.percentiles_initialized = True
                # Don't show success message here to avoid clutter
            except Exception as e:
                st.error(f"Error initializing percentiles: {e}")
                st.session_state.percentiles_initialized = False
                # Stop execution if percentiles fail to load
                st.stop()

    st.title("₿ Bitcoin Retirement Calculator")
    st.markdown(
        "**Powered by the 2.5th percentile price of the Bitcoin power law model**")

    # Display status info
    display_calculation_status()

    # Create tabs - always show Calculator and Power Law, Gap Analysis appears after calculation
    if st.session_state.calculation_data is None:
        # Show Calculator and Power Law tabs
        tab1, tab3 = st.tabs(
            ["🏠 Bitcoin Calculator", "📈 About Power Law Model"])

        with tab1:
            display_calculator_page()

        with tab3:
            display_power_law_page()

    else:
        # Show all three tabs
        tab1, tab2, tab3, tab4 = st.tabs([
            "🏠 Bitcoin Calculator",
            "📊 Accumulation Strategy Analysis",
            "📈 About Power Law Model",
            "💹 Currency Analysis"
        ])

        with tab1:
            display_calculator_page()

        with tab2:
            display_gap_analysis_page()

        with tab3:
            display_power_law_page()
        with tab4:
            display_currency_analysis_page()


def display_calculation_status():
    """Display calculation status information."""
    if st.session_state.calculation_data:
        data = st.session_state.calculation_data
        calc_time = datetime.fromisoformat(data['timestamp']).strftime('%H:%M')
        status_text = f"📋 Last calculation: Age {data['current_age']}, Retirement {data['retirement_year']} (at {calc_time})"
    else:
        status_text = "Calculate your Bitcoin retirement needs to unlock accumulation strategy analysis"

    st.markdown(f"""
    <div class="calculation-status">
        <p class="status-text">{status_text}</p>
    </div>
    """, unsafe_allow_html=True)


def display_calculator_page():
    """Display the main calculator page with persistent results."""
    # Sidebar inputs
    current_age, annual_expenditure_inr, retirement_year = setup_sidebar()

    # Show current input vs saved calculation comparison
    if st.session_state.calculation_data:
        display_input_comparison(
            current_age, annual_expenditure_inr, retirement_year)

    # Single calculate button
    if st.sidebar.button("Calculate Bitcoin Requirements", icon=":material/calculate:", type="primary"):
        with st.spinner("Calculating all scenarios..."):
            scenario_results = calculate_retirement_bitcoin_needs_scenarios(
                current_age, annual_expenditure_inr, retirement_year
            )

        if scenario_results:
            # Store calculation data persistently
            st.session_state.calculation_data = {
                'scenario_results': scenario_results,
                'current_age': current_age,
                'annual_expenditure_inr': annual_expenditure_inr,
                'retirement_year': retirement_year,
                'timestamp': datetime.now().isoformat()
            }

            # Enable results display
            st.session_state.show_results = True
            st.rerun()  # Refresh to show results

        else:
            st.error(
                "Error calculating scenarios. Please check your inputs and ensure retirement timeline is valid.")

    # Display results if we have calculation data (persistent results)
    if st.session_state.calculation_data and st.session_state.show_results:
        display_persistent_results()


def display_gap_analysis_page():
    """Display the gap analysis page."""
    if st.session_state.calculation_data is None:
        st.warning(
            "⚠️ No calculation data found. Please go back to the calculator and run your scenarios first.")
        return

    data = st.session_state.calculation_data

    st.markdown("## Accumulation Strategy Analysis & Action Plan")
    st.markdown(
        "*Personalized Bitcoin accumulation strategies based on your current position*")

    # Show when the underlying calculation was done
    calc_time = datetime.fromisoformat(
        data['timestamp']).strftime('%B %d, %Y at %H:%M')
    st.caption(f"🕒 Based on calculation from {calc_time}")

    # Display summary of calculation data
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Current Age", f"{data['current_age']} years")
    with col2:
        st.metric("Retirement Year", data['retirement_year'])
    with col3:
        st.metric("Annual Expenditure",
                  f"₹{data['annual_expenditure_inr']/100000:.1f} lakhs")

    # Display the gap analysis
    display_portfolio_gap_analysis(
        data['scenario_results'],
        data['current_age'],
        data['retirement_year']
    )


def display_power_law_page():
    """Display the Power Law Model information page."""
    # Display the power law chart and explanation
    display_power_law_chart("coinmcap_consolidated.csv")

    st.warning("""
    **⚠️ Important Disclaimer**
    
    The Power Law model is a mathematical representation based on historical data. Not to be considered financial advice. 
    """)


def display_input_comparison(current_age, annual_expenditure_inr, retirement_year):
    """Show comparison between current inputs and saved calculation."""
    data = st.session_state.calculation_data

    # Check if inputs have changed
    inputs_changed = (
        current_age != data['current_age'] or
        annual_expenditure_inr != data['annual_expenditure_inr'] or
        retirement_year != data['retirement_year']
    )

    if inputs_changed:
        st.info("""
        **Your inputs have changed since the last calculation**
        
        Click "Calculate Bitcoin Requirements" to update your results with the new parameters.
        """)

        # Show what changed
        with st.expander("See what changed", expanded=False):
            col1, col2 = st.columns(2)

            with col1:
                st.markdown("**Previous Calculation:**")
                st.write(f"• Age: {data['current_age']}")
                st.write(
                    f"• Annual Expenditure: ₹{data['annual_expenditure_inr']/100000:.1f} lakhs")
                st.write(f"• Retirement Year: {data['retirement_year']}")

            with col2:
                st.markdown("**Current Inputs:**")
                st.write(
                    f"• Age: {current_age} {'⚠️' if current_age != data['current_age'] else '✅'}")
                st.write(
                    f"• Annual Expenditure: ₹{annual_expenditure_inr/100000:.1f} lakhs {'⚠️' if annual_expenditure_inr != data['annual_expenditure_inr'] else '✅'}")
                st.write(
                    f"• Retirement Year: {retirement_year} {'⚠️' if retirement_year != data['retirement_year'] else '✅'}")


def display_persistent_results():
    """Display the persistent calculation results."""
    data = st.session_state.calculation_data
    scenario_results = data['scenario_results']
    current_age = data['current_age']
    annual_expenditure_inr = data['annual_expenditure_inr']
    retirement_year = data['retirement_year']

    # Add a header for the results section
    st.markdown("## Your Bitcoin Retirement Analysis Results")

    # Show when this was calculated
    calc_time = datetime.fromisoformat(
        data['timestamp']).strftime('%B %d, %Y at %H:%M')
    st.caption(f"🕒 Calculated on {calc_time}")

    # 1. Display scenario comparison table
    display_scenario_comparison(
        scenario_results, retirement_year)

    # 2. Display recomendations
    display_recommendations(scenario_results)

    # 3. Create stacked bar chart for Bitcoin requirements
    create_bitcoin_requirements_chart(
        scenario_results, current_age, retirement_year)

    # 4. Create annual expenses chart for all scenarios
    create_annual_expenses_chart(
        scenario_results, current_age, retirement_year)

    # 5. Show heatmap for BTC needed
    display_interactive_heatmap_chart(
        current_age, annual_expenditure_inr, retirement_year)

    # 6. Show detailed breakdown for each scenario
    display_scenario_breakdowns(scenario_results)

    # 7. Show call-to-action for gap analysis if not calculated yet
    if st.session_state.calculation_data:
        st.markdown("---")
        st.success("**Ready for analysis of your exisitng bitcoin retirement portfolio!** Switch to the 'Accumulation Strategy Analysis' tab above to get your personalized action plan.")


def setup_sidebar():
    """Setup sidebar inputs and return user parameters."""
    st.sidebar.header("Input Parameters")

    current_age = st.sidebar.number_input(
        "Current Age",
        min_value=INPUT_LIMITS['age']['min'], max_value=INPUT_LIMITS['age']['max'], value=INPUT_LIMITS['age']['default'], step=1
    )

    annual_expenditure_inr = st.sidebar.number_input(
        "Annual Expenditure (INR)",
        min_value=INPUT_LIMITS['expenditure']['min'], max_value=INPUT_LIMITS[
            'expenditure']['max'], value=INPUT_LIMITS['expenditure']['default'], step=50000
    )

    retirement_year = st.sidebar.number_input(
        "Retirement Year",
        min_value=INPUT_LIMITS['retirement_year']['min'], max_value=INPUT_LIMITS[
            'retirement_year']['max'], value=INPUT_LIMITS['retirement_year']['default'], step=1
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
        hovertemplate="%{fullData.name}<br>" +
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

    st.plotly_chart(fig, width='stretch')


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
        hovertemplate="%{fullData.name}<br>" +
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

    st.plotly_chart(fig_expenses, width='stretch')


def display_recommendations(scenario_results: dict):
    """Display enhanced recommendations section."""
    st.markdown("### Recommendations")

    conservative_btc = scenario_results['Conservative']['total_bitcoin_needed']
    extreme_btc = scenario_results['Extreme']['total_bitcoin_needed']
    optimistic_btc = scenario_results['Optimistic']['total_bitcoin_needed']

    col1, col2 = st.columns(2)

    with col1:
        st.metric(
            label="Primary Target",
            value=f"{conservative_btc:.4f} BTC",
            help="Conservative scenario - recommended planning baseline"
        )

    with col2:
        st.metric(
            label="Extra Buffer",
            value=f"{extreme_btc - conservative_btc:.4f} BTC",
            help="Additional Buffer needed for extreme scenario protection"
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

    with st.expander("Click here to view the full year-wise breakdown for each scenario", expanded=False):
        scenario_tab1, scenario_tab2, scenario_tab3 = st.tabs(
            ["🟢 Optimistic", "🟡 Conservative", "🔴 Extreme"])

        with scenario_tab1:
            st.markdown("#### Optimistic Scenario")
            optimistic_df = format_breakdown_dataframe(
                scenario_results['Optimistic']['breakdown'])
            st.dataframe(optimistic_df,
                         hide_index=True)

            # Download button for optimistic
            csv_opt = scenario_results['Optimistic']['breakdown'].to_csv(
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
                scenario_results['Conservative']['breakdown'])
            st.dataframe(conservative_df, hide_index=True)

            # Download button for conservative
            csv_cons = scenario_results['Conservative']['breakdown'].to_csv(
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
                scenario_results['Extreme']['breakdown'])
            st.dataframe(extreme_df, hide_index=True)

            # Download button for extreme
            csv_ext = scenario_results['Extreme']['breakdown'].to_csv(
                index=False)
            st.download_button(
                label="Download Extreme Scenario (CSV)",
                data=csv_ext,
                file_name=f'bitcoin_retirement_extreme_scenario.csv',
                mime='text/csv'
            )


if __name__ == "__main__":
    main()
