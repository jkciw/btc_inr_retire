"""
Interactive Heatmap Module
This module creates an interactive heatmap that shows Bitcoin requirements across all parameter combinations in a much clearer way than 3D surface plots.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime
from calculations import calculate_single_scenario


class InteractiveHeatmap:
    """
    A class to handle interactive heatmap visualization for Bitcoin retirement calculator
    """

    def __init__(self, current_age: int, annual_expenditure_inr: float, retirement_year: int):
        """Initialize the InteractiveHeatmap class with user parameters."""
        self.genesis_date = datetime(2009, 1, 3)
        self.current_age = current_age
        self.annual_expenditure_inr = annual_expenditure_inr
        self.retirement_year = retirement_year

        # Calculate retirement parameters

        current_year = datetime.now().year
        self.years_to_retirement = retirement_year - current_year
        retirement_age = current_age + self.years_to_retirement
        self.years_in_retirement = max(0, 90 - retirement_age)

        # Standard scenario parameters
        self.scenarios = {
            'optimistic': {'inflation': 6.5, 'depreciation': 3.0, 'color': '#28a745'},
            'conservative': {'inflation': 8.0, 'depreciation': 4.5, 'color': '#ffc107'},
            'extreme': {'inflation': 10.0, 'depreciation': 6.0, 'color': '#dc3545'}
        }

    def calculate_btc_for_parameters(self, inflation_rate: float, depreciation_rate: float) -> float:
        """Calculate BTC needed for given inflation and depreciation parameters."""

        if self.years_in_retirement <= 0:
            return 0.0
        try:
            result = calculate_single_scenario(
                self.current_age,
                self.retirement_year,
                self.years_to_retirement,
                self.years_in_retirement,
                self.annual_expenditure_inr,
                depreciation_rate,
                inflation_rate,
                self.genesis_date
            )
            return result['total_bitcoin_needed']

        except Exception as e:
            return 0.0

    def generate_heatmap_data(self, grid_size: int = 30) -> tuple:
        """Generate data for the interactive heatmap."""

        # Create parameter ranges
        inflation_rates = np.linspace(0.02, 0.15, grid_size)  # 2% to 15%
        depreciation_rates = np.linspace(0.01, 0.10, grid_size)  # 1% to 10%

        # Initialize BTC requirements grid
        btc_grid = np.zeros((grid_size, grid_size))

        # Show progress
        progress_bar = st.progress(0)
        progress_text = st.empty()
        total_calculations = grid_size * grid_size

        # Calculate BTC requirements for each combination
        calculation_count = 0
        for i, inflation_rate in enumerate(inflation_rates):
            for j, depreciation_rate in enumerate(depreciation_rates):
                btc_grid[j, i] = self.calculate_btc_for_parameters(
                    inflation_rate, depreciation_rate)
                calculation_count += 1
                progress = calculation_count / total_calculations
                progress_bar.progress(progress)
                progress_text.text(
                    f"Calculating heatmap... {calculation_count}/{total_calculations} ({progress:.1%})")
        progress_bar.empty()
        progress_text.empty()
        return inflation_rates, depreciation_rates, btc_grid

    def create_interactive_heatmap(self, grid_size: int = 30) -> go.Figure:
        """Create the interactive heatmap visualization."""
        st.info("""
        **How to Read This Heatmap:**
        - **Darker colors** = Higher Bitcoin requirements
        - **Lighter colors** = Lower Bitcoin requirements  
        - **Hover anywhere** to see exact inflation rate, depreciation rate, and BTC needed
        - **3 scenarios** calculated earlier are marked as colored circles
        - **Click and zoom** to explore specific regions in detail
        """)

        # Generate heatmap data
        with st.spinner("Generating interactive heatmap for all parameter combinations..."):
            inflation_rates, depreciation_rates, btc_grid = self.generate_heatmap_data(
                grid_size)

        # Convert to percentages for display
        inflation_pct = inflation_rates * 100
        depreciation_pct = depreciation_rates * 100

        # Create the heatmap
        fig = go.Figure()

        # Add main heatmap
        fig.add_trace(go.Heatmap(
            z=btc_grid,
            x=inflation_pct,
            y=depreciation_pct,

            # Red-Yellow-Blue reversed (red = high, blue = low)
            colorscale='RdYlBu_r',
            hovertemplate=(
                "<b>Economic Parameters</b><br>" +
                "Inflation Rate: %{x:.1f}%<br>" +
                "USD Depreciation Rate: %{y:.1f}%<br>" +
                "Bitcoin Required: %{z:.4f} BTC<br>" +
                "<extra></extra>"
            ),

            colorbar=dict(

                title="Bitcoin Required (BTC)",
                thickness=20,
                len=0.8
            ),
            name="BTC Requirements"
        ))

        # Add scenario markers

        scenario_x = [self.scenarios[name]['inflation']
                      for name in self.scenarios]
        scenario_y = [self.scenarios[name]['depreciation']
                      for name in self.scenarios]
        scenario_names = ['Optimistic', 'Conservative', 'Extreme']
        scenario_colors = [self.scenarios[name]['color']
                           for name in self.scenarios]

        # Calculate BTC for each scenario to show in hover

        scenario_btc = []
        for name in self.scenarios:
            btc = self.calculate_btc_for_parameters(
                self.scenarios[name]['inflation'] / 100,
                self.scenarios[name]['depreciation'] / 100
            )
            scenario_btc.append(btc)

        # Add scenario points as scatter overlay

        fig.add_trace(go.Scatter(
            x=scenario_x,
            y=scenario_y,
            mode='markers+text',
            text=scenario_names,
            textposition="top center",
            textfont=dict(size=12, color='white'),
            marker=dict(
                size=20,
                color=scenario_colors,
                line=dict(width=3, color='white'),
                symbol='circle'
            ),

            name="Your Scenarios",

            hovertemplate=(
                "<b>%{text} Scenario</b><br>" +
                "Inflation Rate: %{x:.1f}%<br>" +
                "USD Depreciation Rate: %{y:.1f}%<br>" +
                "Bitcoin Required: %{customdata:.4f} BTC<br>" +
                "<extra></extra>"
            ),

            customdata=scenario_btc

        ))

        # Update layout

        fig.update_layout(

            title=dict(
                text=f"Bitcoin Requirements Across All Economic Scenarios<br>",
                font=dict(size=16),
            ),

            xaxis=dict(
                title="Annual Inflation Rate (%)",
                range=[2, 15],
                tickmode='linear',
                tick0=2,
                dtick=2
            ),

            yaxis=dict(

                title="USD/INR Annual Depreciation Rate (%)",
                range=[1, 10],
                tickmode='linear',
                tick0=1,
                dtick=1
            ),

            height=650,
            showlegend=True,
            legend=dict(
                yanchor="top",
                y=0.99,
                xanchor="left",
                x=1.02
            )

        )

        return fig

    def display_heatmap_insights(self):
        """Display insights about the heatmap patterns."""

        col1, col2, col3 = st.columns(3)

        with col1:

            st.success("""

            **🟦 Blue Regions (Low BTC)**
            - **Favorable Economic Conditions**
            - Lower inflation rates (2-6%)
            - Moderate currency depreciation (1-4%)
            - **Requires less Bitcoin accumulation**
            - Optimistic scenario for retirement planning
            """)

        with col2:

            st.warning("""

            **🟨 Yellow Regions (Moderate BTC)**
            - **Balanced Economic Conditions**
            - Moderate inflation rates (6-10%)
            - Standard currency depreciation (4-7%)
            - **Plan with standard scenarios**
            - Conservative scenario for retirement planning
            """)

        with col3:

            st.error("""
            **🟥 Red Regions (High BTC)**

            - **Challenging Economic Conditions**
            - High inflation rates (10-15%)
            - Significant currency depreciation (7-10%)
            - **Requires extra Bitcoin accumulation**
            - Extreme scenario for retirement planning
            """)

    def display_pattern_analysis(self):
        """Display analysis of patterns visible in the heatmap."""

        st.markdown("### Pattern Analysis")

        col1, col2 = st.columns(2)

        with col1:

            st.markdown("""

            **Key Patterns Visible:**
            - **Upper-left corner (High inflation, Low depreciation)**: Domestic supply constraints with strong currency fundamentals
            - **Lower-right corner (Low inflation, High depreciation)**: Currency crisis with controlled domestic prices
            - **Diagonal bands**: Shows interaction between inflation and currency effects
            """)

        with col2:

            st.markdown("""

            **Strategic Insights:**
            - **Sweet spot**: Lower-left region (low inflation + low depreciation)
            - **Danger zone**: Upper-right region (high inflation + high depreciation) 
            - **Currency dominance**: Vertical bands show depreciation has stronger impact
            """)

    def display_scenario_comparison_on_heatmap(self):
        """Display how the three scenarios compare on the heatmap."""

        st.markdown("### Your Scenarios on the Heatmap")

        # Calculate BTC for each scenario

        scenario_data = []
        for name, params in self.scenarios.items():
            btc_needed = self.calculate_btc_for_parameters(
                params['inflation'] / 100, params['depreciation'] / 100
            )

            scenario_data.append({

                'Scenario': f"{name.title()}",
                'Inflation Rate': f"{params['inflation']:.1f}%",
                'USD Depreciation': f"{params['depreciation']:.1f}%",
                'BTC Required': f"{btc_needed:.4f}",
                'Risk Zone': self._get_risk_zone(params['inflation'], params['depreciation'])
            })

        df_scenarios = pd.DataFrame(scenario_data)
        st.dataframe(df_scenarios, use_container_width=True, hide_index=True)
        st.info("""

        **Scenario Positioning:**
        - **Optimistic**: Positioned in the favorable blue-green zone
        - **Conservative**: Well-placed in the moderate yellow zone  
        - **Extreme**: Located in the challenging orange-red zone
        - **Coverage**: These three scenarios span the risk spectrum effectively
        """)

    def _get_risk_zone(self, inflation_rate: float, depreciation_rate: float) -> str:
        """Determine which risk zone the parameters fall into."""

        if inflation_rate <= 6 and depreciation_rate <= 3:
            return "🟦 Low Risk"
        elif inflation_rate <= 9 and depreciation_rate <= 6:
            return "🟨 Moderate Risk"
        else:
            return "🟥 High Risk"


def display_interactive_heatmap_chart(current_age: int, annual_expenditure_inr: float, retirement_year: int):
    """

    Main function to display the interactive heatmap with comprehensive analysis.

    """

    with st.expander("**Interactive Parameter Heatmap** Click to learn more", expanded=False):

        # Introduction
        st.markdown(f"""
        ### Complete Parameter Landscape Visualization
        This interactive heatmap shows **all possible economic scenarios** and their impact on your Bitcoin requirements.
        **Your Personal Parameters:**
        - **Age**: {current_age} years
        - **Annual Expenditure**: ₹{annual_expenditure_inr/100000:.1f} lakhs
        - **Target Retirement**: {retirement_year}
        """)

        # Create and display the heatmap

        heatmap = InteractiveHeatmap(
            current_age, annual_expenditure_inr, retirement_year)

        # Main heatmap visualization
        fig = heatmap.create_interactive_heatmap(grid_size=25)
        st.plotly_chart(fig, use_container_width=True)

        # Analysis sections
        heatmap.display_heatmap_insights()
        heatmap.display_pattern_analysis()
        heatmap.display_scenario_comparison_on_heatmap()
        st.info("""

        **How to Use This Heatmap:**

        1. **Overview**: Dark red areas require more Bitcoin, blue areas require less
        2. **Explore**: Hover anywhere to see exact inflation rate, depreciation rate, and BTC needed
        3. **Compare**: See how your three scenarios (marked circles) relate to the full landscape
        4. **Plan**: Identify economic conditions where your target Bitcoin amount is adequate
        5. **Strategize**: Use risk zones to understand worst-case and best-case scenarios
        """)
