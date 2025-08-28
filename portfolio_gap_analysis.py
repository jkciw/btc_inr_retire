"""
Portfolio Gap Analysis Module with Three SIP Scenarios

This module calculates SIP strategies to reach the Conservative Bitcoin target using
three scenarios with different USD/INR depreciation rates and Bitcoin percentiles.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
from typing import Dict, Any
from market_data import get_market_price
from config import SCENARIOS
from utils import indian_commas


class PortfolioGapAnalysis:
    """
    A class to analyze user's current Bitcoin position vs Conservative retirement target
    and provide three SIP scenarios with different assumptions.
    """

    def __init__(self, scenario_results: Dict[str, Any], current_age: int, retirement_year: int):
        self.scenario_results = scenario_results
        self.current_age = current_age
        self.retirement_year = retirement_year
        self.years_to_retirement = retirement_year - datetime.now().year
        self.genesis_date = datetime(2009, 1, 3)
        self._cache_market_data()

    def _cache_market_data(self):
        "Cache current market data for USD/INR and BTC price"
        with st.spinner("Fetching current market data..."):
            btc_price, usd_inr_rate = get_market_price()
        self.current_usd_inr_rate = usd_inr_rate if usd_inr_rate > 0 else 86.0

    def bitcoin_power_law_price(self, days_since_genesis: int, percentile: str = "trendline") -> float:
        """
        Calculate Bitcoin price using empirically derived percentile multipliers.
        """
        # Import the corrected functions
        from calculations import bitcoin_power_law_price_percentile, get_percentile_multipliers

        # Check if percentiles are initialized
        multipliers = get_percentile_multipliers()
        if multipliers is None:
            # Fallback to your hardcoded values if empirical data fails
            trendline_price = 1.42e-17 * (days_since_genesis ** 5.79)
            if percentile == "trendline":
                return trendline_price * 1.0
            elif percentile == "83.5":
                return trendline_price * 2.8
            elif percentile == "97.5":
                return trendline_price * 4.2
            else:
                return trendline_price

        # Use empirical percentiles
        if percentile == "trendline":
            return bitcoin_power_law_price_percentile(days_since_genesis, 50.0)
        elif percentile == "83.5":
            return bitcoin_power_law_price_percentile(days_since_genesis, 83.5)
        elif percentile == "97.5":
            return bitcoin_power_law_price_percentile(days_since_genesis, 97.5)
        else:
            return bitcoin_power_law_price_percentile(days_since_genesis, 50.0)

    def usd_inr_exchange_rate(self, years_from_now: float, annual_depreciation_rate: float) -> float:
        """
        Project USD/INR exchange rate based on annual depreciation rate.
        """
        current_market_rate = self.current_usd_inr_rate

        # Apply compound depreciation (INR weakens against USD)
        future_rate = current_market_rate * \
            ((1 + annual_depreciation_rate) ** years_from_now)
        return future_rate

    def get_user_btc_holdings(self):
        """Get user's current Bitcoin holdings through UI."""
        st.markdown(
            "### Enter Your Exisitng Bitcoins Earmarked for Retirement")

        current_btc = st.number_input(
            "Bitcoin Holdings",
            min_value=0.0,
            max_value=100.0,
            value=0.0,
            step=0.0005,
            format="%.6f",
            help="Enter the amount of Bitcoin you currently have specifically for retirement planning",
            label_visibility="hidden",
            key="user_btc_holdings"
        )
        return current_btc

    def analyze_portfolio_gap(self, current_btc: float) -> Dict[str, Any]:
        """Analyze the gap between current holdings and Conservative retirement target."""
        conservative_needed = self.scenario_results['Conservative']['total_bitcoin_needed']
        optimistic_needed = self.scenario_results['Optimistic']['total_bitcoin_needed']
        extreme_needed = self.scenario_results['Extreme']['total_bitcoin_needed']

        # Calculate gap only for conservative target (our focus)
        conservative_gap = max(0, conservative_needed - current_btc)

        # Determine user type based on conservative target
        if current_btc >= conservative_needed:
            user_type = "fully_covered"
        elif current_btc > 0:
            user_type = "partial_holdings"
        else:
            user_type = "starting_fresh"

        # Calculate progress towards conservative target
        conservative_progress = min(
            100, (current_btc / conservative_needed) * 100) if conservative_needed > 0 else 100

        return {
            'current_btc': current_btc,
            'conservative_needed': conservative_needed,
            'optimistic_needed': optimistic_needed,
            'extreme_needed': extreme_needed,
            'conservative_gap': conservative_gap,
            'conservative_progress': conservative_progress,
            'user_type': user_type
        }

    def calculate_three_sip_scenarios(self, btc_gap: float) -> Dict[str, Any]:
        """
        Calculate SIP strategies for three scenarios to reach Conservative Bitcoin target.

        Optimistic: 5% USD/INR depreciation + Power Law Trendline
        Conservative: 3% USD/INR depreciation + 83.5th Percentile BTC Price  
        Extreme: 3% USD/INR depreciation + 97.5th Percentile BTC Price
        """
        if btc_gap <= 0:
            return {
                'optimistic': self._empty_sip_result(),
                'conservative': self._empty_sip_result(),
                'extreme': self._empty_sip_result()
            }

        scenarios = {
            'optimistic': self._calculate_scenario_sip(btc_gap, 0.05, "trendline", "Optimistic"),
            'conservative': self._calculate_scenario_sip(btc_gap, 0.03, "83.5", "Conservative"),
            'extreme': self._calculate_scenario_sip(btc_gap, 0.02, "97.5", "Extreme")
        }

        return scenarios

    def _empty_sip_result(self):
        """Return empty SIP result structure."""
        return {
            'monthly_sip_inr': 0,
            'weekly_sip_inr': 0,
            'total_investment_inr': 0,
            'monthly_details': [],
            'avg_btc_price_inr': 0,
            'avg_usd_inr_rate': 0,
            'btc_gap': 0,
            'months_to_retirement': 0,
            'weeks_to_retirement': 0,
            'scenario_name': '',
            'usd_inr_depreciation': 0,
            'btc_percentile': ''
        }

    def _calculate_scenario_sip(self, btc_gap: float, usd_inr_depreciation: float, btc_percentile: str, scenario_name: str) -> Dict[str, Any]:
        """Calculate SIP for a specific scenario."""
        current_date = datetime.now()
        retirement_date = datetime(self.retirement_year, 1, 1)
        current_days_since_genesis = (current_date - self.genesis_date).days

        # Calculate total months and weeks to retirement
        months_to_retirement = max(1, self.years_to_retirement * 12)
        weeks_to_retirement = max(
            1, int((retirement_date - current_date).days / 7))

        # Calculate month-by-month pricing for this scenario
        monthly_details = []
        total_btc_cost_inr = 0
        total_usd_inr_rate = 0

        for month in range(months_to_retirement):
            # Calculate days since genesis for this month (30-day intervals)
            last_day_of_this_month = current_days_since_genesis + (month * 30)
            years_from_now = month / 12.0

            # Get Bitcoin price for this specific month and percentile
            btc_price_usd = self.bitcoin_power_law_price(
                last_day_of_this_month, btc_percentile)

            # Get USD/INR rate for this time period with specified depreciation
            usd_inr_rate = self.usd_inr_exchange_rate(
                years_from_now, usd_inr_depreciation)
            total_usd_inr_rate += usd_inr_rate

            # Convert to INR
            btc_price_inr = btc_price_usd * usd_inr_rate

            # Calculate how much BTC we need to buy this month
            btc_to_buy_this_month = btc_gap / months_to_retirement
            cost_this_month_inr = btc_to_buy_this_month * btc_price_inr

            total_btc_cost_inr += cost_this_month_inr

            monthly_details.append({
                'month': month + 1,
                'date': (current_date + timedelta(days=month*30)).strftime('%Y-%m'),
                'days_since_genesis': last_day_of_this_month,
                'btc_price_usd': btc_price_usd,
                'usd_inr_rate': usd_inr_rate,
                'btc_price_inr': btc_price_inr,
                'btc_to_buy': btc_to_buy_this_month,
                'cost_inr': cost_this_month_inr
            })

        # Calculate averages and SIP amounts
        avg_btc_price_inr = total_btc_cost_inr / btc_gap if btc_gap > 0 else 0
        avg_usd_inr_rate = total_usd_inr_rate / \
            months_to_retirement if months_to_retirement > 0 else 84.0
        monthly_sip_inr = total_btc_cost_inr / months_to_retirement
        weekly_sip_inr = total_btc_cost_inr / weeks_to_retirement

        return {
            'monthly_sip_inr': monthly_sip_inr,
            'weekly_sip_inr': weekly_sip_inr,
            'total_investment_inr': total_btc_cost_inr,
            'avg_btc_price_inr': avg_btc_price_inr,
            'avg_usd_inr_rate': avg_usd_inr_rate,
            'btc_gap': btc_gap,
            'months_to_retirement': months_to_retirement,
            'weeks_to_retirement': weeks_to_retirement,
            'monthly_details': monthly_details,
            'scenario_name': scenario_name,
            'usd_inr_depreciation': usd_inr_depreciation,
            'btc_percentile': btc_percentile
        }

    def display_portfolio_status(self, analysis: Dict[str, Any]):
        """Display user's portfolio status focusing on Conservative target."""
        st.markdown("### Your Retirement Readiness Status")

        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric(
                label="Current Holdings",
                value=f"{analysis['current_btc']:.6f} BTC",
                help="Your current Bitcoin position for retirement"
            )

        with col2:
            st.metric(
                label="Conservative Target",
                value=f"{analysis['conservative_needed']:.6f} BTC",
                delta=f"{-analysis['conservative_gap']:.6f} BTC" if analysis['conservative_gap'] > 0 else "✅ Target Met",
                delta_color="inverse" if analysis['conservative_gap'] > 0 else "normal"
            )

        with col3:
            progress_color = "🟢" if analysis['conservative_progress'] >= 100 else "🟡" if analysis['conservative_progress'] >= 50 else "🔴"
            st.metric(
                label="Progress to Target",
                value=f"{progress_color} {analysis['conservative_progress']:.1f}%",
                help="Percentage of conservative target achieved"
            )

        with col4:
            years_left = self.years_to_retirement
            st.metric(
                label="Years to Retirement",
                value=f"{years_left} years",
                help="Time remaining for accumulation"
            )

    def display_sip_analysis(self, analysis: Dict[str, Any]):
        """Display SIP analysis based on user's current position."""
        user_type = analysis['user_type']

        if user_type == "fully_covered":
            self.show_congratulations_message(analysis)
        else:
            # Calculate three SIP scenarios for conservative gap
            sip_scenarios = self.calculate_three_sip_scenarios(
                analysis['conservative_gap'])

            # Show scenario comparison
            self.show_three_scenario_comparison(sip_scenarios, analysis)

            # Show detailed breakdown
            self.show_detailed_sip_breakdown(
                analysis, sip_scenarios, user_type)

            # Show implementation guidance
            self.show_implementation_guidance(sip_scenarios)

    def show_congratulations_message(self, analysis: Dict[str, Any]):
        """Display congratulations for users who have enough Bitcoin."""
        st.success(
            "**Congratulations! You have achieved your Conservative Bitcoin retirement target!**")

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("""
            **You're financially prepared:**
            - You have enough Bitcoin for conservative retirement planning
            - You've reached the target based on power law projections
            - You can focus on protecting and managing your Bitcoin
            """)

        with col2:
            st.markdown("""
            **Next steps to consider:**
            - **Secure Storage**: Move Bitcoin to cold storage
            - **Estate Planning**: Plan Bitcoin inheritance
            """)

        # Show bonus if they exceed even extreme target
        if analysis['current_btc'] >= analysis['extreme_needed']:
            excess_btc = analysis['current_btc'] - analysis['extreme_needed']
            st.info(
                f"**Bonus**: You have {excess_btc:.6f} BTC above even the extreme scenario requirements!")

    def show_three_scenario_comparison(self, sip_scenarios: Dict[str, Any], analysis: Dict[str, Any]):
        """Display comparison table for three SIP scenarios."""
        st.markdown(
            "### Three SIP Scenarios to Reach Your Conservative BTC Target")
        st.markdown(
            f"*Target: {analysis['conservative_needed']:.6f} BTC | Gap: {analysis['conservative_gap']:.6f} BTC*")
        st.markdown("#### Total Investment Comparison")
        try:
            current_btc_price, _ = get_market_price()
            if current_btc_price <= 0:
                current_btc_price = 100000
        except:
            current_btc_price = 100000

        # Create comparison table
        comparison_data = []
        for scenario_key, sip_data in sip_scenarios.items():
            if sip_data['btc_gap'] > 0:
                scenario_details = self._get_scenario_details(scenario_key)
                comparison_data.append({
                    "Scenario": scenario_details['name'],
                    "Assumed BTC Price Model": scenario_details['btc_model'],
                    "Assumed USD/INR Depreciation": f"{sip_data['usd_inr_depreciation']*100:.1f}% annually",
                    "Monthly SIP": f"₹{indian_commas(sip_data['monthly_sip_inr'], 0)}",
                    "Weekly SIP": f"₹{indian_commas(sip_data['weekly_sip_inr'], 0)}",
                    "Total Investment": f"₹{indian_commas(sip_data['total_investment_inr'], 0)}",
                })

        if comparison_data:
            df_comparison = pd.DataFrame(comparison_data)
            st.dataframe(df_comparison,
                         hide_index=True)

        # Show waterfall chart
        st.markdown("View Detailed Cost Difference Analysis")
        waterfall_fig = self.create_cost_efficiency_summary_chart(
            sip_scenarios,
            current_btc_price,
            analysis['conservative_gap']
        )
        st.plotly_chart(waterfall_fig, width='stretch')

        # Add detailed scenario explanations
        if comparison_data:
            col1, col2, col3 = st.columns(3)

            with col1:
                st.info("""
                **🟢 Optimistic Scenario**
                - USD/INR depreciates 5% annually
                - Bitcoin follows power law trendline
                - SIP planning based on hopeful outlook 
                - Slowest path to reach target
                """)

            with col2:
                st.warning("""
                **🟡 Conservative Scenario**  
                - USD/INR depreciates 3% annually
                - Bitcoin follows 83.5th percentile price
                - SIP planning based on realistic outlook
                - Balanced path to reach target
                """)

            with col3:
                st.error("""
                **🔴 Extreme Scenario**
                - USD/INR depreciates 2% annually  
                - Bitcoin at 97.5th percentile price
                - SIP planning based on pessimistic outlook
                - Fastest path to reach target
                """)

    def _get_scenario_details(self, scenario_key: str) -> Dict[str, str]:
        """Get display details for each scenario."""
        details = {
            'optimistic': {
                'name': '🟢 Optimistic',
                'btc_model': 'Power Law Trendline'
            },
            'conservative': {
                'name': '🟡 Conservative',
                'btc_model': '83.5th Percentile'
            },
            'extreme': {
                'name': '🔴 Extreme',
                'btc_model': '97.5th Percentile'
            }
        }
        return details.get(scenario_key, {'name': 'Unknown', 'btc_model': 'Unknown'})

    def show_detailed_sip_breakdown(self, analysis: Dict[str, Any], sip_scenarios: Dict[str, Any], user_type: str):
        """Show detailed breakdown for selected scenario."""
        st.markdown("### Detailed Scenario Analysis")

        with st.expander("Click to view detailed SIP breakdown", expanded=False):
            # Let user select which scenario to analyze in detail
            selected_scenario = st.selectbox(
                "Select scenario for detailed monthly breakdown:",
                options=['conservative', 'optimistic', 'extreme'],
                index=0,
                format_func=lambda x: self._get_scenario_details(x)[
                    'name'] + " Scenario"
            )

            selected_sip = sip_scenarios[selected_scenario]

            # Show monthly breakdown
            self.show_monthly_breakdown(selected_sip)

    def show_starting_fresh_details(self, analysis: Dict[str, Any], sip_data: Dict[str, Any]):
        """Show details for users starting from zero."""
        st.markdown(
            f"### Starting Fresh - {sip_data['scenario_name']} Scenario")

        col1, col2 = st.columns(2)

        with col1:
            st.markdown(f"""
            **Your Bitcoin Accumulation Journey:**
            - **Target**: {analysis['conservative_needed']:.6f} BTC
            - **Timeline**: {self.years_to_retirement} years
            - **Total Investment**: ₹{indian_commas(sip_data['total_investment_inr'], 0)}
            - **Strategy**: {sip_data['scenario_name']} approach
            """)

        with col2:
            st.markdown(f"""
            **SIP Requirements:**
            
            **Monthly SIP**: ₹{indian_commas(sip_data['monthly_sip_inr'], 0)}
            
            **Weekly SIP**: ₹{indian_commas(sip_data['weekly_sip_inr'], 0)}
            
            **Avg BTC Price**: ₹{indian_commas(sip_data['avg_btc_price_inr'], 0)}
            
            **Final USD/INR**: {sip_data['avg_usd_inr_rate']:.0f}
            """)

    def show_partial_holdings_details(self, analysis: Dict[str, Any], sip_data: Dict[str, Any]):
        """Show details for users with partial holdings."""
        st.markdown(
            f"### Complete Your Journey - {sip_data['scenario_name']} Scenario")

        col1, col2 = st.columns(2)

        with col1:
            st.markdown(f"""
            **Your Current Position:**
            - **Current**: {analysis['current_btc']:.6f} BTC
            - **Target**: {analysis['conservative_needed']:.6f} BTC
            - **Remaining**: {analysis['conservative_gap']:.6f} BTC
            - **Progress**: {analysis['conservative_progress']:.1f}% complete
            """)

        with col2:
            st.markdown(f"""
            **Additional SIP Required:**
            
            **Monthly SIP**: ₹{indian_commas(sip_data['monthly_sip_inr']), 0}
            
            **Weekly SIP**: ₹{indian_commas(sip_data['weekly_sip_inr']), 0}
            
            **Total Additional**: ₹{indian_commas(sip_data['total_investment_inr']), 0}
            
            **Timeline**: {self.years_to_retirement} years
            """)

    def show_monthly_breakdown(self, sip_data: Dict[str, Any]):
        """Display month-by-month breakdown for selected scenario."""
        if not sip_data['monthly_details']:
            return

        st.markdown(
            f"### Monthly Breakdown - {sip_data['scenario_name']} Scenario")

        # Create DataFrame from monthly details
        monthly_df = pd.DataFrame(sip_data['monthly_details'])

        # Format for display
        display_df = monthly_df.copy()
        display_df['Month'] = display_df['month']
        display_df['Date'] = display_df['date']
        display_df['BTC Price (USD)'] = display_df['btc_price_usd'].apply(
            lambda x: f"${x:,.0f}")
        display_df['USD/INR Rate'] = display_df['usd_inr_rate'].apply(
            lambda x: f"{x:.1f}")
        display_df['BTC Price (₹)'] = display_df['btc_price_inr'].apply(
            lambda x: f"₹{x:,.0f}")
        display_df['Monthly SIP (₹)'] = display_df['cost_inr'].apply(
            lambda x: f"₹{x:,.0f}")

        display_columns = [
            'Month', 'Date', 'BTC Price (USD)', 'USD/INR Rate', 'BTC Price (₹)', 'Monthly SIP (₹)']

        # Show sample months for long timelines
        total_months = len(display_df)
        if total_months > 24:
            st.markdown(
                f"**Sample months (showing first 12 and last 12 of {total_months} total months):**")

            col1, col2 = st.columns(2)
            with col1:
                st.markdown("**First 12 months:**")
                st.dataframe(display_df[display_columns].head(
                    12), hide_index=True)

            with col2:
                st.markdown("**Last 12 months:**")
                st.dataframe(display_df[display_columns].tail(
                    12), hide_index=True)
        else:
            st.dataframe(display_df[display_columns],
                         hide_index=True)

    def show_implementation_guidance(self, sip_scenarios: Dict[str, Any]):
        """Show practical implementation guidance."""
        st.markdown("### Implementation Strategy")

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("""
            **Buy Bitcoin Now:**
            - Lumpsum purchase saves you big time
            - Buy here:
            """)

        with col2:
            st.markdown("""
            **Setup a SIP:**
            - SIPs are a useful way to accumulate Bitcoins
            - Setup SIP here: 
            """)

        # Show action checklist
        st.markdown("### Action Checklist")

        checklist_items = [
            "Choose your preferred SIP scenario (Conservative recommended)",
            "Select a reputable exchange (getbit, Unocoin)",
            "Set up automatic bank transfers for SIP amounts",
            "Configure recurring Bitcoin purchases",
            "Set up secure cold storage wallet",
            "Create tracking spreadsheet for progress monitoring",
            "Plan for tax implications and record keeping",
            "Schedule quarterly reviews and adjustments"
        ]

        for item in checklist_items:
            st.checkbox(item, key=f"checklist_{item[:20]}")

    # Waterfall chart for cost differences from lump sum baseline.

    def create_cost_efficiency_summary_chart(self, sip_scenarios: Dict[str, Any], current_btc_price: float, btc_gap: float) -> go.Figure:
        """
        Waterfall chart showing cost differences from lump sum baseline.
        With proper Indian comma formatting, clean y-axis ticks, and well-positioned total costs.
        """
        # Calculate lump sum cost
        lump_sum_cost_usd = btc_gap * current_btc_price
        lump_sum_cost_inr = lump_sum_cost_usd * self.current_usd_inr_rate

        # Prepare waterfall data
        categories = ['Lump Sum\nBaseline']
        values = [lump_sum_cost_inr]
        measure = ['absolute']

        scenario_names = {
            'optimistic': 'Optimistic\nSIP',
            'conservative': 'Conservative\nSIP',
            'extreme': 'Extreme\nSIP'
        }

        # Store total costs for annotations
        scenario_totals = {}
        text_labels = [f"₹{indian_commas(lump_sum_cost_inr, 0)}"]

        for scenario_key in ['optimistic', 'conservative', 'extreme']:
            if scenario_key in sip_scenarios:
                sip_cost = sip_scenarios[scenario_key]['total_investment_inr']
                difference = sip_cost - lump_sum_cost_inr

                # Store total cost for this scenario
                scenario_totals[scenario_key] = sip_cost

                categories.append(scenario_names[scenario_key])
                values.append(difference)
                measure.append('relative')

                if difference >= 0:
                    text_labels.append(
                        f"+₹{indian_commas(abs(difference), 0)}")
                else:
                    text_labels.append(
                        f"-₹{indian_commas(abs(difference), 0)}")

        fig = go.Figure(go.Waterfall(
            name="Cost Comparison",
            orientation="v",
            measure=measure,
            x=categories,
            textposition="outside",
            text=text_labels,
            y=values,
            connector={"line": {"color": "rgb(63, 63, 63)"}},
            hovertemplate="<b>%{x}</b><br>" +
            "Amount: ₹%{y:,.0f}<br>" +
            "<extra></extra>"
        ))

        # Calculate proper y-axis range and ticks
        all_values = values.copy()
        # Add cumulative values to get the full range
        cumulative = lump_sum_cost_inr
        cumulative_values = [cumulative]
        for i, diff in enumerate(values[1:]):  # Skip baseline
            cumulative += diff
            cumulative_values.append(cumulative)
            all_values.append(cumulative)

        y_min = min(all_values) * 0.9  # Add 10% padding below
        y_max = max(all_values) * 1.2  # Add 20% padding above for annotations

        # Create clean, round y-axis ticks
        y_range = y_max - y_min

        # Determine appropriate tick step (round numbers)
        if y_range > 1e9:  # > 100 crores
            tick_step = 2e8  # 20 crore steps
        elif y_range > 5e8:  # > 50 crores
            tick_step = 1e8   # 10 crore steps
        elif y_range > 1e8:  # > 10 crores
            tick_step = 5e7   # 5 crore steps
        elif y_range > 5e7:  # > 5 crores
            tick_step = 1e7   # 1 crore steps
        else:
            tick_step = 5e6   # 50 lakh steps

        # Generate tick values starting from a round number
        first_tick = (int(y_min / tick_step) - 1) * tick_step
        last_tick = (int(y_max / tick_step) + 1) * tick_step

        tick_vals = []
        tick_text = []

        current_tick = first_tick
        while current_tick <= last_tick:
            tick_vals.append(current_tick)

            if current_tick >= 0:
                tick_text.append(f"₹{indian_commas(current_tick, 0)}")
            else:
                tick_text.append(f"-₹{indian_commas(abs(current_tick), 0)}")

            current_tick += tick_step

        # Add annotations with total costs - positioned better
        for i, scenario_key in enumerate(['optimistic', 'conservative', 'extreme']):
            if scenario_key in scenario_totals:
                x_pos = scenario_names[scenario_key]

                # Position annotations at 85% of the chart height (below the top)
                annotation_y = y_max * 1

                fig.add_annotation(
                    x=x_pos,
                    y=annotation_y,
                    text=f"Total Cost:<br>₹{indian_commas(scenario_totals[scenario_key], 0)}",
                    showarrow=False,
                    font=dict(size=10, color='darkblue'),
                    bgcolor="rgba(255,255,255,0.9)",
                    bordercolor="darkblue",
                    borderwidth=1,
                    xanchor='center',
                    yanchor='middle'
                )

        fig.update_layout(
            title="Cost Difference Analysis: SIP Strategies vs Lump Sum",
            xaxis_title="Strategy",
            yaxis_title="Extra sum required (₹)",
            height=500,
            yaxis=dict(
                tickmode='array',
                tickvals=tick_vals,
                ticktext=tick_text,
                range=[y_min, y_max]  # Set explicit range
            )
        )

        return fig


def display_portfolio_gap_analysis(scenario_results: Dict[str, Any], current_age: int, retirement_year: int):
    """
    Main function to display portfolio gap analysis with three SIP scenarios.
    """
    st.markdown("---")

    # Initialize gap analysis
    gap_analyzer = PortfolioGapAnalysis(
        scenario_results, current_age, retirement_year)

    # Get user's current holdings
    current_btc = gap_analyzer.get_user_btc_holdings()

    if current_btc is not None:
        # Analyze the gap
        analysis = gap_analyzer.analyze_portfolio_gap(current_btc)

        # Display portfolio status
        gap_analyzer.display_portfolio_status(analysis)

        # Display SIP analysis
        gap_analyzer.display_sip_analysis(analysis)
