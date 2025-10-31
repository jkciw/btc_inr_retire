# Currency Analysis Module
# Uses actual USD/INR historical data from FRED to analyze CAGR

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from pandas.tseries.offsets import MonthEnd
from datetime import datetime
from config import SCENARIOS


class CurrencyAnalysisChart:

    def __init__(self, csv_path: str = "usd_inr_historical.csv"):
        self.csv_path = csv_path
        self.usd_inr_daily: pd.Series | None = None
        self.usd_inr_monthly: pd.Series | None = None
        self.usd_inr_annual_eoy: pd.Series | None = None

    def load_historical_usd_inr_data(self) -> pd.Series:
        """
        Load, clean, and resample the USD/INR series:
          1) Parse dates (dayfirst), coerce numeric, drop invalids
          2) Resample to daily and ffill/bfill to repair gaps
          3) Aggregate to month-end mean for plotting
          4) Build year-end for CAGR
        Returns: month-end series (pd.Series)
        """
        df = pd.read_csv(self.csv_path, parse_dates=[
                         "observation_date"], dayfirst=True)
        df["DEXINUS"] = pd.to_numeric(df["DEXINUS"], errors="coerce")
        df = df.dropna(subset=["observation_date"]).sort_values(
            "observation_date").set_index("observation_date")
        df = df[df["DEXINUS"] > 0]

        # Fill patchy dates
        daily = df["DEXINUS"].resample("D").ffill().bfill()

        # Month‑end average for smoother historical curve
        monthly = daily.resample("ME").mean()

        # Year‑end (December) for CAGR tables
        annual_eoy = monthly.resample("YE-DEC").last()

        self.usd_inr_daily = daily
        self.usd_inr_monthly = monthly
        self.usd_inr_annual_eoy = annual_eoy
        return monthly

    def _safe_year_value(self, series: pd.Series, year: int) -> float | None:
        if series is None or series.empty:
            return None
        hit = series[series.index.year == year]
        if hit.empty:
            return None
        # take the last value for that calendar year (or use 0 for the first)
        return float(hit.iloc[-1])

    def calculate_cagr_analysis(self) -> pd.DataFrame:
        if self.usd_inr_annual_eoy is None:
            self.load_historical_usd_inr_data()

        s = self.usd_inr_annual_eoy.dropna()
        if s.empty:
            return pd.DataFrame()

        first_year = int(s.index.year.min())
        last_year = int(s.index.year.max())

        periods = [
            {"name": f"Full Period ({first_year}-{last_year})",
             "start": first_year, "end": last_year},
            {"name": "Post Great Financial Crisis(2008 - 2025)",
             "start": 2008, "end": 2025},
            {"name": f"Last 20 Years ({max(last_year-20, first_year)}-{last_year})",
             "start": max(last_year-20, first_year), "end": last_year},
            {"name": f"Last 5 Years ({max(last_year-5, first_year)}-{last_year})",
             "start": max(last_year-5, first_year), "end": last_year},
        ]

        rows = []
        for p in periods:
            start_val = self._safe_year_value(s, p["start"])
            end_val = self._safe_year_value(s, p["end"])
            years = p["end"] - p["start"]
            if start_val is not None and end_val is not None and years > 0:
                usd_app_cagr = (end_val / start_val) ** (1/years) - 1.0
                inr_dep_cagr = ((1/end_val) / (1/start_val)) ** (1/years) - 1.0
                rows.append({
                    "Period": p["name"],
                    "Years": years,
                    "Start Rate": start_val,
                    "End Rate": end_val,
                    "USD Appreciation CAGR(%)": usd_app_cagr * 100,
                    "INR Depreciation CAGR(%)": inr_dep_cagr * 100
                })
        return pd.DataFrame(rows)

    def create_historical_chart(self) -> go.Figure:
        """
        Plot monthly historical USD/INR and add monthly projections by converting 
        each scenario's annual depreciation_rate into a monthly compounding rate.
        """
        if self.usd_inr_monthly is None:
            self.load_historical_usd_inr_data()

        hist = self.usd_inr_monthly
        last_date = hist.index[-1]
        last_rate = float(hist.iloc[-1])

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=hist.index,
            y=hist.values,
            mode="lines",
            name="USD/INR (Monthly, avg)",
            line=dict(width=3, color="#1f77b4"),
            hovertemplate="Date: %{x|%b %Y}<br>USD/INR: %{y:.2f}<extra></extra>"
        ))

        # Projections: monthly compounding
        proj_dates = pd.date_range(
            start=(last_date + MonthEnd(1)), end="2050-12-31", freq="ME")
        months = np.arange(1, len(proj_dates) + 1)

        for scenario_name, params in SCENARIOS.items():
            annual = params.get("depreciation_rate", params.get(
                "usd_appreciation_rate", 0.0))
            mrate = (1.0 + annual) ** (1.0/12.0) - 1.0
            proj_vals = last_rate * (1.0 + mrate) ** months
            fig.add_trace(go.Scatter(
                x=proj_dates,
                y=proj_vals,
                mode="lines",
                name=f"{annual*100} % Appreciation",
                line=dict(width=2, dash="dash",
                          color=params.get("color", "#888")),
                hovertemplate=f"{scenario_name}<br>Date: %{{x|%b %Y}}<br>USD/INR: %{{y:.2f}}<extra></extra>"
            ))

        # Divider at last historical date: use shape + annotation (avoids add_vline datetime mean bug)
        fig.add_shape(
            type="line",
            x0=last_date, x1=last_date,
            y0=0, y1=1,
            xref="x", yref="paper",
            line=dict(color="gray", width=2, dash="dot")
        )
        fig.add_annotation(
            x=last_date, y=1.02, xref="x", yref="paper",
            text="← Historical | Projected →",
            showarrow=False, align="center", font=dict(color="gray")
        )

        fig.update_layout(
            title=dict(
                text="USD/INR Exchange Rate: Historical and Projections",
                font=dict(size=18)
            ),
            xaxis_title="Date",
            yaxis_title="USD/INR (INR per 1 USD)",
            height=620,
            showlegend=True,
            legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01),
            hovermode="x unified"
        )
        return fig


def display_currency_analysis_page():
    st.markdown("# 💹 Currency Analysis")
    st.markdown(
        "Understanding how USD/INR has moved over the years and how it might move in the future (three scenarios)")
    analyzer = CurrencyAnalysisChart("usd_inr_historical.csv")

    st.markdown("## Historical USD/INR with projections")
    fig = analyzer.create_historical_chart()
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("## CAGR Analysis (Year‑end values)")
    cagr_df = analyzer.calculate_cagr_analysis()
    if not cagr_df.empty:
        df_show = cagr_df.copy()
        df_show["USD Appreciation CAGR(%)"] = df_show["USD Appreciation CAGR(%)"].map(
            lambda v: f"{v:.2f}%")
        df_show["INR Depreciation CAGR(%)"] = df_show["INR Depreciation CAGR(%)"].map(
            lambda v: f"{v:.2f}%")
        df_show["Start Rate"] = df_show["Start Rate"].map(lambda v: f"{v:.2f}")
        df_show["End Rate"] = df_show["End Rate"].map(lambda v: f"{v:.2f}")
        st.dataframe(df_show, hide_index=True)
    else:
        st.warning(
            "CAGR table unavailable because boundary years are missing in the data.")
