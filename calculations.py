"""
Bitcoin Power Law and Financial Calculations Module - CORRECTED VERSION

This module contains core financial calculations including:
- Bitcoin power law price predictions with mathematically derived percentiles
- USD/INR exchange rate projections
- Retirement scenario calculations

"""

import numpy as np
import pandas as pd
from datetime import datetime
from typing import Tuple, Dict, Any, Optional, List
import statsmodels.api as sm
import streamlit as st
from config import SCENARIOS


class BitcoinPowerLawCalculator:
    """
    Mathematically rigorous Bitcoin Power Law calculator with empirically derived percentiles.
    """

    def __init__(self):
        self.genesis_date = datetime(2009, 1, 3)
        self.a = 1.42e-17  # Power law coefficient
        self.b = 5.79      # Power law exponent
        self.percentile_multipliers = None
        self.is_fitted = False

    def power_law_price(self, days_since_genesis: int) -> float:
        """Calculate power law trendline price."""
        return self.a * (days_since_genesis ** self.b)

    def load_historical_data(self, csv_file_path: str) -> List[Tuple[datetime, float]]:
        """
        Load historical Bitcoin price data from CSV file.
        CORRECTED for coinmcap_consolidated.csv format with month-end sampling.

        Args:
            csv_file_path: Path to CSV file with Bitcoin price data

        Returns:
            List of (datetime, price) tuples
        """
        try:
            df = pd.read_csv(csv_file_path)

            # For your specific CSV format - expect timeClose and close columns
            date_col = 'timeClose'
            price_col = 'close'

            # Check if expected columns exist
            if date_col not in df.columns or price_col not in df.columns:
                print(f"Expected columns 'timeClose' and 'close' not found.")
                print(f"Available columns: {list(df.columns)}")
                # Try fallback column names
                for col in ['date', 'Date', 'DATE', 'timestamp', 'Timestamp', 'time']:
                    if col in df.columns:
                        date_col = col
                        break
                for col in ['price', 'Price', 'PRICE', 'value', 'Value']:
                    if col in df.columns:
                        price_col = col
                        break

                if date_col not in df.columns or price_col not in df.columns:
                    # Last fallback to first two columns
                    date_col = df.columns[0]
                    price_col = df.columns[1]

                print(
                    f"Using columns: {date_col} (date) and {price_col} (price)")

            # Convert to datetime and float
            df[date_col] = pd.to_datetime(df[date_col])

            # MONTH-END SAMPLING - Apply resample only after setting index properly
            df = df.set_index(date_col).resample(
                'ME').last().reset_index()  # Changed 'M' to 'ME'

            # Clean timezone info if present
            df[date_col] = df[date_col].dt.tz_convert(
                None) if df[date_col].dt.tz else df[date_col]
            df[date_col] = df[date_col].dt.tz_localize(
                None) if df[date_col].dt.tz else df[date_col]

            df[price_col] = pd.to_numeric(df[price_col], errors='coerce')

            # Remove rows with invalid data
            df = df.dropna(subset=[date_col, price_col])
            df = df[df[price_col] > 0]  # Remove zero or negative prices

            # Sort by date to ensure proper ordering
            df = df.sort_values(date_col)

            # Convert to list of tuples (datetime, price)
            historical_data = []
            for _, row in df.iterrows():
                date_obj = row[date_col].to_pydatetime() if hasattr(
                    row[date_col], 'to_pydatetime') else row[date_col]
                if hasattr(date_obj, 'replace'):
                    date_obj = date_obj.replace(tzinfo=None)
                price_val = float(row[price_col])
                historical_data.append((date_obj, price_val))

            print(
                f"✅ Loaded {len(historical_data)} historical price data points")
            if len(historical_data) > 0:
                print(
                    f"Date range: {historical_data[0][0]} to {historical_data[-1][0]}")
                prices = [p[1] for p in historical_data]
                print(
                    f"Price range: ${min(prices):,.2f} to ${max(prices):,.2f}")

            return historical_data

        except Exception as e:
            raise ValueError(f"Error loading CSV file: {str(e)}")

    def fit_percentiles_from_csv(self, csv_file_path: str) -> Dict[float, float]:
        """
        Fit percentile multipliers from CSV file with historical Bitcoin price data.

        Args:
            csv_file_path: Path to CSV file

        Returns:
            Dictionary with percentile multipliers
        """
        print(f"Loading historical data from: {csv_file_path}")
        historical_data = self.load_historical_data(csv_file_path)

        if not historical_data:
            raise ValueError("No historical data loaded from CSV file")

        print(
            f"Fitting percentiles from {len(historical_data)} data points...")
        return self.fit_percentiles(historical_data)

    def fit_percentiles(self, historical_data: List[Tuple[datetime, float]]) -> Dict[float, float]:
        """
        Fit percentile multipliers from historical Bitcoin price data.

        Args:
            historical_data: List of (datetime, price) tuples

        Returns:
            Dictionary with percentile multipliers
        """
        ratios = []

        for date, price in historical_data:
            days = (date - self.genesis_date).days
            if days > 0:  # Ensure positive days since genesis
                trendline = self.power_law_price(days)
                if trendline > 0 and price > 0:  # Valid data points
                    ratio = price / trendline
                    ratios.append(ratio)

        if len(ratios) == 0:
            raise ValueError("No valid historical data points found")

        ratios = np.array(ratios)

        # Calculate percentiles using exact mathematical formula
        self.percentile_multipliers = {
            2.5: np.percentile(ratios, 2.5),
            16.5: np.percentile(ratios, 16.5),
            50.0: np.percentile(ratios, 50.0),
            83.5: np.percentile(ratios, 83.5),
            97.5: np.percentile(ratios, 97.5)
        }

        self.is_fitted = True

        # Print results for verification
        print("✅ Calculated percentile multipliers:")
        for p, mult in self.percentile_multipliers.items():
            print(f"  {p}th percentile: {mult:.4f}x")

        return self.percentile_multipliers

    def get_percentile_price(self, days_since_genesis: int, percentile: float) -> float:
        """
        Get Bitcoin price for specific percentile.

        Args:
            days_since_genesis: Days since Bitcoin genesis
            percentile: Percentile (2.5, 16.5, 50.0, 83.5, 97.5)

        Returns:
            Bitcoin price for specified percentile
        """
        if not self.is_fitted or self.percentile_multipliers is None:
            raise ValueError(
                "Must fit percentiles first using fit_percentiles() or fit_percentiles_from_csv()")

        if percentile not in self.percentile_multipliers:
            raise ValueError(
                f"Percentile {percentile} not available. Available: {list(self.percentile_multipliers.keys())}")

        trendline = self.power_law_price(days_since_genesis)
        return trendline * self.percentile_multipliers[percentile]

    def fit_percentiles_rolling(self, historical_data):
        """
        Fit rolling power-law regressions and derive percentile multipliers.
        historical_data: List of (date, price) tuples, month-end sampled.
        """
        # Prepare logs
        days = np.array(
            [(d - self.genesis_date).days for d, _ in historical_data])
        prices = np.array([p for _, p in historical_data])
        x = np.log(days)
        y = np.log(prices)

        ratios = []
        # Rolling regressions
        for i in range(1, len(days)):
            xi = x[: i + 1]
            yi = y[: i + 1]
            Xmat = sm.add_constant(xi)
            model = sm.OLS(yi, Xmat).fit()
            a, b = model.params  # intercept = a, slope = b
            # Trend price at this point
            trend_log = a + b * x[i]
            trend = np.exp(trend_log)
            ratios.append(prices[i] / trend)

        arr = np.array(ratios)
        self.percentile_multipliers = {
            2.5: np.percentile(arr, 2.5),
            16.5: np.percentile(arr, 16.5),
            50.0: np.percentile(arr, 50.0),
            83.5: np.percentile(arr, 83.5),
            97.5: np.percentile(arr, 97.5),
        }
        self.is_fitted = True
        return self.percentile_multipliers


# Global calculator instance - initialize once and reuse
_global_calculator = BitcoinPowerLawCalculator()


@st.cache_data(ttl=None)
def initialize_percentiles_from_csv(csv_file_path: str):
    """
    Initialize the global calculator with percentiles from CSV file.
    Call this once at the start of your application.

    Args:
        csv_file_path: Path to CSV file with Bitcoin price data
    """
    global _global_calculator

    try:
        # load month-end data
        hist = _global_calculator.load_historical_data(csv_file_path)
        print(f"✅ Loaded {len(hist)} points for rolling fit")
        # rolling regression + percentile fitting
        multipliers = _global_calculator.fit_percentiles_rolling(hist)
        print("✅ Rolling percentiles:", multipliers)
    except Exception as e:
        print(f"❌ Rolling fit failed: {e}")
        print("⚠️ Setting up fallback multipliers")

        # Fallback multipliers instead of leaving them as None
        _global_calculator.percentile_multipliers = {
            2.5: 0.24,
            16.5: 0.50,
            50.0: 1.0,
            83.5: 1.75,
            97.5: 2.5
        }
        _global_calculator.is_fitted = True

        print("✅ Fallback multipliers set successfully")
        for p, mult in _global_calculator.percentile_multipliers.items():
            print(f"  {p}th percentile: {mult:.4f}x")


def bitcoin_power_law_price(days_since_genesis: int) -> Tuple[float, float]:
    """
    Calculate Bitcoin price using power law model with empirically derived percentiles.
    Price = 1.42*10^-17 × (days since genesis)^5.79
    Returns the 2.5th percentile (conservative estimate)

    Args:
        days_since_genesis: Number of days since Bitcoin genesis block

    Returns:
        Tuple of (trendline_price, percentile_2_5_price)
    """
    global _global_calculator

    if not _global_calculator.is_fitted or _global_calculator.percentile_multipliers is None:
        raise ValueError(
            "Percentiles not initialized. Call initialize_percentiles_from_csv() first.")

    trendline_price = _global_calculator.power_law_price(days_since_genesis)
    percentile_2_5_price = _global_calculator.get_percentile_price(
        days_since_genesis, 2.5)

    return trendline_price, percentile_2_5_price


def bitcoin_power_law_price_percentile(days_since_genesis: int, percentile: float) -> float:
    """
    Calculate Bitcoin price for any percentile using empirically derived multipliers.

    Args:
        days_since_genesis: Number of days since Bitcoin genesis block
        percentile: Percentile (2.5, 16.5, 50.0, 83.5, 97.5)

    Returns:
        Bitcoin price for specified percentile
    """
    global _global_calculator

    if not _global_calculator.is_fitted or _global_calculator.percentile_multipliers is None:
        raise ValueError(
            "Percentiles not initialized. Call initialize_percentiles_from_csv() first.")

    return _global_calculator.get_percentile_price(days_since_genesis, percentile)


def get_percentile_multipliers() -> Optional[Dict[float, float]]:
    """
    Get the current percentile multipliers.

    Returns:
        Dictionary with percentile multipliers or None if not fitted
    """
    global _global_calculator
    return _global_calculator.percentile_multipliers if _global_calculator.is_fitted else None


def calculate_usd_inr_rate(years_from_current: int, base_rate: float = 83.0,
                           depreciation_rate: float = 0.045) -> float:
    """
    Calculate future USD/INR exchange rate based on depreciation.

    Args:
        years_from_current: Number of years from current date
        base_rate: Current USD/INR rate
        depreciation_rate: Annual depreciation rate (default 4.5%)

    Returns:
        Future USD/INR exchange rate
    """
    return base_rate * (1 + depreciation_rate) ** years_from_current


def get_scenario_parameters(scenario_type: str) -> Dict[str, Any]:
    """
    Returns depreciation and inflation parameters for different scenarios.

    Args:
        scenario_type: 'optimistic', 'conservative', or 'extreme'

    Returns:
        Dictionary containing scenario parameters
    """
    return SCENARIOS.get(scenario_type)


def calculate_single_scenario(current_age: int, retirement_year: int,
                              years_to_retirement: int, years_in_retirement: int,
                              annual_expenditure_inr: float, depreciation_rate: float,
                              inflation_rate: float, genesis_date: datetime) -> Dict[str, Any]:
    """
    Calculate Bitcoin needs for a single retirement scenario using empirically derived percentiles.

    Args:
        current_age: Current age of the person
        retirement_year: Target retirement year
        years_to_retirement: Years until retirement
        years_in_retirement: Expected years in retirement
        annual_expenditure_inr: Current annual expenditure in INR
        depreciation_rate: USD/INR depreciation rate
        inflation_rate: Annual inflation rate
        genesis_date: Bitcoin genesis date

    Returns:
        Dictionary containing scenario results including total Bitcoin needed and breakdown
    """
    # Calculate exchange rate for retirement year
    years_from_current = retirement_year - datetime.now().year
    retirement_usd_inr = calculate_usd_inr_rate(
        years_from_current, depreciation_rate=depreciation_rate)

    # Calculate annual expenditure at retirement
    annual_expenditure_at_retirement_inr = annual_expenditure_inr * \
        (1 + inflation_rate) ** years_to_retirement

    # Calculate Bitcoin needed for each year
    total_bitcoin_needed = 0.0
    total_expense_retirement_inr = 0.0
    breakdown = []

    for year in range(years_in_retirement):
        current_retirement_year = retirement_year + year
        current_date = datetime(current_retirement_year, 1, 1)
        days_since_genesis = (current_date - genesis_date).days

        # Get Bitcoin price for this year (2.5th percentile) - NOW USING EMPIRICAL DATA
        _, btc_price_2_5_usd = bitcoin_power_law_price(days_since_genesis)

        # Calculate exchange rate for this year (continues depreciating)
        current_usd_inr = retirement_usd_inr * (1 + depreciation_rate) ** year

        # Calculate annual expenditure with inflation
        this_year_expense_inr = annual_expenditure_at_retirement_inr * \
            (1 + inflation_rate) ** year

        # Calculate Bitcoin needed
        btc_price_2_5_inr = btc_price_2_5_usd * current_usd_inr
        btc_needed_this_year = this_year_expense_inr / btc_price_2_5_inr

        total_bitcoin_needed += btc_needed_this_year
        total_expense_retirement_inr += this_year_expense_inr

        # Store breakdown
        breakdown.append({
            'Year': current_retirement_year,
            'Age': current_age + years_to_retirement + year,
            'Expense (INR)': this_year_expense_inr,
            'BTC Price (USD)': btc_price_2_5_usd,
            'BTC Price (INR)': btc_price_2_5_inr,
            'BTC Needed': btc_needed_this_year,
            'USD/INR Rate': current_usd_inr
        })

    return {
        'total_bitcoin_needed': total_bitcoin_needed,
        'retirement_usd_inr': retirement_usd_inr,
        'annual_expenditure_at_retirement': annual_expenditure_at_retirement_inr,
        'total_inr_needed': total_expense_retirement_inr,
        'retirement_age': current_age + years_to_retirement,
        'years_in_retirement': years_in_retirement,
        'breakdown': pd.DataFrame(breakdown)
    }


def calculate_retirement_bitcoin_needs_scenarios(current_age: int,
                                                 annual_expenditure_inr: float,
                                                 retirement_year: int) -> Dict[str, Any]:
    """
    Calculate Bitcoin requirements for all three scenarios using empirically derived percentiles.

    Args:
        current_age: Current age of the person
        annual_expenditure_inr: Current annual expenditure in INR
        retirement_year: Target retirement year

    Returns:
        Dictionary containing results for all scenarios or None if invalid timeline
    """
    # Check if percentiles are initialized
    if not _global_calculator.is_fitted or _global_calculator.percentile_multipliers is None:
        raise ValueError(
            "Percentiles not initialized. Call initialize_percentiles_from_csv() first.")

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
    for scenario_name in ['Optimistic', 'Conservative', 'Extreme']:
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
