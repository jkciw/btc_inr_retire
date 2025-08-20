"""
Bitcoin Power Law and Financial Calculations Module

This module contains core financial calculations including:
- Bitcoin power law price predictions
- USD/INR exchange rate projections
- Retirement scenario calculations
"""

import numpy as np
import pandas as pd
from datetime import datetime
from typing import Tuple, Dict, Any


def bitcoin_power_law_price(days_since_genesis: int) -> Tuple[float, float]:
    """
    Calculate Bitcoin price using power law model.
    
    Price = 10^-17 × (days since genesis)^5.8
    Returns the 2.5th percentile (conservative estimate) - 24% of trendline
    
    Args:
        days_since_genesis: Number of days since Bitcoin genesis block
        
    Returns:
        Tuple of (trendline_price, percentile_2_5_price)
    """
    trendline_price = 10**(-17) * (days_since_genesis ** 5.8)
    # 2.5th percentile is approximately 24% of trendline price
    percentile_2_5_price = trendline_price * 0.24
    return trendline_price, percentile_2_5_price


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


def calculate_single_scenario(current_age: int, retirement_year: int, 
                            years_to_retirement: int, years_in_retirement: int,
                            annual_expenditure_inr: float, depreciation_rate: float,
                            inflation_rate: float, genesis_date: datetime) -> Dict[str, Any]:
    """
    Calculate Bitcoin needs for a single retirement scenario.
    
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


def calculate_retirement_bitcoin_needs_scenarios(current_age: int, 
                                               annual_expenditure_inr: float, 
                                               retirement_year: int) -> Dict[str, Any]:
    """
    Calculate Bitcoin requirements for all three scenarios.
    
    Args:
        current_age: Current age of the person
        annual_expenditure_inr: Current annual expenditure in INR
        retirement_year: Target retirement year
        
    Returns:
        Dictionary containing results for all scenarios or None if invalid timeline
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