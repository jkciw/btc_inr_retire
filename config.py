"""
Configuration Module

This module contains configuration constants and settings
for the Bitcoin Retirement Calculator application.
"""

from datetime import datetime
from typing import Dict, Any

# Application Configuration
APP_CONFIG = {
    'title': 'Bitcoin Retirement Calculator',
    'icon': '₿',
    'layout': 'wide',
    'subtitle': 'Powered by the 2.5th percentile price of the Bitcoin power law model'
}

# Bitcoin Configuration
BITCOIN_CONFIG = {
    'genesis_date': datetime(2009, 1, 3),
    'power_law_exponent': 5.8,
    'power_law_coefficient': 10**(-17),
    'percentile_multiplier': 0.24  # 2.5th percentile is 24% of trendline
}

# Economic Scenarios Configuration
SCENARIOS = {
    'Optimistic': {
        'usd_appreciation_rate': 0.050,
        'inflation_rate': 0.06,
        'description': 'Best-case economic conditions',
        'color': '#28a745'  # Green
    },
    'Conservative': {
        'usd_appreciation_rate': 0.03,
        'inflation_rate': 0.080,
        'description': 'Prudent retirement planning baseline',
        'color': '#ffc107'  # Yellow
    },
    'Extreme': {
        'usd_appreciation_rate': 0.020,
        'inflation_rate': 0.100,
        'description': 'Worst-case stress testing',
        'color': '#dc3545'  # Red
    }
}

# API Configuration
API_CONFIG = {
    'timeout': 10,
    'cache_ttl': 600,  # 10 minutes
    'max_retries': 5,
    'fallback_btc_price': 100000.0,
    'fallback_usd_inr': 87.0,
    'apis': [
        {
            'name': 'CoinPaprika',
            'btc_url': 'https://api.coinpaprika.com/v1/tickers/btc-bitcoin',
            'priority': 1
        },
        {
            'name': 'Binance',
            'btc_url': 'https://api.binance.com/api/v3/ticker/price',
            'symbol': 'BTCUSDT',
            'priority': 2
        },
        {
            'name': 'CoinGecko',
            'btc_url': 'https://api.coingecko.com/api/v3/simple/price',
            'params': {'ids': 'bitcoin', 'vs_currencies': 'usd'},
            'priority': 3
        }
    ],
    'usd_inr_url': 'https://cdn.jsdelivr.net/npm/@fawazahmed0/currency-api@latest/v1/currencies/usd.json'
}

# Input Validation Configuration
INPUT_LIMITS = {
    'age': {'min': 18, 'max': 80, 'default': 30},
    'expenditure': {'min': 100000, 'max': 50000000, 'default': 1000000, 'step': 50000},
    'retirement_year': {'min': 2026, 'max': 2070, 'default': 2045},
    'retirement_age_limit': 90
}

# Chart Configuration
CHART_CONFIG = {
    'colors': {
        'optimistic': '#28a745',
        'conservative': '#ffc107',
        'extreme': '#dc3545',
        'btc_price': 'green'
    },
    'line_width': 3,
    'marker_size': 6,
    'hover_mode': 'x unified'
}

# Number Formatting Configuration
FORMAT_CONFIG = {
    'indian_locale': 'en_IN',
    'default_decimals': 2,
    'btc_decimals': 6,
    'currency_decimals': 0
}
