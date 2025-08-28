"""
Market Data API Module

This module handles fetching Bitcoin prices and USD/INR exchange rates
from free APIs with retry logic and fallback mechanisms.
"""

import requests
from requests.exceptions import HTTPError
import streamlit as st
import time
from typing import Tuple


def get_market_price_coinpaprika() -> Tuple[float, float]:
    """
    Fetch Bitcoin price and USD/INR rate from CoinPaprika and currency API.

    Returns:
        Tuple of (btc_price_usd, usd_inr_rate)
    """
    try:
        # Fetch from CoinPaprika
        btc_url = 'https://api.coinpaprika.com/v1/tickers/btc-bitcoin'
        response = requests.get(btc_url, timeout=10)
        response.raise_for_status()
        btc_data = response.json()
        btc_price = btc_data.get('quotes', {}).get('USD', {}).get('price', 0.0)

        # Fetch the latest USD/INR exchange rate
        url = 'https://cdn.jsdelivr.net/npm/@fawazahmed0/currency-api@latest/v1/currencies/usd.json'
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        data = response.json()

        if 'usd' not in data or 'inr' not in data['usd']:
            st.warning("Couldn't fetch USD/INR exchange rate")
            usd_inr = 0.0
        else:
            usd_inr = data['usd']['inr']

        return float(btc_price), float(usd_inr)

    except Exception as e:
        st.error(f"Error fetching market data: {e}")
        return 0.0, 0.0


def get_market_price_binance() -> Tuple[float, float]:
    """
    Fetch Bitcoin price from Binance and USD/INR rate from currency API.

    Returns:
        Tuple of (btc_price_usd, usd_inr_rate)
    """
    try:
        # Binance allows 1200 calls/minute without API key
        btc_url = 'https://api.binance.com/api/v3/ticker/price'
        btc_params = {'symbol': 'BTCUSDT'}
        response = requests.get(btc_url, params=btc_params, timeout=10)
        response.raise_for_status()
        btc_data = response.json()
        btc_price = float(btc_data.get('price', 0.0))

        # Fetch the latest USD/INR exchange rate
        url = 'https://cdn.jsdelivr.net/npm/@fawazahmed0/currency-api@latest/v1/currencies/usd.json'
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        data = response.json()

        if 'usd' not in data or 'inr' not in data['usd']:
            st.warning("Couldn't fetch USD/INR exchange rate")
            usd_inr = 0.0
        else:
            usd_inr = data['usd']['inr']

        return btc_price, usd_inr

    except Exception as e:
        return 0.0, 0.0


def get_market_price_kraken() -> Tuple[float, float]:
    """
    Fetch Bitcoin price from Kraken and USD/INR rate from currency API.
    Returns:
        Tuple of (btc_price_usd, usd_inr_rate)
    """
    try:
        # Kraken public API
        btc_url = 'https://api.kraken.com/0/public/Ticker'
        btc_params = {'pair': 'XBTUSD'}
        response = requests.get(btc_url, params=btc_params, timeout=10)
        response.raise_for_status()
        data = response.json()

        if 'result' in data and 'XXBTZUSD' in data['result']:
            btc_price = float(data['result']['XXBTZUSD']['c'][0])
        else:
            return 0.0, 0.0

        # Fetch the latest USD/INR exchange rate
        url = 'https://cdn.jsdelivr.net/npm/@fawazahmed0/currency-api@latest/v1/currencies/usd.json'
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        data = response.json()

        if 'usd' not in data or 'inr' not in data['usd']:
            usd_inr = 86.0  # Fallback rate
        else:
            usd_inr = data['usd']['inr']

        return btc_price, usd_inr

    except Exception as e:
        return 0.0, 0.0


def get_market_price_coinbase() -> Tuple[float, float]:
    """
    Fetch Bitcoin price from Coinbase and USD/INR rate from currency API.
    Returns:
        Tuple of (btc_price_usd, usd_inr_rate)
    """
    try:
        # Coinbase public API
        btc_url = 'https://api.coinbase.com/v2/exchange-rates'
        btc_params = {'currency': 'BTC'}
        response = requests.get(btc_url, params=btc_params, timeout=10)
        response.raise_for_status()
        data = response.json()

        if 'data' in data and 'rates' in data['data'] and 'USD' in data['data']['rates']:
            btc_price = float(data['data']['rates']['USD'])
        else:
            return 0.0, 0.0

        # Fetch the latest USD/INR exchange rate
        url = 'https://cdn.jsdelivr.net/npm/@fawazahmed0/currency-api@latest/v1/currencies/usd.json'
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        data = response.json()

        if 'usd' not in data or 'inr' not in data['usd']:
            usd_inr = 86.0  # Fallback rate
        else:
            usd_inr = data['usd']['inr']

        return btc_price, usd_inr

    except Exception as e:
        return 0.0, 0.0


def fetch_with_retry(url: str, params=None, max_retries: int = 5):
    """
    Fetch data from URL with exponential backoff retry logic.

    Args:
        url: URL to fetch data from
        params: Request parameters
        max_retries: Maximum number of retry attempts

    Returns:
        JSON response data

    Raises:
        Exception: If all retries fail
    """
    retries = 0
    while retries < max_retries:
        try:
            response = requests.get(url, params=params, timeout=10)
            response.raise_for_status()
            return response.json()
        except HTTPError:
            if response.status_code == 429:
                # Exponential backoff: 1,2,4,8,16 seconds
                wait_time = (2 ** retries) * 1
                st.warning(f"Rate limit hit. Waiting {wait_time} seconds...")
                time.sleep(wait_time)
                retries += 1
                continue
            else:
                raise
        except Exception:
            raise

    raise Exception(
        f"Failed after {max_retries} retries due to rate limiting.")


def get_market_price_with_retry() -> Tuple[float, float]:
    """
    Fetch Bitcoin price from CoinGecko with retry logic.

    Returns:
        Tuple of (btc_price_usd, usd_inr_rate)
    """
    try:
        # Fetch Bitcoin price with retry logic
        btc_url = 'https://api.coingecko.com/api/v3/simple/price'
        btc_params = {'ids': 'bitcoin', 'vs_currencies': 'usd'}
        btc_data = fetch_with_retry(btc_url, btc_params)
        btc_price = btc_data.get('bitcoin', {}).get('usd', 0.0)

        # Add delay between requests
        time.sleep(2)

        # Fetch the latest USD/INR exchange rate
        url = 'https://cdn.jsdelivr.net/npm/@fawazahmed0/currency-api@latest/v1/currencies/usd.json'
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        data = response.json()

        if 'usd' not in data or 'inr' not in data['usd']:
            st.warning("Couldn't fetch USD/INR exchange rate")
            usd_inr = 0.0
        else:
            usd_inr = data['usd']['inr']

        return float(btc_price), float(usd_inr)

    except Exception as e:
        st.error(f"Error fetching market data: {e}")
        return 0.0, 0.0


@st.cache_data(ttl=600)  # Cache for 10 minutes
def get_cached_market_price() -> Tuple[float, float]:
    """
    Get cached market price data to avoid excessive API calls.

    Returns:
        Tuple of (btc_price_usd, usd_inr_rate)
    """
    return get_market_price_with_retry()


def get_market_price() -> Tuple[float, float]:
    """
    Try multiple free APIs in order of preference to get Bitcoin price and USD/INR rate.

    Returns:
        Tuple of (btc_price_usd, usd_inr_rate)
    """
    # API sources in order of preference
    apis = [
        ('CoinPaprika', get_market_price_coinpaprika),
        ('Binance', get_market_price_binance),
        ('Kraken', get_market_price_kraken),
        ('Coinbase', get_market_price_coinbase),
        ('CoinGecko', get_market_price_with_retry)  # CoinGecko as fallback
    ]

    for api_name, api_func in apis:
        try:
            btc_price, usd_inr = api_func()
            if btc_price > 0 and usd_inr > 0:
                # st.success(f"Market data fetched from {api_name}")
                return btc_price, usd_inr
        except Exception as e:
            st.warning(f"{api_name} failed: {str(e)}")
            continue

    # If all APIs fail, use conservative fallbacks
    st.error("All APIs failed. Using fallback values.")
    return 100000.0, 87.0  # Conservative fallback values
