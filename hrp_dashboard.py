import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.cluster.hierarchy import linkage, fcluster, leaves_list # Added linkage
from pypfopt import HRPOpt, risk_models
from pypfopt.plotting import plot_dendrogram
from scipy import stats
from scipy.stats import norm, skew, kurtosis, t, gaussian_kde

# --- Page Configuration ---
st.set_page_config(layout="wide", page_title="HRP Dashboard")

# --- Global Initializations ---
user_tickers_list = []
w_user = pd.Series(dtype=float)
uploaded_file_state = None # To hold the state of the uploaded file across reruns if needed

# --- Helper Functions ---
@st.cache_data
def download_prices(tickers, start_date, end_date=None):
    if not tickers:
        st.warning("download_prices called with no tickers.") # Added warning
        return pd.DataFrame()
    st.info(f"Attempting to download prices for: {tickers} from {start_date} to {end_date}") # Added info
    try:
        prices = yf.download(tickers, start=start_date, end=end_date, progress=False, auto_adjust=True)['Close']
        if isinstance(prices, pd.Series): 
            # Ensure tickers is a list and has at least one element before accessing tickers[0]
            ticker_name = tickers[0] if isinstance(tickers, list) and tickers else "UNKNOWN_TICKER"
            prices = prices.to_frame(name=ticker_name)
        
        if prices.empty:
            st.warning(f"yf.download returned an empty DataFrame for tickers: {tickers} (period: {start_date}-{end_date}).")
            return pd.DataFrame()
        
        if prices.isnull().all().all():
            st.warning(f"yf.download returned DataFrame with all NaN values for tickers: {tickers} (period: {start_date}-{end_date}).")
            return pd.DataFrame()

        prices_cleaned = prices.dropna(axis=1, how='all') 
        
        if prices_cleaned.empty and not prices.empty:
             st.warning(f"All ticker columns were NaN for {tickers} (period: {start_date}-{end_date}) and were dropped. Original columns: {prices.columns.tolist()}")
        
        return prices_cleaned
    except Exception as e:
        st.error(f"Exception during yf.download for tickers {tickers} (period: {start_date}-{end_date}): {e}") # Changed to st.error
        return pd.DataFrame()

@st.cache_data # Not caching this as cov_matrix can change
def diversification_ratio(w, cov_matrix):
    if cov_matrix is None or w is None or cov_matrix.shape[0] == 0 or len(w) == 0:
        return 0.0
    w_vol = np.sqrt(w.T @ cov_matrix @ w)
    if w_vol == 0: return 0.0
    ind_vol = np.sqrt(np.diag(cov_matrix))
    return (w @ ind_vol) / w_vol if w_vol else 0.0

@st.cache_data # Not caching this
def risk_contributions_calc(w, cov_matrix):
    if cov_matrix is None or w is None or cov_matrix.shape[0] == 0 or len(w) == 0:
        return np.array([])
    if not isinstance(w, np.ndarray): w = np.array(w)
    if not isinstance(cov_matrix, np.ndarray): cov_matrix = np.array(cov_matrix)
    
    # Ensure w is 1D
    if w.ndim > 1: w = w.flatten()

    portfolio_var = w.T @ cov_matrix @ w
    if portfolio_var == 0: return np.zeros_like(w)
    
    # Marginal contributions: (Sigma * w)
    mrc = cov_matrix @ w
    # Risk contributions: MRC_i * w_i / portfolio_sigma
    rc = (mrc * w) / np.sqrt(portfolio_var) # This is one definition
    # More common: RC_i = w_i * (Cov_matrix * w)_i / PortfolioVariance. Then normalize to sum to 100%.
    rc_abs = w * (cov_matrix @ w)
    return (rc_abs / portfolio_var) * 100 if portfolio_var else np.zeros_like(w)

@st.cache_data
def enb_mlt(weights, cov_matrix):
    """
    Effective Number of Bets via Minimal Linear Torsion (Meucci, 2009).

    Steps
    -----
    1.  Σ = V Λ Vᵀ             (eigen-decomposition of the asset covariance)
    2.  Σ½ = V √Λ Vᵀ           (matrix square-root)
    3.  g  = Σ½ · w            (factor exposures of the portfolio)
    4.  vᵢ = gᵢ²               (variance each uncorrelated factor contributes)
    5.  pᵢ = vᵢ / Σ vᵢ         (normalised contributions, Σ pᵢ = 1)
    6.  ENB = exp( −Σ pᵢ ln pᵢ )   (Shannon-entropy effective dimensionality)
    """
    if cov_matrix is None or weights is None:                       # basic guards
        return 0.0
    w = np.asarray(weights, dtype=float).flatten()
    Σ = np.asarray(cov_matrix, dtype=float)
    if w.size == 0 or Σ.size == 0 or np.allclose(w, 0):
        return 0.0

    #  Eigen-decomposition (guaranteed SPD after Ledoit-Wolf shrinkage above)
    λ, V = np.linalg.eigh(Σ)
    λ = np.clip(λ, 1e-12, None)                                     # numeric guard
    Σ_half = (V * np.sqrt(λ)) @ V.T                                 # Σ^{1/2}

    g = Σ_half @ w                                                  # factor loads
    v = g ** 2                                                      # factor variances
    if v.sum() == 0:
        return 0.0
    p = v / v.sum()                                                 # normalise
    return np.exp(-np.sum(p * np.log(p)))                           # Shannon ENB


@st.cache_data # Not caching this
def enb(rc_vector):
    if rc_vector is None or len(rc_vector) == 0 or np.sum(rc_vector**2) == 0:
        return 0.0
    # Ensure rc_vector sums to 1 (or 100 for percentage) for correct ENB calculation
    rc_normalized = np.array(rc_vector) / np.sum(rc_vector) if np.sum(rc_vector) != 0 else np.array(rc_vector)
    return 1.0 / np.sum(rc_normalized**2) if np.sum(rc_normalized**2) != 0 else 0.0

@st.cache_data # Not caching this
def rc_table_calc(w_series, cov_matrix_df, name):
    if w_series.empty or cov_matrix_df.empty:
        return pd.DataFrame(columns=[name, f'{name} RC (%)'])
    
    common_assets = w_series.index.intersection(cov_matrix_df.index)
    if common_assets.empty:
        return pd.DataFrame(columns=[name, f'{name} RC (%)'])
        
    w_aligned = w_series.reindex(common_assets).fillna(0).values
    cov_aligned = cov_matrix_df.reindex(index=common_assets, columns=common_assets).fillna(0).values
    
    rc_values = risk_contributions_calc(w_aligned, cov_aligned)
    rc_series = pd.Series(rc_values, index=common_assets, name=f'{name} RC (%)')
    
    df = pd.concat([w_series.reindex(common_assets).fillna(0).rename(name), rc_series], axis=1)
    return df.sort_values(by=f'{name} RC (%)', ascending=False)

@st.cache_data
def calculate_portfolio_performance(returns, weights, name="Portfolio"):
    """Calculate portfolio performance statistics"""
    if returns.empty or weights.empty:
        return {}
    
    # Align returns and weights
    common_assets = returns.columns.intersection(weights.index)
    if common_assets.empty:
        return {}
    
    returns_aligned = returns[common_assets]
    weights_aligned = weights.reindex(common_assets).fillna(0)
    
    # Calculate portfolio returns
    portfolio_returns = (returns_aligned * weights_aligned).sum(axis=1)
    
    # Calculate cumulative returns
    cumulative_returns = (1 + portfolio_returns).cumprod()
    
    # Calculate statistics
    total_return = cumulative_returns.iloc[-1] - 1
    annual_return = (1 + total_return) ** (252 / len(portfolio_returns)) - 1
    annual_volatility = portfolio_returns.std() * np.sqrt(252)
    sharpe_ratio = annual_return / annual_volatility if annual_volatility > 0 else 0
    
    # Calculate max drawdown
    rolling_max = cumulative_returns.expanding().max()
    drawdown = (cumulative_returns - rolling_max) / rolling_max
    max_drawdown = drawdown.min()
    
    # Calculate Calmar ratio
    calmar_ratio = annual_return / abs(max_drawdown) if max_drawdown < 0 else 0
    
    # Calculate VaR and CVaR at 99% confidence level
    var_99 = np.percentile(portfolio_returns, 1)  # 1st percentile for 99% VaR
    cvar_99 = portfolio_returns[portfolio_returns <= var_99].mean()  # Expected shortfall
    
    # Calculate skewness and kurtosis
    portfolio_skew = skew(portfolio_returns.dropna())
    portfolio_kurt = kurtosis(portfolio_returns.dropna(), fisher=True)  # Excess kurtosis
    
    return {
        'portfolio_returns': portfolio_returns,
        'cumulative_returns': cumulative_returns,
        'total_return': total_return,
        'annual_return': annual_return,
        'annual_volatility': annual_volatility,
        'sharpe_ratio': sharpe_ratio,
        'max_drawdown': max_drawdown,
        'calmar_ratio': calmar_ratio,
        'var_99': var_99,
        'cvar_99': cvar_99,
        'skewness': portfolio_skew,
        'kurtosis': portfolio_kurt
    }

# --- Sidebar for Inputs ---
st.sidebar.header("User Portfolio & Settings")

# Date Range
st.sidebar.subheader("Date Range for Price Data")
default_start_date = pd.to_datetime("2022-01-01")
default_end_date = pd.to_datetime("today")
start_date = st.sidebar.date_input("Start Date", value=default_start_date, max_value=default_end_date - pd.Timedelta(days=1))
end_date = st.sidebar.date_input("End Date", value=default_end_date, min_value=start_date + pd.Timedelta(days=1))

# Portfolio Input Method
st.sidebar.subheader("Portfolio Input")
input_method = st.sidebar.radio("Choose input method:", ("Manual", "Upload CSV"), key="input_method_radio")

if input_method == "Manual":
    tickers_str = st.sidebar.text_area("Tickers (comma-separated):", "AAPL,MSFT,GOOG,AMZN,NVDA", key="manual_tickers")
    weights_str = st.sidebar.text_area("Weights (comma-separated):", "0.2,0.2,0.2,0.2,0.2", key="manual_weights")
    if tickers_str:
        user_tickers_list = [ticker.strip().upper() for ticker in tickers_str.split(",") if ticker.strip()]
        if weights_str:
            try:
                user_weights_list_values = [float(w.strip()) for w in weights_str.split(",") if w.strip()]
                if len(user_tickers_list) == len(user_weights_list_values):
                    w_user = pd.Series(user_weights_list_values, index=user_tickers_list, name="User Weights")
                # Else: Mismatch, will be handled by validation in button click
            except ValueError:
                pass # Invalid format, will be handled by validation
        else: # No weights string, implies equal weight or error if tickers exist
            w_user = pd.Series(dtype=float) # Clear to trigger equal weighting later if tickers exist
    else: # No tickers string
        user_tickers_list = []
        w_user = pd.Series(dtype=float)

elif input_method == "Upload CSV":
    uploaded_file = st.sidebar.file_uploader("Upload CSV (Columns: Ticker, [Weight])", type="csv", key="csv_uploader") # Changed help text
    if uploaded_file is not None:
        uploaded_file_state = uploaded_file # Store it
        try:
            df_upload = pd.read_csv(uploaded_file)
            if 'Ticker' in df_upload.columns: # Changed from 'Tickers'
                # Get all non-empty, stripped, and uppercased tickers from the 'Ticker' column
                current_csv_tickers = [str(t).strip().upper() for t in df_upload['Ticker'].tolist() if str(t).strip()] # Changed from 'Tickers'

                if not current_csv_tickers:
                    st.sidebar.warning("The 'Ticker' column in the CSV is present but contains no valid ticker symbols.")
                    user_tickers_list = [] # Update global
                    w_user = pd.Series(dtype=float) # Update global
                else:
                    user_tickers_list = current_csv_tickers # Update global: these are all non-empty tickers from CSV

                    if 'Weight' in df_upload.columns: # Changed from 'Weights'
                        # Weights column exists. Attempt to parse weights.
                        # Create a dictionary to store valid weights for tickers found in current_csv_tickers.
                        ticker_to_weight_map = {}
                        for _, row in df_upload.iterrows():
                            ticker_in_row = str(row['Ticker']).strip().upper() # Changed from 'Tickers'
                            # Only consider weights for tickers that are in our valid current_csv_tickers list
                            if ticker_in_row in user_tickers_list:
                                weight_val = pd.to_numeric(row.get('Weight'), errors='coerce') # Changed from 'Weights'
                                if not pd.isna(weight_val):
                                    # If a ticker is duplicated in CSV, this takes the last valid weight.
                                    # Or, if you want the first, check: if ticker_in_row not in ticker_to_weight_map:
                                    ticker_to_weight_map[ticker_in_row] = weight_val
                        
                        if ticker_to_weight_map:
                            # Create w_user Series from the map, ensuring it only contains tickers from user_tickers_list
                            # The order will be dictated by dict insertion order (Python 3.7+) or Series creation from dict.
                            # Reindexing by user_tickers_list can ensure specific order if needed, but map keys are already from it.
                            w_user = pd.Series(ticker_to_weight_map, name="User Weights")
                            # It's possible that some tickers in user_tickers_list do not have valid weights in the map.
                            # w_user will only contain those with valid weights. This is handled by downstream logic.
                        else:
                            # Weights column existed, but no valid numeric weights found for any of the parsed tickers.
                            w_user = pd.Series(dtype=float) # Update global to empty Series
                    else:
                        # No 'Weight' column in df_upload, so w_user is empty.
                        w_user = pd.Series(dtype=float) # Update global
            else:
                # 'Ticker' column missing in df_upload
                st.sidebar.error("CSV file must contain a 'Ticker' column.") # Changed from 'Tickers'
                user_tickers_list = [] # Update global
                w_user = pd.Series(dtype=float) # Update global
        except Exception as e:
            st.sidebar.error(f"Error processing CSV: {e}")
            user_tickers_list = [] # Update global on error
            w_user = pd.Series(dtype=float) # Update global on error
    elif uploaded_file_state is not None and uploaded_file is None: # File was removed
        user_tickers_list = []
        w_user = pd.Series(dtype=float)
        uploaded_file_state = None


# Analysis Parameters
st.sidebar.subheader("Parameters")
top_n_rc = st.sidebar.number_input("Top N Risk Contributors", min_value=1, max_value=100, value=50, step=1, key="top_n_rc_input")
max_clusters_rc_detail_input = st.sidebar.number_input("Max Clusters", min_value=2, max_value=100, value=30, step=1, key="max_clusters_input")

# --- Main Application ---
st.title("Risk Parity Dashboard")

if st.sidebar.button("Run"):
    # --- Input Validation and Preparation ---
    if not user_tickers_list:
        st.error("No tickers provided. Please enter tickers manually or upload a CSV file with a 'Ticker' column.") # Corrected 'Tickers' to 'Ticker'
        st.stop()

    # Consolidate w_user based on inputs
    if input_method == "Manual":
        if tickers_str: # Recalculate from sidebar state to be sure
            current_manual_tickers = [ticker.strip().upper() for ticker in tickers_str.split(",") if ticker.strip()]
            if not current_manual_tickers: # Should be caught by user_tickers_list check above
                 st.error("Manual tickers input is empty.")
                 st.stop()
            user_tickers_list = current_manual_tickers # Ensure user_tickers_list is up-to-date

            if weights_str:
                try:
                    current_manual_weights_val = [float(w.strip()) for w in weights_str.split(",") if w.strip()]
                    if len(user_tickers_list) == len(current_manual_weights_val):
                        w_user = pd.Series(current_manual_weights_val, index=user_tickers_list, name="User Weights")
                        if abs(w_user.sum() - 1.0) > 1e-6:
                            st.warning(f"Manual weights sum to {w_user.sum():.4f}. Renormalizing.")
                            if w_user.sum() == 0:
                                st.error("Manual weights sum to 0. Cannot renormalize. Applying equal weights.")
                                w_user = pd.Series(1/len(user_tickers_list), index=user_tickers_list, name="User Weights")
                            else:
                                w_user = w_user / w_user.sum()
                    else:
                        st.error(f"Mismatch between number of manual tickers ({len(user_tickers_list)}) and weights ({len(current_manual_weights_val)}). Applying equal weights.")
                        w_user = pd.Series(1/len(user_tickers_list), index=user_tickers_list, name="User Weights")
                except ValueError:
                    st.error("Invalid format for manual weights. Applying equal weights.")
                    w_user = pd.Series(1/len(user_tickers_list), index=user_tickers_list, name="User Weights")
            else: # No manual weights string
                st.warning("No manual weights provided. Applying equal weights.")
                w_user = pd.Series(1/len(user_tickers_list), index=user_tickers_list, name="User Weights")
        # else: No manual tickers string, already caught by initial user_tickers_list check

    elif input_method == "Upload CSV":
        if not user_tickers_list: 
            st.error("Could not extract tickers from CSV. Ensure CSV has a 'Ticker' column and it contains valid ticker symbols.") # Corrected and expanded message
            st.stop()
        if w_user.empty and user_tickers_list: # No 'Weights' column or all weights NaN
            st.warning("No valid 'Weights' column in CSV or all weights are invalid. Applying equal weights.")
            w_user = pd.Series(1/len(user_tickers_list), index=user_tickers_list, name="User Weights")
        elif not w_user.empty: # Weights column was present and parsed
            if abs(w_user.sum() - 1.0) > 1e-6:
                st.warning(f"CSV weights sum to {w_user.sum():.4f}. Renormalizing.")
                if w_user.sum() == 0:
                    st.error("CSV weights sum to 0. Cannot renormalize. Applying equal weights.")
                    w_user = pd.Series(1/len(user_tickers_list), index=user_tickers_list, name="User Weights")
                else:
                    w_user = w_user / w_user.sum()
    
    # Final check for w_user after all input processing
    if w_user.empty and user_tickers_list: # Should be redundant if logic above is correct
        st.warning("User weights could not be determined. Applying equal weights as a fallback.")
        w_user = pd.Series(1/len(user_tickers_list), index=user_tickers_list, name="User Weights")
    elif w_user.empty and not user_tickers_list: # Should be caught
        st.error("Critical error: No tickers or weights to process.")
        st.stop()


    st.info(f"Preparing to analyze with tickers: {user_tickers_list}, Start: {start_date}, End: {end_date}") # Debug info

    with st.spinner("Fetching data"):
        try:
            prices = download_prices(user_tickers_list, start_date, end_date)
            if prices.empty or prices.shape[0] < 2:
                st.error("Could not retrieve enough price data for the selected tickers and date range. Please check tickers and dates. Review messages above for download details.") # Augmented message
                st.stop()
            
            returns = prices.pct_change().dropna(how='all') # Drop rows that are all NaN
            if returns.empty or returns.shape[0] < 2 : # Need at least 2 periods for cov
                st.error("Could not calculate returns or insufficient data after dropping NaNs. Price data might be constant or too short.")
                st.stop()
            
            valid_tickers_from_prices = returns.dropna(axis=1, how='all').columns.tolist()
            if not valid_tickers_from_prices:
                st.error("No tickers with valid price data found for the selected period.")
                st.stop()

            returns = returns[valid_tickers_from_prices]
            prices = prices[valid_tickers_from_prices] # Keep prices aligned

            # Align w_user with valid_tickers_from_prices
            w_user_original_sum = w_user.sum()
            w_user = w_user.reindex(valid_tickers_from_prices).fillna(0)
            if w_user.sum() == 0 and w_user_original_sum != 0 and valid_tickers_from_prices:
                st.warning("User portfolio had non-zero weights, but none for tickers with valid price data. Reverting to equal weight for valid tickers.")
                w_user = pd.Series(1/len(valid_tickers_from_prices), index=valid_tickers_from_prices)
            elif abs(w_user.sum() - 1.0) > 1e-6 and w_user.sum() != 0 : # Renormalize if needed after reindex
                w_user = w_user / w_user.sum()
            elif w_user.sum() == 0 and not valid_tickers_from_prices: # Should not happen due to earlier check
                st.error("No valid tickers to form a user portfolio.")
                st.stop()
            elif w_user.sum() == 0 and valid_tickers_from_prices: # All user weights zeroed out for valid tickers
                 st.warning("All user-specified weights are for tickers with no valid price data or are zero. Applying equal weight to available tickers.")
                 w_user = pd.Series(1/len(valid_tickers_from_prices), index=valid_tickers_from_prices)

            w_user.name = "User Weights"


            # 1. Covariance Matrix
            if returns.shape[0] < returns.shape[1]: # More assets than observations
                st.warning(f"Number of observations ({returns.shape[0]}) is less than the number of assets ({returns.shape[1]}). Covariance matrix might be unstable. Using Ledoit-Wolf shrinkage.")
                Sigma = risk_models.CovarianceShrinkage(prices, returns_data=True, log_returns=False).ledoit_wolf() # prices here is actually returns
            else:
                Sigma = returns.cov() # Sample covariance
            
            Sigma = pd.DataFrame(Sigma, index=valid_tickers_from_prices, columns=valid_tickers_from_prices)
            Sigma_df_aligned = Sigma.reindex(index=valid_tickers_from_prices, columns=valid_tickers_from_prices).fillna(0)
            


            # 2. HRP Portfolio
            hrp = HRPOpt(returns, Sigma)
            hrp.optimize() # This sets hrp.clusters
            w_hrp_dict = hrp.clean_weights()
            # Get HRP-ordered tickers from the keys of the cleaned weights dictionary
            ordered_tickers_hrp = list(w_hrp_dict.keys()) 
            w_hrp = pd.Series(w_hrp_dict, name="HRP Weights").reindex(valid_tickers_from_prices).fillna(0)


            st.header("Portfolio Weights Comparison")
            # 4. Dendrogram & Weight Deviations
            st.subheader("HRP Clustering & Weight Deviations")
            
            col1_den, col2_den = st.columns(2)
            with col1_den:
                fig_den, ax_den = plt.subplots(figsize=(10, 6))
                # The HRPOpt object 'hrp' should be passed directly to pypfopt.plotting.plot_dendrogram.
                # hrp.optimize() has already been called, so hrp.clusters should be populated.
                plot_dendrogram(hrp, ax=ax_den, show_tickers=True)
                ax_den.set_title("HRP Dendrogram")
                st.pyplot(fig_den)

            # Create a DataFrame for comparing HRP and User weights
            weights_comparison_df = pd.concat([w_hrp, w_user], axis=1).fillna(0)
            weights_comparison_df.columns = ["HRP Weights", "User Weights"] # Ensure column names
            weights_comparison_df["Difference (User - HRP)"] = weights_comparison_df["User Weights"] - weights_comparison_df["HRP Weights"]
            
            with col2_den:
                st.write("HRP vs User Weights:")
                st.dataframe(weights_comparison_df.style.format("{:.2%}"))
            
            # Bar chart for weight deviations
            # Order by descending User Weights
            weights_for_plot = weights_comparison_df.sort_values(by="HRP Weights", ascending=False)

            fig_dev, ax_dev = plt.subplots(figsize=(12, 7))
            weights_for_plot[["HRP Weights", "User Weights"]].plot(kind='bar', ax=ax_dev, width=0.8)
            ax_dev.set_title("HRP vs User Weights by Asset")
            ax_dev.set_ylabel("Weight")
            ax_dev.set_xlabel("Assets")
            plt.xticks(rotation=90)
            plt.tight_layout()
            st.pyplot(fig_dev)
            
            # Correlation Matrix ordered by dendrogram
            st.subheader("Asset Correlation Matrix")
            

            # Compute correlation matrix from covariance matrix
            std_dev = np.sqrt(np.diag(Sigma_df_aligned.values))
            corr_matrix = Sigma_df_aligned.values / std_dev[:, None] / std_dev[None, :]
            corr_df = pd.DataFrame(corr_matrix, index=valid_tickers_from_prices, columns=valid_tickers_from_prices)
            
            # Reorder by dendrogram (HRP ordering)
            corr_df_ordered = corr_df.reindex(index=ordered_tickers_hrp, columns=ordered_tickers_hrp)
            
            # Calculate figure size based on number of assets (minimum 8x8, scale with asset count)
            n_assets = len(ordered_tickers_hrp)
            fig_size = max(8, min(16, n_assets * 0.4))
            
            fig_corr, ax_corr = plt.subplots(figsize=(fig_size, fig_size))
            
            # Create heatmap
            im = ax_corr.imshow(corr_df_ordered.values, cmap='RdBu_r', vmin=-1, vmax=1, aspect='equal')
            
            # Set ticks and labels
            ax_corr.set_xticks(range(n_assets))
            ax_corr.set_yticks(range(n_assets))
            
            # Calculate font size based on number of assets
            if n_assets <= 10:
                font_size = 10
            elif n_assets <= 20:
                font_size = 8
            elif n_assets <= 30:
                font_size = 8
            else:
                font_size = max(4, 120 // n_assets)  # Minimum font size of 4
            
            ax_corr.set_xticklabels(ordered_tickers_hrp, rotation=90, fontsize=font_size)
            ax_corr.set_yticklabels(ordered_tickers_hrp, fontsize=font_size)
            
            # Add colorbar
            cbar = plt.colorbar(im, ax=ax_corr, shrink=0.8)
            cbar.set_label('Correlation', rotation=270, labelpad=15)
            
            # Add correlation values as text (only for smaller matrices to avoid clutter)
            if n_assets <= 15:
                for i in range(n_assets):
                    for j in range(n_assets):
                        text_color = 'white' if abs(corr_df_ordered.iloc[i, j]) > 0.5 else 'black'
                        ax_corr.text(j, i, f'{corr_df_ordered.iloc[i, j]:.2f}', 
                                   ha='center', va='center', color=text_color, fontsize=max(6, font_size-2))
            
            ax_corr.set_title(f'Correlation Matrix (Dendrogram Ordered)\n{n_assets} Assets', fontsize=min(14, font_size + 4))
            plt.tight_layout()
            st.pyplot(fig_corr)

            st.header("Risk & Diversification")
            # 7. Diversification-loss Metrics
            st.subheader("Portfolio Diversification Metrics")
            
            w_hrp_vals = w_hrp.reindex(valid_tickers_from_prices).fillna(0).values
            w_user_vals = w_user.reindex(valid_tickers_from_prices).fillna(0).values
            Sigma_m_vals = Sigma_df_aligned.values # Already aligned

            DR_HRP = diversification_ratio(w_hrp_vals, Sigma_m_vals)
            DR_usr = diversification_ratio(w_user_vals, Sigma_m_vals)
            
            RC_HRP_vals = risk_contributions_calc(w_hrp_vals, Sigma_m_vals)
            RC_usr_vals = risk_contributions_calc(w_user_vals, Sigma_m_vals)
            
            ENB_HRP = enb_mlt(w_hrp_vals, Sigma_m_vals)
            ENB_usr = enb_mlt(w_user_vals, Sigma_m_vals)

            metrics_data = {
                "Metric": ["Diversification Ratio", "Effective # of Bets"],
                "HRP": [f"{DR_HRP:.3f}", f"{ENB_HRP:.2f}"],
                "User": [f"{DR_usr:.3f}", f"{ENB_usr:.2f}"],
            }
            try:
                metrics_data["Loss/Δ"] = [
                    f"{(DR_HRP-DR_usr)/DR_HRP if DR_HRP else 0:.2%}",
                    f"{(ENB_HRP-ENB_usr)/ENB_HRP if ENB_HRP else 0:.2%}"
                ]
            except ZeroDivisionError:
                 metrics_data["Loss/Δ"] = ["N/A", "N/A"]

            metrics_df = pd.DataFrame(metrics_data).set_index("Metric")
            st.table(metrics_df)

            # 8. Top Asset Risk Contributors
            st.subheader(f"Top {top_n_rc} Asset Risk Contributors")
            rc_hrp_series = pd.Series(RC_HRP_vals, index=valid_tickers_from_prices, name="HRP RC (%)")
            rc_user_series = pd.Series(RC_usr_vals, index=valid_tickers_from_prices, name="User RC (%)")

            rc_combined = pd.concat([rc_hrp_series, rc_user_series], axis=1).fillna(0)
            # Sort by absolute HRP RC to see largest contributors regardless of sign if RC could be negative (not typical for % total variance)
            rc_combined_sorted = rc_combined.reindex(rc_combined["HRP RC (%)"].abs().sort_values(ascending=False).index)


            fig_top_rc, ax_top_rc = plt.subplots(figsize=(12, 7))
            rc_combined_sorted.head(top_n_rc).plot(kind='bar', ax=ax_top_rc)
            ax_top_rc.set_title(f"Top {top_n_rc} Asset Risk Contributions (HRP vs User)")
            ax_top_rc.set_ylabel("Risk Contribution (%)")
            ax_top_rc.set_xlabel("Assets")
            plt.xticks(rotation=45, ha="right")
            plt.tight_layout()
            st.pyplot(fig_top_rc)

            # 10. Detailed Cluster Analysis
            st.subheader(f"Cluster Analysis ({max_clusters_rc_detail_input} clusters)")
            if hrp.clusters is not None: # hrp.clusters is the linkage matrix
                cl_labels_detail = fcluster(hrp.clusters, t=max_clusters_rc_detail_input, criterion="maxclust")
                # Use ordered_tickers_hrp which is derived from clean_weights()
                cl_ser_detail = pd.Series(cl_labels_detail, index=ordered_tickers_hrp, name="Cluster")

                st.write("Tickers per Cluster:")
                # Align weights to ordered_tickers_hrp for this section
                w_hrp_aligned_detail = w_hrp.reindex(ordered_tickers_hrp).fillna(0)
                w_user_aligned_detail = w_user.reindex(ordered_tickers_hrp).fillna(0)

                # Correctly group tickers by cluster ID
                cluster_groups_detail = (
                    pd.DataFrame({'ticker': ordered_tickers_hrp, 'cluster_id': cl_labels_detail})
                    .groupby('cluster_id')['ticker']
                    .apply(list)
                    .sort_index()
                )
                
                rows_detail = []
                for cl_id, names_in_cluster in cluster_groups_detail.items():
                    # names_in_cluster is a list of tickers in dendrogram order for this cluster
                    cluster_w_hrp = w_hrp_aligned_detail.loc[names_in_cluster].sum()
                    cluster_w_user = w_user_aligned_detail.loc[names_in_cluster].sum()
                    
                    # Calculate RC for the cluster (sum of RCs of assets in cluster)
                    # Note: RC_HRP_vals and RC_usr_vals are numpy arrays. Need to map them back to tickers first.
                    rc_hrp_series_detail = pd.Series(RC_HRP_vals, index=valid_tickers_from_prices)
                    rc_user_series_detail = pd.Series(RC_usr_vals, index=valid_tickers_from_prices)

                    # Align these series with ordered_tickers_list before summing over names_in_cluster
                    cluster_rc_hrp = rc_hrp_series_detail.reindex(names_in_cluster).sum()
                    cluster_rc_user = rc_user_series_detail.reindex(names_in_cluster).sum()

                    rows_detail.append({
                        "Tickers": ", ".join(names_in_cluster),
                        "# Tickers": len(names_in_cluster),
                        "HRP Weight": cluster_w_hrp,
                        "User Weight": cluster_w_user,
                        "HRP RC (%)": cluster_rc_hrp,
                        "User RC (%)": cluster_rc_user
                    })
                
                df_rows_detail = pd.DataFrame(rows_detail)
                if not df_rows_detail.empty:
                    st.dataframe(df_rows_detail.style.format({
                        "HRP Weight": "{:.2%}", "User Weight": "{:.2%}",
                        "HRP RC (%)": "{:.2f}%", "User RC (%)": "{:.2f}%"
                    }))
                else:
                    st.write("No cluster details to display.")
            else:
                st.warning("HRP clustering data (hrp.clusters) not available for analysis.")

            # 11. Cumulative Returns Analysis
            st.header("Portfolio Performance")
            st.subheader("Cumulative Returns")
            
            # Calculate performance for both portfolios
            hrp_performance = calculate_portfolio_performance(returns, w_hrp, "HRP Portfolio")
            user_performance = calculate_portfolio_performance(returns, w_user, "User Portfolio")
            
            if hrp_performance and user_performance:
                # Plot cumulative returns
                fig_cumret, ax_cumret = plt.subplots(figsize=(12, 8))
                
                # Get the date index from returns
                dates = returns.index
                
                hrp_cumret = hrp_performance['cumulative_returns']
                user_cumret = user_performance['cumulative_returns']
                
                ax_cumret.plot(dates, hrp_cumret, label='HRP Portfolio', linewidth=2, color='blue')
                ax_cumret.plot(dates, user_cumret, label='User Portfolio', linewidth=2, color='red')
                
                ax_cumret.set_title('Cumulative Returns Comparison', fontsize=14, fontweight='bold')
                ax_cumret.set_xlabel('Date')
                ax_cumret.set_ylabel('Cumulative Return')
                ax_cumret.legend()
                ax_cumret.grid(True, alpha=0.3)
                
                # Format y-axis as percentage
                ax_cumret.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: '{:.0%}'.format(y-1)))
                
                plt.xticks(rotation=45)
                plt.tight_layout()
                st.pyplot(fig_cumret)
                
                # Performance statistics table
                st.subheader("Performance Statistics")
                
                stats_data = {
                    "Metric": [
                        "Total Return",
                        "Annualized Return", 
                        "Annualized Volatility",
                        "Sharpe Ratio",
                        "Maximum Drawdown",
                        "Calmar Ratio",
                        "99% VaR (Daily)",
                        "99% CVaR (Daily)",
                        "Skewness",
                        "Excess Kurtosis"
                    ],
                    "HRP Portfolio": [
                        f"{hrp_performance['total_return']:.2%}",
                        f"{hrp_performance['annual_return']:.2%}",
                        f"{hrp_performance['annual_volatility']:.2%}",
                        f"{hrp_performance['sharpe_ratio']:.3f}",
                        f"{hrp_performance['max_drawdown']:.2%}",
                        f"{hrp_performance['calmar_ratio']:.3f}",
                        f"{hrp_performance['var_99']:.2%}",
                        f"{hrp_performance['cvar_99']:.2%}",
                        f"{hrp_performance['skewness']:.3f}",
                        f"{hrp_performance['kurtosis']:.3f}"
                    ],
                    "User Portfolio": [
                        f"{user_performance['total_return']:.2%}",
                        f"{user_performance['annual_return']:.2%}",
                        f"{user_performance['annual_volatility']:.2%}",
                        f"{user_performance['sharpe_ratio']:.3f}",
                        f"{user_performance['max_drawdown']:.2%}",
                        f"{user_performance['calmar_ratio']:.3f}",
                        f"{user_performance['var_99']:.2%}",
                        f"{user_performance['cvar_99']:.2%}",
                        f"{user_performance['skewness']:.3f}",
                        f"{user_performance['kurtosis']:.3f}"
                    ]
                }
                
                # Calculate differences
                try:
                    stats_data["Difference (User - HRP)"] = [
                        f"{user_performance['total_return'] - hrp_performance['total_return']:.2%}",
                        f"{user_performance['annual_return'] - hrp_performance['annual_return']:.2%}",
                        f"{user_performance['annual_volatility'] - hrp_performance['annual_volatility']:.2%}",
                        f"{user_performance['sharpe_ratio'] - hrp_performance['sharpe_ratio']:.3f}",
                        f"{user_performance['max_drawdown'] - hrp_performance['max_drawdown']:.2%}",
                        f"{user_performance['calmar_ratio'] - hrp_performance['calmar_ratio']:.3f}",
                        f"{user_performance['var_99'] - hrp_performance['var_99']:.2%}",
                        f"{user_performance['cvar_99'] - hrp_performance['cvar_99']:.2%}",
                        f"{user_performance['skewness'] - hrp_performance['skewness']:.3f}",
                        f"{user_performance['kurtosis'] - hrp_performance['kurtosis']:.3f}"
                    ]
                except:
                    stats_data["Difference (User - HRP)"] = ["N/A"] * 10
                
                stats_df = pd.DataFrame(stats_data).set_index("Metric")
                st.table(stats_df)
                
                # Returns distribution analysis
                st.subheader("Returns Distribution")
                
                col1_hist, col2_hist = st.columns(2)
                
                with col1_hist:
                    # HRP Portfolio histogram with fitted distributions
                    fig_hrp_hist, ax_hrp_hist = plt.subplots(figsize=(10, 6))
                    
                    hrp_returns = hrp_performance['portfolio_returns'].dropna()
                    
                    # Plot histogram
                    n_bins = min(50, max(10, len(hrp_returns) // 10))
                    ax_hrp_hist.hist(hrp_returns, bins=n_bins, density=True, alpha=0.7, 
                                   color='blue', edgecolor='black', linewidth=0.5)
                    
                    # Fit distributions
                    x_hrp = np.linspace(hrp_returns.min(), hrp_returns.max(), 100)
                    
                    # Kernel Density Estimation (non-parametric)
                    try:
                        kde_hrp = gaussian_kde(hrp_returns)
                        kde_pdf_hrp = kde_hrp(x_hrp)
                        ax_hrp_hist.plot(x_hrp, kde_pdf_hrp, 'g-', linewidth=2.5, 
                                       label='KDE (Kernel Density)')
                    except:
                        pass

                    # Add VaR line
                    ax_hrp_hist.axvline(hrp_performance['var_99'], color='red', linestyle='--', 
                                      label=f"99% VaR: {hrp_performance['var_99']:.2%}")
                    
                    ax_hrp_hist.set_title('HRP Portfolio Returns Distribution', fontweight='bold')
                    ax_hrp_hist.set_xlabel('Daily Returns')
                    ax_hrp_hist.set_ylabel('Density')
                    ax_hrp_hist.legend()
                    ax_hrp_hist.grid(True, alpha=0.3)
                    
                    # Format x-axis as percentage
                    ax_hrp_hist.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: '{:.1%}'.format(x)))
                    
                    plt.tight_layout()
                    st.pyplot(fig_hrp_hist)
                
                with col2_hist:
                    # User Portfolio histogram with fitted distributions
                    fig_user_hist, ax_user_hist = plt.subplots(figsize=(10, 6))
                    
                    user_returns = user_performance['portfolio_returns'].dropna()
                    
                    # Plot histogram
                    n_bins = min(50, max(10, len(user_returns) // 10))
                    ax_user_hist.hist(user_returns, bins=n_bins, density=True, alpha=0.7, 
                                    color='red', edgecolor='black', linewidth=0.5)
                    
                    # Fit distributions
                    x_user = np.linspace(user_returns.min(), user_returns.max(), 100)
                    
                    # Kernel Density Estimation (non-parametric)
                    try:
                        kde_user = gaussian_kde(user_returns)
                        kde_pdf_user = kde_user(x_user)
                        ax_user_hist.plot(x_user, kde_pdf_user, 'g-', linewidth=2.5, 
                                        label='KDE (Kernel Density)')
                    except:
                        pass

                    # Add VaR line
                    ax_user_hist.axvline(user_performance['var_99'], color='blue', linestyle='--', 
                                       label=f"99% VaR: {user_performance['var_99']:.2%}")
                    
                    ax_user_hist.set_title('User Portfolio Returns Distribution', fontweight='bold')
                    ax_user_hist.set_xlabel('Daily Returns')
                    ax_user_hist.set_ylabel('Density')
                    ax_user_hist.legend()
                    ax_user_hist.grid(True, alpha=0.3)
                    
                    # Format x-axis as percentage
                    ax_user_hist.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: '{:.1%}'.format(x)))
                    
                    plt.tight_layout()
                    st.pyplot(fig_user_hist)
                
                
                # Rolling performance comparison
                st.subheader("Rolling Performance Metrics")
                
                col1_roll, col2_roll = st.columns(2)
                
                with col1_roll:
                    # Rolling Sharpe ratio (30-day window)
                    window = min(30, len(returns) // 4)  # Adaptive window size
                    if window >= 5:  # Only calculate if we have enough data
                        hrp_rolling_sharpe = (hrp_performance['portfolio_returns'].rolling(window).mean() * np.sqrt(252)) / (hrp_performance['portfolio_returns'].rolling(window).std() * np.sqrt(252))
                        user_rolling_sharpe = (user_performance['portfolio_returns'].rolling(window).mean() * np.sqrt(252)) / (user_performance['portfolio_returns'].rolling(window).std() * np.sqrt(252))
                        
                        fig_sharpe, ax_sharpe = plt.subplots(figsize=(10, 6))
                        ax_sharpe.plot(dates, hrp_rolling_sharpe, label=f'HRP Portfolio ({window}d)', alpha=0.8)
                        ax_sharpe.plot(dates, user_rolling_sharpe, label=f'User Portfolio ({window}d)', alpha=0.8)
                        ax_sharpe.set_title(f'Rolling Sharpe Ratio ({window}-day window)')
                        ax_sharpe.set_xlabel('Date')
                        ax_sharpe.set_ylabel('Sharpe Ratio')
                        ax_sharpe.legend()
                        ax_sharpe.grid(True, alpha=0.3)
                        plt.xticks(rotation=45)
                        plt.tight_layout()
                        st.pyplot(fig_sharpe)
                
                with col2_roll:
                    # Rolling volatility
                    if window >= 5:
                        hrp_rolling_vol = hrp_performance['portfolio_returns'].rolling(window).std() * np.sqrt(252)
                        user_rolling_vol = user_performance['portfolio_returns'].rolling(window).std() * np.sqrt(252)
                        
                        fig_vol, ax_vol = plt.subplots(figsize=(10, 6))
                        ax_vol.plot(dates, hrp_rolling_vol, label=f'HRP Portfolio ({window}d)', alpha=0.8)
                        ax_vol.plot(dates, user_rolling_vol, label=f'User Portfolio ({window}d)', alpha=0.8)
                        ax_vol.set_title(f'Rolling Volatility ({window}-day window)')
                        ax_vol.set_xlabel('Date')
                        ax_vol.set_ylabel('Annualized Volatility')
                        ax_vol.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: '{:.1%}'.format(y)))
                        ax_vol.legend()
                        ax_vol.grid(True, alpha=0.3)
                        plt.xticks(rotation=45)
                        plt.tight_layout()
                        st.pyplot(fig_vol)
                
            else:
                st.warning("Could not calculate portfolio performance metrics due to insufficient data.")

        except RuntimeError as e:
            if "ledoit_wolf" in str(e).lower() and "positive semi-definite" in str(e).lower():
                st.error(f"Covariance matrix error: {e}. This can happen with too few data points or highly correlated assets. Try a longer date range or different assets.")
            else:
                st.error(f"A runtime error occurred during analysis: {e}")
        except ValueError as e:
            st.error(f"A value error occurred: {e}. Check data inputs and ranges.")
        except Exception as e:
            st.error(f"An unexpected error occurred: {e}")
            # st.exception(e) # For more detailed traceback during development

else:
    st.info("Configure your portfolio settings in the sidebar and click 'Run'.")
