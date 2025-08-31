import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller, grangercausalitytests
from statsmodels.tsa.api import VAR
from statsmodels.tsa.vector_ar.var_model import VARResults
from statsmodels.tools.eval_measures import rmse, aic
import warnings
warnings.filterwarnings('ignore')


class GrangerCausalityAnalysis:
    """
    A class for performing comprehensive Granger causality analysis on time series data.
    
    This class implements a step-by-step approach:
    1. Check stationarity of the time series
    2. Correct for non-stationarity if needed
    3. Select optimal lag order using information criteria
    4. Perform preliminary VAR analysis
    5. Test Granger causality on promising variable combinations
    """
    
    def __init__(self, data=None):
        """
        Initialize the GrangerCausalityAnalysis class.
        
        Parameters:
        -----------
        data : pandas.DataFrame, optional
            DataFrame containing the time series data
        """
        self.data = data
        self.stationary_data = None
        self.var_model = None
        self.optimal_lag = None
        self.diff_order = {}
        self.causality_results = {}
    
    def check_stationarity(self, series=None, alpha=0.05, verbose=True):
        """
        Check stationarity of time series using Augmented Dickey-Fuller test.
        
        Parameters:
        -----------
        series : pandas.Series or str, optional
            Series to test or column name in self.data
        alpha : float, optional
            Significance level for the test
        verbose : bool, optional
            Whether to print results
            
        Returns:
        --------
        dict : Results of stationarity tests
        """
        results = {}

        for column in self.data.columns:
            series_data = self.data[column].dropna()
            result = adfuller(series_data)
            is_stationary = result[1] < alpha
            results[column] = {
                'Test Statistic': result[0],
                'p-value': result[1],
                'Critical Values': result[4],
                'Stationary': is_stationary
            }
            
            if verbose:
                print(f"Series '{column}': {'Stationary' if is_stationary else 'Non-stationary'} "
                        f"(p-value: {result[1]:.4f}, ADF: {result[0]:.4f})")
                    
        return results
    
    def make_stationary(self, max_diff=2, verbose=True, regression='c', autolag='AIC'):
        """
        Transform non-stationary series to stationary by differencing.
        
        Parameters:
        -----------
        max_diff : int, optional
            Maximum number of differences to apply
        verbose : bool, optional
            Whether to print information about the differencing process
        regression : str, optional
            Regression type for the ADF test. Options:
            'c' : constant only (default)
            'ct' : constant and trend
            'ctt' : constant, linear and quadratic trend
            'n' : no regression components
        autolag : str or None, optional
            Method for lag selection in ADF test. Options:
            'AIC' : Akaike Information Criterion (default)
            'BIC' : Bayesian Information Criterion
            't-stat' : Based on t-statistic significance
            None : No automatic lag selection
            
        Returns:
        --------
        pandas.DataFrame : Stationary data
        """
        stationary_data = self.data.copy()
        
        # Check and transform each column
        for column in stationary_data.columns:
            is_stationary = False
            d = 0
            series = stationary_data[column]
            
            # Check initial stationarity
            test_result = adfuller(series.dropna())
            is_stationary = test_result[1] < 0.05
            
            # Apply differencing until stationary or max_diff reached
            while not is_stationary and d < max_diff:
                # Apply differencing
                series = series.diff().dropna()
                d += 1
                
                # Check if stationary after differencing
                test_result = adfuller(series.dropna(),regression = regression,autolag=autolag)
                is_stationary = test_result[1] < 0.05
            
            # Store differencing order
            self.diff_order[column] = d
            
            # Apply the differencing to the data
            if d > 0:
                stationary_data[column] = stationary_data[column].diff(d).dropna()
                
        # Align all series after differencing (they might have different lengths)
        self.stationary_data = stationary_data.dropna()
        
        if verbose:
            print("Differencing applied:")
            for var, order in self.diff_order.items():
                print(f"  {var}: {order} difference(s)")
            print(f"Final data shape: {self.stationary_data.shape}")
            
        return self.stationary_data
    
    def select_lag_order(self, max_lag=10, criterion='BIC', verbose=True):
        """
        Select optimal lag order for VAR model using information criteria.
        
        Parameters:
        -----------
        max_lag : int, optional
            Maximum lag order to consider
        verbose : bool, optional
            Whether to print information about lag selection
            
        Returns:
        --------
        dict : Optimal lag orders according to different criteria
        """            
        model = VAR(self.stationary_data)
        results = model.select_order(maxlags=max_lag)
        
        # Get the optimal lags according to different criteria
        optimal_lags = { 'AIC': results.aic, 'BIC': results.bic, 'FPE': results.fpe,'HQIC': results.hqic}
        
        # Use BIC as default (tends to be more conservative)
        self.optimal_lag = optimal_lags[criterion]
            
        if verbose:
            print("Optimal lag selection by information criteria:")
            for criterion_i, lag_i in optimal_lags.items():
                print(f"  {criterion_i}: {lag_i}")
            print(f"Selected lag ({criterion}): {self.optimal_lag}")
            
        return optimal_lags
    
    def perform_var_analysis(self, lag=None, verbose=True,):
        """
        Perform preliminary VAR analysis.
        
        Parameters:
        -----------
        lag : int, optional
            Lag order for VAR model (uses optimal lag if None)
        verbose : bool, optional
            Whether to print VAR analysis results
            
        Returns:
        --------
        statsmodels.tsa.vector_ar.var_model.VARResults : VAR model results
        """
        if lag is None:
            lag = self.optimal_lag
            
        model = VAR(self.stationary_data)
        self.var_model = model.fit(lag)
        
        if verbose:
            print(f"\nVAR Analysis Summary (lag order = {lag}):")
            print("----------------------------------------")
            print(f"Number of observations: {self.var_model.nobs}")
            print(f"Log likelihood: {self.var_model.llf:.4f}")
            print(f"AIC: {self.var_model.aic:.4f}")
            print(f"BIC: {self.var_model.bic:.4f}")
            #print(f"Durbin-Watson: {self.var_model.durbin_watson()}")
            
        return self.var_model
    
    def select_promising_combinations(self, threshold=0.05):
        """
        Select promising variable combinations based on VAR results.
        
        Parameters:
        -----------
        threshold : float, optional
            P-value threshold for significance
            
        Returns:
        --------
        list : List of promising (cause, effect) combinations
        """
        if self.var_model is None:
            self.perform_var_analysis(verbose=False)
            
        promising_combinations = []
        variable_names = self.stationary_data.columns
        
        # Analyze coefficients from VAR model
        for i, effect in enumerate(variable_names):
            for j, cause in enumerate(variable_names):
                if i == j:  # Skip self-causality
                    continue
                    
                # Check if coefficients of cause are significant in effect equation
                # Get positions for the cause variable in each lag
                cause_indices = range(j, len(variable_names) * self.var_model.k_ar, len(variable_names))
                
                # Extract coefficients and p-values for these positions in the effect equation
                coeffs = [self.var_model.params.iloc[idx, i] for idx in cause_indices]
                p_values = [self.var_model.pvalues.iloc[idx, i] for idx in cause_indices]
                
                # If any coefficient is significant, add to promising combinations
                if any(p < threshold for p in p_values):
                    promising_combinations.append((cause, effect))
        
        return promising_combinations
    
    def test_granger_causality(self, variables=None, max_lag=None, verbose=True):
        """
        Perform Granger causality tests.
        
        Parameters:
        -----------
        variables : list of tuples, optional
            List of (cause, effect) tuples to test
        max_lag : int, optional
            Maximum lag to test (uses optimal lag if None)
        verbose : bool, optional
            Whether to print causality test results
            
        Returns:
        --------
        dict : Granger causality test results
        """       
        if max_lag is None:
            max_lag = self.optimal_lag
            
        # If still None (e.g., no promising combinations), test all pairs
        if not variables:
            columns = list(self.stationary_data.columns)
            variables = [(x, y) for x in columns for y in columns if x != y]
            
        # Perform Granger causality tests
        results = {}
        for cause, effect in variables:
            # Test if cause Granger-causes effect
            data = self.stationary_data[[effect, cause]]  # [y, x] tests if x Granger-causes y
            gc_result = grangercausalitytests(data, maxlag=max_lag, verbose=False)
            
            # Extract p-values for each lag (using F-test)
            p_values = {lag: round(result[0]['ssr_ftest'][1], 4) for lag, result in gc_result.items()}
            
            # Determine if there's causality at any lag
            causality = any(p < 0.05 for p in p_values.values())
            min_p = min(p_values.values())
            significant_lags = [lag for lag, p in p_values.items() if p < 0.05]
            
            # Store results
            results[(cause, effect)] = {
                'p_values': p_values,
                'causality': causality,
                'min_p_value': min_p,
                'significant_lags': significant_lags
            }
            
            if verbose:
                status = "Granger causes" if causality else "does NOT Granger cause"
                print(f"{cause} {status} {effect} (min p-value: {min_p:.4f}, "
                      f"significant lags: {significant_lags if significant_lags else 'none'})")
        
        self.causality_results = results
        return results
    
    def full_analysis(self, max_lag=10, max_diff=2, criterion='BIC',make_stationnary_regression='c', make_stationnary_autolag='AIC'):
        """
        Perform complete Granger causality analysis pipeline.
        
        Parameters:
        -----------
        max_lag : int, optional
            Maximum lag order to consider
        max_diff : int, optional
            Maximum differencing order
            
        Returns:
        --------
        dict : Analysis results
        """
        if self.data is None:
            raise ValueError("No data available. Please set data first.")
            
        print("====== GRANGER CAUSALITY ANALYSIS ======\n")
        
        # Step 1: Check stationarity
        print("STEP 1: Checking stationarity of time series")
        print("-------------------------------------------")
        stationarity_results = self.check_stationarity()
        print()
        
        # Step 2: Make data stationary if needed
        print("STEP 2: Making time series stationary")
        print("------------------------------------")
        if any(not result['Stationary'] for result in stationarity_results.values()):
            self.make_stationary(max_diff=max_diff,regression=make_stationnary_regression, autolag=make_stationnary_autolag)
        else:
            self.stationary_data = self.data.copy()
            print("All series are already stationary. No differencing applied.")
        print()
        
        # Step 3: Select optimal lag order
        print("STEP 3: Selecting optimal lag order")
        print("---------------------------------")
        lag_criteria = self.select_lag_order(max_lag=max_lag, criterion=criterion)
        print()
        
        # Step 4: Perform VAR analysis
        print("STEP 4: Performing VAR analysis")
        print("-----------------------------")
        var_model = self.perform_var_analysis()
        print()
        
        # Step 5: Test Granger causality
        print("STEP 5: Testing Granger causality")
        print("-------------------------------")
        promising = self.select_promising_combinations()
        
        if promising:
            print(f"Found {len(promising)} promising variable combinations:")
            for cause, effect in promising:
                print(f"  {cause} → {effect}")
            print("\nGranger causality test results:")
            print("-----------------------------")
        else:
            print("No promising combinations found. Testing all variable pairs.")
            
        causality_results = self.test_granger_causality(promising)
        
        # Summarize results
        print("\nSUMMARY:")
        print("--------")
        causal_pairs = [(cause, effect) for (cause, effect), result in self.causality_results.items() 
                        if result['causality']]
        
        if causal_pairs:
            print("Detected Granger causal relationships:")
            for cause, effect in causal_pairs:
                print(f"  {cause} → {effect}")
        else:
            print("No significant Granger causal relationships detected.")
            
        return {
            'stationarity': stationarity_results,
            'differencing': self.diff_order,
            'optimal_lag': self.optimal_lag,
            'var_model': self.var_model,
            'causality_results': self.causality_results
        }


if __name__ == "__main__":
    from pipeline.generate_variables import load_variables_sinusoidales
    t,x,y,z = load_variables_sinusoidales(n=100,T=2*np.pi,lag=np.pi/3,noise=True,seed=42,cos=True)

    # Create a DataFrame with your time series
    df = pd.DataFrame({
        'x': x,
        'y': y,
        'z': z
    }, index=t)

    # Initialize the Granger causality analysis with your data
    gc = GrangerCausalityAnalysis(df)

    # Run the full analysis pipeline
    results = gc.full_analysis()

    # Or run individual steps as needed:
    # gc.check_stationarity()
    # gc.make_stationary()
    # gc.select_lag_order()
    # gc.perform_var_analysis()
    # promising_combinations = gc.select_promising_combinations()
    # gc.test_granger_causality(promising_combinations)

    causal_relationships = [(cause, effect) for (cause, effect), result 
                        in results['causality_results'].items() 
                        if result['causality']]