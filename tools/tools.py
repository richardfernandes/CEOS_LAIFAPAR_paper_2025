import pandas as pd
import glob
import os
from google.cloud import storage
from io import BytesIO
import numpy as np
from statsmodels.stats.weightstats import DescrStatsW
from scipy.optimize import curve_fit
from scipy.interpolate import make_splrep
from scipy import stats
from numpy.polynomial import Chebyshev as T
from numpy.polynomial.chebyshev import chebvander
from sklearn.preprocessing import MinMaxScaler
from numpy.polynomial.chebyshev import chebvander
import Algos
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
import thielsen as thielsen

def get_bias_year(year,df, df_FRM,r,r_ci,x,x_ci,x_ci_minimum,w,inc,exc,removeUFRMflag=True):
    """
    Estimates bias of product estimates using a dataframe of FRM matchups
    where both have columns required to determine similarity.


    Parameters:
    -----------
    df: pd.DataFrame
        a single product estimate and ancillary quantities need to compute similarity 
    df_FRM : pd.DataFrame
        FRM matchups including residual
    r: string
        df_FRM column name corresponding to residual
    r_ci: string
        df_FRM column name corresponding to standard uncertainty of residual
    x: list
        self and df_FRM column names that determine similarity in error conditions between product estimate and df_FRM matchps
    x_ci: list
        self and df_FRM column names that determine confidence interal of x
    x_ci_minimum: list
        minimum x_ci values
    w: list 
        weights for each x column
    inc: list of lists
        list of lists of column of which one sublist that must match between a given product estimate and df_FRM matchup
    exc: list of lists
        list of lists of column of which no sublist can match between a given product estimate and df_FRM matchup   
    removeUFRMflag: boolean
        removes the FRM uncertainty from estimated uncertainty is true

    Returns
    -------
    N: effective degrees of freedom
    B: bias
    B_ci_upper: size of upper one standard deviation confidence interval of B
    B_ci_lower: size of lower one standard deviation confidence interval of B

    
    """
    #force year of the product measurement df
    df['year'] = year


    # filter matchups based on inclusion and exclusion column matches

    df_FRM = filter_FRM(df, df_FRM, inc,exc)
    if not df_FRM.empty:
        #compute similarity between between product estimate condition and each FRM
        weights = df.apply(estimate_similarity_weights,axis=1,args=(df_FRM,x,x_ci,w,x_ci_minimum)).to_numpy()[0]
    
        
        #now incorporate uncertainty of residuals, with a lower bound specified to avoid infinite weights
        r_ci_lowerbound = np.quantile(df_FRM[r_ci].to_numpy(),0.20)

        weights = weights / np.maximum(np.power(df_FRM[r_ci].to_numpy(),2),r_ci_lowerbound ) 
        if (np.isnan(weights).any() or np.isinf(weights).any() or np.all(weights == 0)):
            weights = np.ones(len(weights))
        weights = weights / np.sum(weights)

            
        # compute validation statistics for these product estimates 
        # if there are multiple the result with be a list of statistics 

        B = estimate_B(df_FRM[r].to_numpy(),weights)

            
        #estimate confidence interval widths
        B['ci'] = B['ci_upper']+B['ci_lower']
        B['N'] = estimate_N(weights)
    else:
        B = {'q': np.nan , 'ci_upper': np.nan, 'ci_lower': np.nan, 'ci': np.nan, 'N':np.nan} 


    return {'B':B,'year':year}

def get_stability(df,df_FRM,params):

    a=pd.Series(df_FRM['year'].unique(),name='year').apply(lambda year: get_bias_year(year,
                                                                df,                                                        
                                                               params['df_FRM'],
                                                               params['r'],
                                                               params['r_ci'],
                                                               params['x'],
                                                               params['x_ci'],
                                                               params['x_ci_minimum'],
                                                               params['w'],
                                                                params['inc'],
                                                                [['year']],
                                                                params['removeUFRMflag']))

    

    df1 = pd.DataFrame(a.to_list()).dropna()
    df1['B_est'] =  df1['B'].str.get('q').to_numpy()
    df1['B_lower'] = df1['B'].str.get('ci_lower').to_numpy()
    df1['B_upper'] =  df1['B'].str.get('ci_upper').to_numpy()
    
    #drop years with no accuracy of duplicate accuracy
    df1 = df1.dropna().drop_duplicates(subset=['B_est'], keep='first')
    if (df1.shape[0]>1):
        S=weightedthielslopes(df1['year'],df1['B_est'],1/np.power((df1['B_upper'].to_numpy()-df1['B_lower'].to_numpy())/2,2)) 
        S['N'] = df1.shape[0] * (df1.shape[0]-1 )
    else:
        S = {'q':np.nan,'ci_upper':np.nan,'ci_lower':np.nan, 'N':np.nan}
    return S

def stability(df, df_FRM, variable ,params):
    result = df.apply(lambda row: get_stability(pd.DataFrame(row).T, df_FRM, variable ,params),axis=1)
    return result    

def weightedthielslopes(x, y, sample_weights):
    """
    Calculates the median of pairwise slopes, weighted by the Euclidean 
    combination of the user-specified weights for each sample.
    
    Parameters:
    - x: array-like, independent variable (length N)
    - y: array-like, dependent variable (length N)
    - sample_weights: array-like, user-specified weight for each sample (length N)
    
    Returns:
    - float: The weighted median slope.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    w = np.asarray(sample_weights, dtype=float)
    
    n = len(x)
    if n != len(y) or n != len(w):
        raise ValueError("x, y, and sample_weights must all have the same length.")
    if n < 2:
        raise ValueError("At least two data points are required.")
        
    slopes = []
    pairwise_weights = []

    # Compute pairwise slopes and Euclidean-combined weights
    for i in range(n):
        for j in range(i + 1, n):
            dx = x[j] - x[i]
            if dx == 0:
                continue  # Skip vertical lines to prevent division by zero
                
            slope = (y[j] - y[i]) / dx
            slopes.append(slope)
            
            # Calculate the Euclidean sum of the two sample weights
            pairwise_w = 1/np.sqrt(w[i]**2 + w[j]**2)
            pairwise_weights.append(pairwise_w)
            
    slopes = np.array(slopes)
    pairwise_weights = np.array(pairwise_weights)
    
    num_slopes = len(slopes)
    if num_slopes == 0:
        raise ValueError("No valid pairwise slopes found (all x-values are identical).")
        
    # Sort slopes and weights together
    sort_idx = np.argsort(slopes)
    slopes = slopes[sort_idx]
    pairwise_weights = pairwise_weights[sort_idx]
    
    # Calculate the weighted median
    cum_weights = np.cumsum(pairwise_weights)

    
    # Find where the cumulative weight hits or crosses the 50% threshold, , 0.5-0.675/2, 0.5+0.765/2
    slope = slopes[np.maximum(0,np.minimum(num_slopes-1,np.searchsorted(cum_weights, cum_weights[-1] *0.5)))]
    upper_ci = slopes[np.maximum(0,np.minimum(num_slopes-1,np.searchsorted(cum_weights, cum_weights[-1] *(0.5-0.675/2))))]
    lower_ci= slopes[np.maximum(0,np.minimum(num_slopes-1,np.searchsorted(cum_weights, cum_weights[-1] *(0.5+0.675/2))))]

    return {'slope':slope,'upper_ci':upper_ci,'lower_ci':lower_ci}

def compute_weights(x_ci):
    return 1/np.power(np.maximum(x_ci/2,np.quantile(x_ci/2,0.2)),2)
    
class ThielSenChebyshevRegressor:
    def __init__(self, degree=2):
        self.degree = degree
        self.model = None


    def fit(self, x, y):
        """
        Fits the a Chebyshev model using Thiel Sen
        """

        self.model = thielsen.TheilSenRegressor(random_state=42)
        self.fit(chebvander(x,self.degree), y)

       
        return self

    def predict(self, x_new):
        """
        Predicts y values 
        """
        if self.model is None:
            raise ValueError("Model must be fitted before calling predict.")

        return  self.model.predict(chebvander(x_new,self.degree)),
    
class WeightedChebyshevRegressor:
    def __init__(self, degree=1, alpha=0.05):
        self.degree = degree
        self.alpha = alpha
        self.model = None
        self.scaler = MinMaxScaler(feature_range=(-1, 1))
        self.cov_matrix = None
        self.dof = None

    def fit(self, x, y, weights):
        """
        Fits the weighted Chebyshev model and calculates the covariance matrix.
        weights: 1/sigma^2
        """
        # 1. Fit the polynomial
        self.model = T.fit(x, y, deg=self.degree, w=weights)
        
        # 2. Scale x to [-1, 1] for the Vandermonde matrix
        x_reshaped = x.reshape(-1, 1)
        self.scaler.fit(x_reshaped)
        x_scaled = self.scaler.transform(x_reshaped).flatten()
        
        # 3. Calculate Covariance Matrix for weighted least squares
        # Design Matrix V
        V = chebvander(x_scaled, self.degree)
        
        # W is the diagonal weight matrix. 
        # Cov = (V.T * W * V)^-1
        W = np.diag(weights)
        self.cov_matrix = np.linalg.inv(V.T @ W @ V)
        
        # 4. Degrees of freedom
        self.dof = len(x) - (self.degree + 1)
        
        return self

    def predict(self, x_new):
        """
        Predicts y values and returns confidence intervals.
        """
        if self.model is None:
            raise ValueError("Model must be fitted before calling predict.")

        x_new = x_new.reshape(-1, 1)
        # 1. Predict point estimates
        y_hat = self.model(x_new).flatten()
        
        # 2. Scale new x and generate Vandermonde matrix
        x_new_scaled = self.scaler.transform(x_new).flatten()
        V_new = chebvander(x_new_scaled, self.degree)

        # 3. Calculate Standard Error of Prediction
        # SE = sqrt(diag(V_new * Cov * V_new^T))
        step1 = V_new @ self.cov_matrix
        prediction_var = np.sum(step1 * V_new, axis=1)
        se_pred = np.sqrt(prediction_var)
        
        # 4. Confidence Intervals using t-distribution
        t_val = stats.t.ppf(1 - self.alpha / 2, self.dof)
        
        lower_ci = y_hat-t_val * se_pred
        upper_ci = y_hat+t_val * se_pred
        
        return {
            "y_hat": y_hat,
            "lower_ci": lower_ci,
            "upper_ci": upper_ci,
            "se_pred": se_pred
        }

def constant_model(y):
    return y
    
class ConstantRegressor:
    def __init__(self):
        self.model = None
        self.dof = None

    def fit(self, x,y):
        """
        Fits the weighted Chebyshev model and calculates the covariance matrix.
        weights: 1/sigma^2
        """
        # 1. Fit 
        self.model = constant_model(y)

        
        # 4. Degrees of freedom
        self.dof = 1
        
        return self

    def predict(self, x_new):
        """
        Predicts y values and returns confidence intervals.
        """
        if self.model is None:
            raise ValueError("Model must be fitted before calling predict.")

        # 1. Predict point estimates
        y_hat = self.model(x_new)
        
        # 2. Scale new x and generate Vandermonde matrix
        se_pred = 0
        
        lower_ci = y_hat
        upper_ci = y_hat
        
        return {
            "y_hat": y_hat,
            "lower_ci": lower_ci,
            "upper_ci": upper_ci,
            "se_pred": se_pred
        }


def polyfit_robust(x,y,ci,degree):

    w = 1/np.maximum(np.abs(0.01*y),(ci/2))
    if (len(x) < 2):
        model = ConstantRegressor().fit(x=x, y=y)
    elif (len(x) < 10 ):
        model = LinearRegression().fit(x.reshape(-1, 1) , y, w)
    else:
        model = WeightedChebyshevRegressor(degree=degree, alpha=0.05).fit(x=x, y=y, weights=w)
    return model
    
def weighted_quantile(X, W, alpha):
    """
    Weighted quantile estimation
    Parameters
    ----------
    X : array
        main random variable in 
    W : array
        weight random variable in 
    alpha : float
        risk level in [0,1]

    Returns
    -------
    weighted_quantile: float
    """
    return np.quantile(a=X, q=alpha, weights=W, method='inverted_cdf')

def std_Z(X, W, alpha, qW_hat):
    """
    Standard deviation of the variable Z
    Parameters
    ----------
    X : array
        main random variable in 
    W : array
        weight random variable in 
    alpha : float
        risk level in [0,1]
    qW_hat : array
        estimated weighted quantile

    Returns
    -------
    Z: float or array
    """
    return np.sqrt(np.mean(W**2 * (alpha - (X<=qW_hat))**2))
    
def confidence_interval_qW(X, W, alpha, eta):
    """
    Confidence interval estimation of the weighted quantile

    Parameters
    ----------
    X : array
        main random variable in
    W : array
        weight random variable in 
    alpha : float
        risk level in [0,1]
    eta : float in [0, 1]
        confidence level

    Returns
    -------
    ci_lower: width of lower eta confidence interval
    ci_right: width of upper eta confidence interval
    q: weighted quantile
    """
    qW_hat = weighted_quantile(X, W, alpha)  # Weighted quantile estimation
    c_eta = stats.norm.ppf(1 - (1-eta)/2)  # confidence threshold
    n = len(X)  # number of samples
    c = c_eta * std_Z(X, W, alpha, qW_hat) / np.mean(W)
    alpha_left = np.maximum(0, alpha-(c/np.sqrt(n)))
    alpha_right = np.minimum(alpha + (c/np.sqrt(n)), 1)


    ci_left =qW_hat -  weighted_quantile(X, W, alpha_left)
    ci_right = weighted_quantile(X, W, alpha_right) - qW_hat
    return {'ci_lower':abs(float(ci_left)) , 'ci_upper': abs(float(ci_right)) , 'q': abs(float(qW_hat))  * np.sign(qW_hat)}
    
def estimate_N(weights):
    return np.power(np.sum(weights),2)/np.sum(np.power(weights,2))

def estimate_B(residuals,weights):

    B = confidence_interval_qW(residuals, weights, 0.5, eta=0.67)
    B['ci'] = (B['ci_upper'] - B['ci_lower'])/2
    B['N'] = estimate_N(weights) 

    return B 
    
def estimate_A(residuals,weights):

    A = confidence_interval_qW(np.abs(residuals), weights, 0.5, eta=0.67)
    A['ci'] = (A['ci_upper'] - A['ci_lower'])/2
    A['N'] = estimate_N(weights) 

    return A

def estimate_U(residuals,weights):

    B = confidence_interval_qW(residuals, weights, 0.5, eta=0.67)
    U = confidence_interval_qW(np.abs(residuals - B['q']), weights, 0.5, eta=0.67)
    
    # increase confidence intervals due to uncertainty in B 
    U['N'] = estimate_N(weights)
    U['ci_lower'] = np.sqrt(np.power(U['ci_lower'],2)+np.power(B['ci_lower']/np.sqrt(U['N']),2)) 
    U['ci_upper'] = np.sqrt(np.power(U['ci_upper'],2)+np.power(B['ci_upper']/np.sqrt(U['N']),2)) 
    U['ci'] = (U['ci_upper'] - U['ci_lower'])/2
    

    return U
    
def sum_if_2d(arr):
    if arr.ndim == 2:
        return arr.sum(axis=1)
    return arr

def estimate_similarity(row,df_FRM,col_names,col_names_ci,bandwidth_weights,ci_minimum):
    """
    Estimates similarity between a vector x and an array y of the same columns
    based on assuming a Gaussian similarity function and uncorrelated column values.
    
    The variance is specified by the euclidean sums of confidence intervals of the series value and each dataframe value.
    
    The confidence intervals correspond to the products of a provided bandwidth_weight and the maximum of input values and provided minimum intervals.  
    
    Parameters
    ----------
    x : np.array()
        1d array of observation for which similarity is assessed
    x_ci: np.array()
        1d array of confidence interval of observation for which similarity is assessed
    y : np.array()
        2d array of observations to which the similarity of x is assessed
    y_ci: np.array()
        2d array of confidence intervals of observations to which the similarity of x is assessed
    bandwidth_weights:  np.array()
        1d array of weights of bandwidths for computing similarity - large weight means similarity is easier
    ci_minimum :  np.array()
        1d array of minimum ci values 
    
        
    -------
    weights : np.array()
        normalzied weights between x and FRM
    """    

    x = row[col_names].to_numpy()
    x_ci = row[col_names_ci].to_numpy()
    y = df_FRM[col_names].to_numpy()
    y_ci = df_FRM[col_names_ci].to_numpy()
    bandwidths = np.multiply(bandwidth_weights, np.sqrt((np.power(np.maximum(x_ci,ci_minimum),2) + np.power(np.maximum(y_ci,ci_minimum),2)).astype(float)))
    # print('x:',x)
    # print('y:',y)
    # print('bw:',bandwidths)
    # print('delta:',x-y)

    distances = sum_if_2d(np.power((x-y)/bandwidths,2))
    weights=np.exp(-0.5*distances.astype(float))
    # print('weights:',weights)
    return weights

def estimate_similarity_weights(row,df_FRM,col_names,col_names_ci,bandwidth_weights,ci_minimum):
    """
    Estimates weights proportional to similarity between a vector x and an array y of the same columns
    based on assuming a Gaussian similarity function and uncorrelated column values.
    
    The variance is specified by the euclidean sums of confidence intervals of the series value and each dataframe value.
    
    The confidence intervals correspond to the products of a provided bandwidth_weight and the maximum of input values and provided minimum intervals.  
    
    Parameters
    ----------
    x : np.array()
        1d array of observation for which similarity is assessed
    x_ci: np.array()
        1d array of confidence interval of observation for which similarity is assessed
    y : np.array()
        2d array of observations to which the similarity of x is assessed
    y_ci: np.array()
        2d array of confidence intervals of observations to which the similarity of x is assessed
    bandwidth_weights:  np.array()
        1d array of weights of bandwidths for computing similarity - large weight means similarity is easier
    ci_minimum :  np.array()
        1d array of minimum ci values 
    
        
    -------
    weights : np.array()
        normalzied weights between x and FRM
    """    

    x = row[col_names].to_numpy()
    x_ci = row[col_names_ci].to_numpy()
    y = df_FRM[col_names].to_numpy()
    y_ci = df_FRM[col_names_ci].to_numpy()

    bandwidths = np.multiply(bandwidth_weights, np.sqrt((np.power(np.maximum(x_ci,ci_minimum),2) + np.power(np.maximum(y_ci,ci_minimum),2)).astype(float)))
    distances = sum_if_2d(np.power((x-y)/bandwidths,2))
    weights=np.exp(-0.5*distances.astype(float))

    return weights / np.sum(weights)


    
def filter_FRM(df1, df2,inc,exc):
    """
    Filter a dataframe (self) retaining only rows that match columns in dataframe df specified by inc and do not match columns in df specified by exc.
    
    Parameters
    ----------
    df1 : pd.DataFrame()
        data frame being filtered 
    df2 : pd.DataFrame()
        data frame of matching rows 
    inc : list of list of strngs
        list of list of column names, any of which must be matched to be retained
    exc : list of list of strings
        list of list of column names, any of which must not be matched to be retained
        
    Returns
    -------
    df_final : pd.DataFrame()
        data frame of rows in self that are retained
    """
    #initialze final result
    df_final = df2

    #find all rows in df2 matching df1 for any of the inc columns
    if ( inc ):
        all_matches = []
        for cols in inc:
            # Find rows in df2 that match df1 on this specific set of columns
            match = df1.merge(df2, on=cols, how='left',suffixes=('_left',''))
            all_matches.append(match)
        # Combine all found rows and remove duplicates
        df_inc= pd.concat(all_matches).drop_duplicates()
        df_final = df_inc.drop(columns=df_inc.filter(regex=f'{'_left'}$').columns)

    #drop all rows in df_inc matching df1 for any of the exc columns
    if ( exc ):
        
        #iterate over all exc column combinations
        all_matches = []
        for cols in exc:

            # Find rows in current final df that match df1 on this specific set of columns
            match = df1[cols].merge(df_final, on=cols, how='left',suffixes=('_left',''))
            all_matches.append(match)
            
        #drop duplicate matches over all combinations
        df_exc= pd.concat(all_matches).drop_duplicates()
        
        #remove the additional columns from the product df
        df_exc = df_exc.drop(columns=df_exc.filter(regex=f'{'_left'}$').columns)
    
        # remove the excluded rows from the included rows
        df_final = pd.concat([df_final, df_exc]).drop_duplicates(keep=False)

    return df_final

def f1(row,df_FRM,bandwidth_weights,ci_minimum):
    print('test')
    return
    


def get_stats(df, df_FRM,params):
    """
    Estimates accuracy, bias and uncertainty of product estimates using a dataframe of FRM matchups
    where both have columns required to determine similarity.


    Parameters:
    -----------
    df: pd.Frame
        a single product estimate and ancillary quantities need to compute similarity 
    df_FRM : pd.DataFrame
        FRM matchups including residual
 

    Returns
    -------
    N: effective degrees of freedom
    A: apparent accuracy
    A_ci_upper: size of upper one standard deviation confidence interval of A
    A_ci_lower: size of lower one standard deviation confidence interval of A
    B: bias
    B_ci_upper: size of upper one standard deviation confidence interval of B
    B_ci_lower: size of lower one standard deviation confidence interval of B
    U: uncertainty
    U_ci_upper: size of upper one standard deviation confidence interval of U
    U_ci_lower: size of lower one standard deviation confidence interval of U
        
    """
    # filter matchups based on inclusion and exclusion column matches
    df_FRM = filter_FRM(df, df_FRM, params['inc'],params['exc'])

    #parse parameters                                                       
    r = params['r']
    r_ci = params['r_ci']
    x = params['x']
    x_ci = params['x_ci']
    x_ci_minimum = params['x_ci_minimum']
    w = params['w']
    removeUFRMflag = params['removeUFRMflag']

    if not df_FRM.empty:
        #compute similarity between between product estimate condition and each FRM
        #first based only on a weighted z-score 
        weights = df.apply(estimate_similarity_weights,axis=1,args=(df_FRM,x,x_ci,w,x_ci_minimum)).to_numpy()[0]
        similarity = df.apply(estimate_similarity,axis=1,args=(df_FRM,x,x_ci,w,x_ci_minimum)).to_numpy()[0]
        
        #now incorporate uncertainty of residuals, with a lower bound specified to avoid infinite weights
        r_ci_lowerbound = np.quantile(df_FRM[r_ci].to_numpy(),0.20)
        weights = weights / np.maximum(np.power(df_FRM[r_ci].to_numpy(),2),r_ci_lowerbound ) 
        if (np.isnan(weights).any() or np.isinf(weights).any() or np.all(weights == 0)):
            weights = np.ones(len(weights))
        weights = weights / np.sum(weights)

            
        # compute validation statistics for these product estimates 
        # if there are multiple the result with be a list of statistics 
        A = estimate_A(df_FRM[r].to_numpy(),weights)
        B = estimate_B(df_FRM[r].to_numpy(),weights)
        U = estimate_U(df_FRM[r].to_numpy(),weights) 

        if ( removeUFRMflag ) :
            #increase U CI to account for uncertanity in FRM
            U_FRM = estimate_B(df_FRM[r_ci].to_numpy(),weights)
            U['q'] = np.sqrt(np.maximum(0,np.power(U['q'],2)-np.power(U_FRM['q'],2)))
            U['ci_upper'] = np.sqrt(np.power(U['ci_upper'],2)+np.power(U_FRM['ci_upper']/np.sqrt(B['N']),2))
            U['ci_lower'] = np.sqrt(np.power(U['ci_lower'],2)+np.power(U_FRM['ci_lower']/np.sqrt(B['N']),2))
            U['ci'] = U['ci_upper']+U['ci_lower']
        S = get_stability(df, df_FRM, params)
    else:
        A = {'q':np.nan,'ci_upper':np.nan,'ci_lower':np.nan,'ci':np.nan,'N':np.nan} 
        B = {'q':np.nan,'ci_upper':np.nan,'ci_lower':np.nan,'ci':np.nan,'N':np.nan} 
        U = {'q':np.nan,'ci_upper':np.nan,'ci_lower':np.nan,'ci':np.nan,'N':np.nan} 
        S = {'q':np.nan,'ci_upper':np.nan,'ci_lower':np.nan,'ci':np.nan,'N':np.nan} 
        similarity = np.nan
        weights = np.nan

    return {'A':A,'B':B,'U':U,'S':S, 'W':weights, 'P':similarity}

def calval(df, variable, params):
    """
    Performance calval of product measurements in df using reference measurements in df_FRM for specified variable.

    Parameters:
    -----------
    df: pd.Frame
        a single product estimate and ancillary quantities need to compute similarity 
    df_FRM : pd.DataFrame
        FRM matchups including residual
    variable: string
        variable being validated (LAI, fAPAR)
    params: dictionary
        dictionary of validation parameters:
        r: string
                df_FRM column name corresponding to residual
            r_ci: string
                df_FRM column name corresponding to standard uncertainty of residual
            x: list
                self and df_FRM column names that determine similarity in error conditions between product estimate and df_FRM matchps
            x_ci: list
                self and df_FRM column names that determine confidence interal of x
            x_ci_minimum: list
                minimum x_ci values
            w: list 
                weights for each x column
            inc: list of lists
                list of lists of column of which one sublist that must match between a given product estimate and df_FRM matchup
            exc: list of lists
                list of lists of column of which no sublist can match between a given product estimate and df_FRM matchup   
            removeUFRMflag: boolean
                removes the FRM uncertainty from estimated uncertainty if true
        
    Returns
    -------
    validation: dictionary
        dictionary with validated product dataframe, parameters of validation, and polynomial fits of conditional uncertainty and accuracy
    """
    validation = {}
    validation['variable'] = variable
    validation['params'] = params
    validation['data']= df.merge( df.apply(lambda row: get_stats(row.to_frame().T,params['df_FRM'],params),
                                            axis=1).rename('validation_stats'), left_index=True, right_index=True)

    
    return validation

def visualize_validation(validation_dict_list,variable,groups,user_requirements,minimum_df=10,bivariate=False,legend=True):

    """
    Visualizes validation results of product measurements from a list of validation dictionaries.

    Parameters:
    -----------
    validation_dict_list: list
        List of validation dictionaries that include a dataframe of validation results 
    variable: string
        Variable validated (LAI, fAPAR)

    groups: string
        Column name of all validation dataframes for grouping used when visualization results 
    user_requirements: dictionary
        Dictionary of product requirements {Uabs,Urel,Sabs}
        minimum_df: integer
        Minimum degrees of freedom for valid matchups
    bivariate: boolean
        True if bivariate visualizations are shown between different experiments
    legend: boolean
        True if legends are to be shown
        
    Returns
    -------
    validation: dictionary
        dictionary with validated product dataframe, parameters of validation, and polynomial fits of conditional uncertainty and accuracy
    """    
     # Set limits 
    if ( variable == 'LAI' ):
        x_linspace = np.linspace(0,5,100)
    else:
        x_linspace = np.linspace(0,1,100)

    #conformity test
    #-1 does not conform, 0 inconclusive, 1 conforms
    for validation_dict in validation_dict_list:

        df1 = validation_dict['data']
        variable = validation_dict['variable']
        #augment df with standard quatities
        df1 = df1.assign(y = df1[variable+'_FRM'].to_numpy(),
                         x = df1['medianestimate'+variable].to_numpy(),
                         x_ci = df1['medianestimate'+variable+'_ci'].to_numpy(),
                         r = df1['medianresidual'+variable].to_numpy(),
                        r_ci = np.sqrt(np.power(df1['medianestimate'+variable+'_ci'].to_numpy(),2)+np.power(df1[variable+'_ci'].to_numpy(),2)),
                        A = df1['validation_stats'].str.get('A').str.get('q').to_numpy(),     
                        A_lower = 1.35*df1['validation_stats'].str.get('A').str.get('ci_lower').to_numpy(),
                        A_upper = 1.35*df1['validation_stats'].str.get('A').str.get('ci_lower').to_numpy(),
                        A_ci = 1.35*df1['validation_stats'].str.get('A').str.get('ci').to_numpy(),
                        A_N = df1['validation_stats'].str.get('A').str.get('N').to_numpy(),     
                        B = df1['validation_stats'].str.get('B').str.get('q').to_numpy(),     
                        B_lower = 1.35*df1['validation_stats'].str.get('B').str.get('ci_lower').to_numpy(),
                        B_upper = 1.35*df1['validation_stats'].str.get('B').str.get('ci_lower').to_numpy(),
                        B_ci = 1.35*df1['validation_stats'].str.get('B').str.get('ci').to_numpy(),
                        B_N = df1['validation_stats'].str.get('B').str.get('N').to_numpy(),     
                        U = df1['validation_stats'].str.get('U').str.get('q').to_numpy(),     
                        U_lower = 1.35*df1['validation_stats'].str.get('U').str.get('ci_lower').to_numpy(),
                        U_upper = 1.35*df1['validation_stats'].str.get('U').str.get('ci_lower').to_numpy(),
                        U_ci = 1.35*df1['validation_stats'].str.get('U').str.get('ci').to_numpy(),
                        U_N = df1['validation_stats'].str.get('U').str.get('N').to_numpy(),     
                        S = df1['validation_stats'].str.get('S').str.get('slope').to_numpy(),     
                        S_lower = df1['validation_stats'].str.get('S').str.get('upper_ci').to_numpy(),
                        S_upper = df1['validation_stats'].str.get('S').str.get('lower_ci').to_numpy(),
                        S_ci = df1['validation_stats'].str.get('S').str.get('ci').to_numpy(),
                        S_N = df1['validation_stats'].str.get('S').str.get('N').to_numpy()     
                        )
        
        #Evaluate conformity for user requirements
        df1['U_conformity'] = "Unknown"
        df1['U_upper_value'] = df1['U'] + df1['U_upper']
        df1['U_upper_relative'] = df1['U_upper_value'] /np.maximum(df1['medianestimate'+variable].to_numpy(),df1[variable+'_ci'].to_numpy())
        df1['U_lower_value'] = df1['U'] - df1['U_lower']
        df1['U_lower_relative'] = df1['U_lower_value'] /np.maximum(df1['medianestimate'+variable].to_numpy(),df1['medianestimate'+variable].to_numpy())
        df1.loc[(df1['U_lower_value']>user_requirements['Uabs']) & (df1['U_lower_relative']>user_requirements['Urel']),'U_conformity'] = "False"
        df1.loc[(df1['U_upper_value']<user_requirements['Uabs']) | (df1['U_upper_relative']<user_requirements['Urel']),'U_conformity'] = "True"
        print(validation_dict['name'], df1['U_conformity'].value_counts())

        #correct S ci's until we fix it in methods
        df1['S_conformity'] = "Unknown"
        df1['S_upper_value'] = df1['S_upper']
        df1['S_lower_value'] = df1['S_lower']
        df1['S_upper'] = df1['S_upper'] - df1['S']
        df1['S_lower'] = df1['S'] - df1['S_lower']
        df1['S_ci'] = df1['S_upper'] - df1['S_lower']

        df1.loc[(df1['S_upper_value']<-user_requirements['Sabs']) ,'S_conformity'] = "False"
        df1.loc[(df1['S_lower_value']>user_requirements['Sabs']) ,'S_conformity'] = "False"
        df1.loc[(df1['S_upper_value']<user_requirements['Sabs'])  & (df1['S_lower_value']>-user_requirements['Sabs']) ,'S_conformity'] = "True"
        print(validation_dict['name'], df1['S_conformity'].value_counts())

        validation_dict['data'] = df1

    for validation_dict1 in validation_dict_list:



        #drop zero uncertainty estimates that imply FRM uncertainty was too large 
        df1 = validation_dict1['data']
        df1 = df1[(df1['U']>0) ]

        #recode uniformity
        df1['Uniformity'] = df1['Uniformity'].map({1:'Homog.',2:'Homog.',3:'Heterog.'})



        # #display results for each group
        ngroups = len(groups)
        colm = -1
        fig1,axs=plt.subplots(5,ngroups,figsize=(40,30),layout="constrained")
        sns.set_theme(
            style="white",
            rc={
                'font.size': 14,           # Global font size
                'axes.labelsize': 20,      # Font size for x and y labels
                'axes.titlesize': 16,      # Font size for plot title
                'xtick.labelsize': 20,     # Font size for x-tick labels
                'ytick.labelsize': 20,     # Font size for y-tick labels
                'legend.fontsize': 20,     # Font size for legend
                'legend.markerscale':1.5,
                'font.family': 'sans-serif'# Font family
            }
        )
        for group in groups:
            colm= colm + 1
            df1_group = df1[df1['NLCD_group']==group]
            # iterate thjrough all provided validation experiments
            if not df1_group.empty:

                #statistics for a single experiment with samples meting minimum degrees of freedom 
                df1_group_A = df1_group[(df1_group['A_N']>=minimum_df)]
                df1_group_B = df1_group[(df1_group['B_N']>=minimum_df)]
                df1_group_U = df1_group[(df1_group['U_N']>=minimum_df)]
                df1_group_S = df1_group[(df1_group['S_N']>=minimum_df)]

                #1:1 scatterplot, uncertainty and uniformity
                ax=axs[0,colm]
                sns.scatterplot(data=df1_group_U,x='x',y='y',ax=ax,style='U_conformity',markers= {"True":'o',"False":'s', "Unknown":'d'},style_order= {"True":'o',"False":'s', "Unknown":'d'},hue='U',size='U_N',sizes=(100,500),palette=sns.color_palette("viridis", as_cmap=True), zorder=2,legend=legend)
                sns.scatterplot(data=df1_group_U[ (df1_group_U['Uniformity']==3)],x='x',y='y',ax=ax,marker='o',fc="none",  ec='red',size='U_N', sizes=(100,500), zorder=3,legend=False)               
                
                #legend and axis lavels
                # --- FIX 1: FORCE RECTANGULAR PLOT (HEIGHT:WIDTH = 1:2) ---
                # set_box_aspect takes a ratio of (height / width). 0.5 means height is half the width.
                ax.set_box_aspect(0.75)

                if ax.get_legend() :
                    # --- FIX 2: SHRINK AND CONTROL THE LEGEND ---
                    # We pass specific kwargs to sns.move_legend to shrink text, markers, and pad space
                    sns.move_legend(
                        ax, 
                        "upper left", 
                        bbox_to_anchor=(1.02, 1), 
                        fontsize='small',          # Shrinks text
                        title_fontsize='medium',   # Shrinks title text
                        labelspacing=0.4,          # Compresses row spacing
                        markerscale=0.6            # Prevents oversized marker indicators
                    )                     
                ax.set_xlabel(validation_dict1['name']+' '+variable)
                ax.set_ylabel("FRM "+variable) 

                # Set limits and 1:1 line
                if ( variable == 'LAI' ):
                    ax.set_xlim(0,5)
                else:
                    ax.set_xlim(0,1)    

                # Get current axis limits
                x_lim = ax.get_xlim()
                y_lim = ax.get_ylim()
                
                # Find the shared range for a perfect 1:1 diagonal
                limit = [min(x_lim[0], y_lim[0]), max(x_lim[1], y_lim[1])]
                
                # Plot and reset limits so the line doesn't expand the plot area
                ax.plot(limit, limit, color='grey', ls='--')
                ax.set_xlim(x_lim)
                ax.set_ylim(y_lim)   
                
                if ( variable == 'LAI' ):
                    ax.set_xlim(0,5)
                else:
                    ax.set_xlim(0,1)   
                    
                #Accuracy and ThielSen fit
               
                ax=axs[1,colm]


                # sns.scatterplot(data=df1_group,x='x',y='A',ax=ax,style='U_conformity',markers= {"True":'o',"False":'s', "Unknown":'d'},style_order= {"True":'o',"False":'s', "Unknown":'d'},hue='NLCD',size='A_N',sizes=(100,500), zorder=2,legend=legend)
                # sns.scatterplot(data=df1_group[df1_group['Uniformity']==3],x='x',y='A',ax=ax,marker='o',fc="none",  ec='red',size='A_N', sizes=(100,500), zorder=3,legend=False)
                sns.scatterplot(data=df1_group_A,x='x',y='A',ax=ax,style='Uniformity',markers= {"Homog.":'o',"Heterog.":'s'},\
                                        style_order= {"Homog.":'o',"Heterog.":'s'},hue='U_conformity',hue_order=["True","False","Unknown"],size='A_N',sizes=(100,500), zorder=2,legend=legend)

                # --- FIX 1: FORCE RECTANGULAR PLOT (HEIGHT:WIDTH = 1:2) ---
                # set_box_aspect takes a ratio of (height / width). 0.5 means height is half the width.
                ax.set_box_aspect(0.75)

                if ax.get_legend() :
                    # --- FIX 2: SHRINK AND CONTROL THE LEGEND ---
                    # We pass specific kwargs to sns.move_legend to shrink text, markers, and pad space
                    sns.move_legend(
                        ax, 
                        "upper left", 
                        bbox_to_anchor=(1.02, 1), 
                        fontsize='small',          # Shrinks text
                        title_fontsize='medium',   # Shrinks title text
                        labelspacing=0.4,          # Compresses row spacing
                        markerscale=0.6            # Prevents oversized marker indicators
                    )   
                ax.errorbar(df1_group_A['x'],df1_group_A['A'],yerr=(df1_group_A['A_lower'],df1_group_A['A_upper']),fmt='none',color='lightgrey',ecolor='grey',capsize=0,zorder=1,label='_nolegend_')    

                reg = thielsen.TheilSenRegressor(random_state=42).fit(chebvander(df1_group_A['x'].to_numpy(),4), df1_group_A['A'].to_numpy(), w=1/np.power(df1_group_A['A_ci'].to_numpy(),2))
                ax.plot(x_linspace,reg.predict( chebvander(x_linspace,4)),zorder=4)

                reg_A = thielsen.TheilSenRegressor(random_state=42).fit(chebvander(df1_group_A['x'].to_numpy(),4), np.abs(df1_group_A['r'].to_numpy()), w=1/np.power(df1_group_A['r_ci'].to_numpy(),2))
                ax.plot(x_linspace,reg_A.predict( chebvander(x_linspace,4)),color='grey',zorder=5)
   
                ax.set_xlabel(validation_dict1['name']+' '+variable)
                ax.set_ylabel('A '+validation_dict1['name']+variable)    
                
                 # Set limits and 0 line
                if ( variable == 'LAI' ):
                    ax.set_xlim(0,5)
                    ax.set_ylim(0,2)
                else:
                    ax.set_xlim(0,1)
                    ax.set_ylim(0,0.2)
        
                # Get current axis limits
                x_lim = ax.get_xlim()
                y_lim = ax.get_ylim()
                
                #plot 0 line
                x_lim = ax.get_xlim()
                ax.plot(x_lim, [0,0], color='grey', ls='--')
                    
                # Plot and reset limits so the line doesn't expand the plot area

                ax.set_xlim(x_lim)
                ax.set_ylim(y_lim)

                
                #Bias and ThielSen fit
                
                ax=axs[2,colm]
                # sns.scatterplot(data=df1_group,x='x',y='B',ax=ax,style='U_conformity',markers= {"True":'o',"False":'s', "Unknown":'d'},style_order= {"True":'o',"False":'s', "Unknown":'d'},hue='NLCD',size='B_N',sizes=(100,500), zorder=2,legend=legend)
                # sns.scatterplot(data=df1_group[df1_group['Uniformity']==3],x='x',y='B',ax=ax,marker='o',fc="none",  ec='red',size='B_N', sizes=(100,500), zorder=3,legend=False)
                sns.scatterplot(data=df1_group_B,x='x',y='B',ax=ax,style='Uniformity',markers= {"Homog.":'o',"Heterog.":'s'},\
                                        style_order= {"Homog.":'o',"Heterog.":'s'},hue='S_conformity',hue_order=["True","False","Unknown"],size='B_N',sizes=(100,500), zorder=2,legend=legend)

 
                # --- FIX 1: FORCE RECTANGULAR PLOT (HEIGHT:WIDTH = 1:2) ---
                # set_box_aspect takes a ratio of (height / width). 0.5 means height is half the width.
                ax.set_box_aspect(0.75)

                if ax.get_legend() :
                    # --- FIX 2: SHRINK AND CONTROL THE LEGEND ---
                    # We pass specific kwargs to sns.move_legend to shrink text, markers, and pad space
                    sns.move_legend(
                        ax, 
                        "upper left", 
                        bbox_to_anchor=(1.02, 1), 
                        fontsize='small',          # Shrinks text
                        title_fontsize='medium',   # Shrinks title text
                        labelspacing=0.4,          # Compresses row spacing
                        markerscale=0.6            # Prevents oversized marker indicators
                    )   
                ax.errorbar(df1_group_B['x'],df1_group_B['B'],yerr=(df1_group_B['B_lower'],df1_group_B['B_upper']),fmt='none',color='lightgrey',ecolor='grey',capsize=0,zorder=1,label='_nolegend_')    
 
                reg = thielsen.TheilSenRegressor(random_state=42).fit(chebvander(df1_group_B['x'].to_numpy(),4), df1_group_B['B'].to_numpy(), w=1/np.power(df1_group_B['B_ci'].to_numpy(),2))
                ax.plot(x_linspace,reg.predict( chebvander(x_linspace,4)),zorder=4)

                reg_B = thielsen.TheilSenRegressor(random_state=42).fit(chebvander(df1_group_B['x'].to_numpy(),4), df1_group_B['r'].to_numpy(), w=1/np.power(df1_group_B['r_ci'].to_numpy(),2))
                ax.plot(x_linspace,reg_B.predict( chebvander(x_linspace,4)),color='grey',zorder=5)

                ax.set_xlabel(validation_dict1['name']+' '+variable)
                ax.set_ylabel('B '+validation_dict1['name']+variable)    
                
                 # Set limits and 0 line
                if ( variable == 'LAI' ):
                    ax.set_xlim(0,5)
                    ax.set_ylim(-3,2)
                else:
                    ax.set_xlim(0,1)
                    ax.set_ylim(-0.5,0.5)
        
                # Get current axis limits
                x_lim = ax.get_xlim()
                y_lim = ax.get_ylim()
                
                #plot 0 line
                x_lim = ax.get_xlim()
                ax.plot(x_lim, [0,0], color='grey', ls='--')
                    
                # Plot and reset limits so the line doesn't expand the plot area

                ax.set_xlim(x_lim)
                ax.set_ylim(y_lim)

                #Uncertainty and Thiel Sen fit
                ax=axs[3,colm]
    
                # sns.scatterplot(data=df1_group,x='x',y='U',ax=ax,style='U_conformity',markers= {"True":'o',"False":'s', "Unknown":'d'},style_order= {"True":'o',"False":'s', "Unknown":'d'},hue='NLCD',size='U_N',sizes=(100,500), zorder=2,legend=legend)
                # sns.scatterplot(data=df1_group[df1_group['Uniformity']==3],x='x',y='U',ax=ax,marker='o',fc="none",  ec='red',size='U_N', sizes=(100,500), zorder=3,legend=False)
                sns.scatterplot(data=df1_group,x='x',y='U',ax=ax,style='Uniformity',markers= {"Homog.":'o',"Heterog.":'s'},\
                                        style_order= {"Homog.":'o',"Heterog.":'s'},hue='U_conformity',hue_order=["True","False","Unknown"],size='U_N',sizes=(100,500), zorder=2,legend=legend)

 
                ax.plot(x_linspace,np.maximum(user_requirements['Urel']*x_linspace,user_requirements['Uabs']),':',zorder=4)

                # --- FIX 1: FORCE RECTANGULAR PLOT (HEIGHT:WIDTH = 1:2) ---
                # set_box_aspect takes a ratio of (height / width). 0.5 means height is half the width.
                ax.set_box_aspect(0.75)

                if ax.get_legend() :
                    # --- FIX 2: SHRINK AND CONTROL THE LEGEND ---
                    # We pass specific kwargs to sns.move_legend to shrink text, markers, and pad space
                    sns.move_legend(
                        ax, 
                        "upper left", 
                        bbox_to_anchor=(1.02, 1), 
                        fontsize='small',          # Shrinks text
                        title_fontsize='medium',   # Shrinks title text
                        labelspacing=0.4,          # Compresses row spacing
                        markerscale=0.6            # Prevents oversized marker indicators
                    )         
                ax.errorbar(df1_group_U['x'],df1_group_U['U'],yerr=(df1_group_U['U_lower'],df1_group_U['U_upper']),fmt='none',color='lightgrey',ecolor='grey',capsize=0,zorder=1,label='_nolegend_')   

                reg = thielsen.TheilSenRegressor(random_state=42).fit(chebvander(df1_group_U['x'].to_numpy(),4), df1_group_U['U'].to_numpy(), w=1/np.power(df1_group_U['U_ci'].to_numpy(),2))   
                ax.plot(x_linspace,reg.predict( chebvander(x_linspace,4)),zorder=4)


                reg_U = thielsen.TheilSenRegressor(random_state=42).fit(chebvander(df1_group_U['x'].to_numpy(),4), \
                                             np.abs(df1_group_U['r'].to_numpy() - reg_B.predict( chebvander(df1_group_U['x'].to_numpy(),4))) , w=1/np.power(df1_group_U['r_ci'].to_numpy(),2))
                ax.plot(x_linspace,reg_U.predict( chebvander(x_linspace,4)),color='grey',zorder=5)

                ax.set_xlabel(validation_dict1['name']+' '+variable)
                ax.set_ylabel('U '+validation_dict1['name']+variable)    
                
                 # Set limits and 0 line
                if ( variable == 'LAI' ):
                    ax.set_xlim(0,5)
                    ax.set_ylim(0,1)
                else:
                    ax.set_xlim(0,1)
                    ax.set_ylim(0,0.2)
    
        
                # Get current axis limits
                x_lim = ax.get_xlim()
                y_lim = ax.get_ylim()
                
                #plot 0 line
                x_lim = ax.get_xlim()
                ax.plot(x_lim, [0,0], color='grey', ls='--')
                    
                # Plot and reset limits so the line doesn't expand the plot area')
                ax.set_xlim(x_lim)
                ax.set_ylim(y_lim)


                #Stability and Thiel Sen fit
                ax=axs[4,colm]

                #remove nan from stability df
                df1_group_S = df1_group.dropna(subset=['S'])
                if not df1_group_S.empty:
                    reg = thielsen.TheilSenRegressor(random_state=42).fit(chebvander(df1_group_S['x'].to_numpy(),4), df1_group_S['S'].to_numpy(), w=1/np.power(df1_group_S['S_ci'].to_numpy(),2))

                    # sns.scatterplot(data=df1_group_S,x='x',y='S',ax=ax,style='S_conformity',markers= {"True":'o',"False":'s', "Unknown":'d'},style_order= {"True":'o',"False":'s', "Unknown":'d'},hue='NLCD',size='S_N',sizes=(100,500), zorder=2,legend=legend)
                    # sns.scatterplot(data=df1_group_S[df1_group_S['Uniformity']==3],x='x',y='S',ax=ax,marker='o',fc="none",  ec='red',size='S_N', sizes=(100,500), zorder=3,legend=False)
                    sns.scatterplot(data=df1_group_S,x='x',y='S',ax=ax,style='Uniformity',markers= {"Homog.":'o',"Heterog.":'s'},\
                                        style_order= {"Homog.":'o',"Heterog.":'s'},hue='S_conformity',hue_order=["True","False","Unknown"],size='S_N',sizes=(100,500), zorder=2,legend=legend)

 
 
                    ax.plot(x_linspace,user_requirements['Sabs']+0*x_linspace,':',zorder=4)
                    ax.plot(x_linspace,-user_requirements['Sabs']+0*x_linspace,':',zorder=4)

                    # --- FIX 1: FORCE RECTANGULAR PLOT (HEIGHT:WIDTH = 1:2) ---
                    # set_box_aspect takes a ratio of (height / width). 0.5 means height is half the width.
                    ax.set_box_aspect(0.75)

                    if ax.get_legend() :
                        # --- FIX 2: SHRINK AND CONTROL THE LEGEND ---
                        # We pass specific kwargs to sns.move_legend to shrink text, markers, and pad space
                        sns.move_legend(
                            ax, 
                            "upper left", 
                            bbox_to_anchor=(1.02, 1), 
                            fontsize='small',          # Shrinks text
                            title_fontsize='medium',   # Shrinks title text
                            labelspacing=0.4,          # Compresses row spacing
                            markerscale=0.6            # Prevents oversized marker indicators
                        )   
                    ax.errorbar(df1_group_S['x'],df1_group_S['S'],yerr=(df1_group_S['S_lower'],df1_group_S['S_upper']),fmt='none',color='lightgrey',ecolor='grey',capsize=0,zorder=1,label='_nolegend_')  
        
                    ax.plot(x_linspace,reg.predict( chebvander(x_linspace,4)),zorder=4)

                    ax.set_xlabel(validation_dict1['name']+' '+variable)
                    ax.set_ylabel('S '+validation_dict1['name']+variable)    
                    
                    # Set limits and 0 line
                    if ( variable == 'LAI' ):
                        ax.set_xlim(0,5)
                        ax.set_ylim(-0.4,0.4)
                    else:
                        ax.set_xlim(0,1)
                        ax.set_ylim(-0.1,0.1)
        
            
                    # Get current axis limits
                    x_lim = ax.get_xlim()
                    y_lim = ax.get_ylim()
                    
                    #plot 0 line
                    x_lim = ax.get_xlim()
                    ax.plot(x_lim, [0,0], color='grey', ls='--')
                        
                    # Plot and reset limits so the line doesn't expand the plot area')
                    ax.set_xlim(x_lim)
                    ax.set_ylim(y_lim)

        #show the univariate plots
        plt.tight_layout()
        # 2. Save the plot to a file BEFORE calling plt.show()
        plt.savefig('D:/CEOS/OutputData/plots_'+validation_dict1['name']+variable+'.png', dpi=300, bbox_inches='tight')
        plt.show()

        if (bivariate==True):
            for validation_dict2 in validation_dict_list:

                if (validation_dict2['name'] != validation_dict1['name']) :


                    df2 = validation_dict2['data']
            
                    #drop zero uncertainty estimates that imply FRM uncertainty was too large
                    df2 = df2[df2['U']>0]



                    #biivariate visualization
                    if ( not df1.equals(df2) ):
                        
                        

                        fig2,axs = plt.subplots(4,ngroups,figsize=(40,40),layout="constrained")

                        colm = -1
                        for group in groups:
                            colm = colm+1


                            df1_group = df1[df1['NLCD_group']==group]   
                            df2_group = df2[df2['NLCD_group']==group]      



                            df1_group_A = df1_group[(df1_group['A_N']>=minimum_df)]   
                            df1_group_B = df1_group[(df1_group['B_N']>=minimum_df)]   
                            df1_group_U = df1_group[(df1_group['U_N']>=minimum_df)]   
                            df1_group_S = df1_group[(df1_group['S_N']>=minimum_df)]   


                            df2_group_A = df2_group[(df2_group['A_N']>=minimum_df)]   
                            df2_group_B = df2_group[(df2_group['B_N']>=minimum_df)]   
                            df2_group_U = df2_group[(df2_group['U_N']>=minimum_df)]   
                            df2_group_S = df2_group[(df2_group['S_N']>=minimum_df)]   

                            df12_group_A = pd.merge(df1_group_A,df2_group_A,on=['Plot','Date'],suffixes=['_df1','_df2'])
                            df12_group_B = pd.merge(df1_group_B,df2_group_B,on=['Plot','Date'],suffixes=['_df1','_df2'])
                            df12_group_U = pd.merge(df1_group_U,df2_group_U,on=['Plot','Date'],suffixes=['_df1','_df2'])
                            df12_group_S = pd.merge(df1_group_S,df2_group_S,on=['Plot','Date'],suffixes=['_df1','_df2'])
                            

                            if not df12_group_A.empty:
            
                                

                                #1:1 plots of accuracy
                                ax = axs[0,colm]
                                df12_group_A['U_conformity']=df12_group_A['U_conformity_df1']+df12_group_A['U_conformity_df2']
                                sns.scatterplot(
                                    data=df12_group_A[df12_group_A['U_conformity_df1']==df12_group_A['U_conformity_df2']],
                                    x='A_df1',
                                    y='A_df2',
                                    ax=ax,
                                    style='U_conformity',
                                    markers= {"TrueTrue":'o',"FalseFalse":'s', "UnknownUnknown":'d',"TrueFalse":'X',"FalseTrue":'P',
                                                 "UnknownTrue":'^',"UnknownFalse":'v',"TrueUnknown":'<', "FalseUnknown":'>'},
                                    style_order= ["TrueTrue","FalseFalse", "UnknownUnknown","TrueFalse","FalseTrue",
                                                 "UnknownTrue","UnknownFalse","TrueUnknown", "FalseUnknown"],
                                    hue='NLCD_df1',
                                    size='A_N_df1',
                                    sizes=(100,500),
                                    zorder=2,
                                    legend=legend)
                                # sns.scatterplot(df12_group_A[df12_group_A['U_conformity_df1']!=df12_group_A['U_conformity_df2']],x='A_df1',y='A_df2',ax=ax,style='U_conformity_df1',markers= {"True":'o',"False":'s', "Unknown":'d'},style_order= {"True":o',"False":'s', "Unknown":'d'},hue='NLCD_df1',facecolors="none", size='A_N_df1',sizes=(100,500), zorder=2,legend=False)

                                ax.set_box_aspect(1)

                                if ax.get_legend() :
                                    # --- FIX 2: SHRINK AND CONTROL THE LEGEND ---
                                    # We pass specific kwargs to sns.move_legend to shrink text, markers, and pad space
                                    sns.move_legend(
                                        ax, 
                                        "upper left", 
                                        bbox_to_anchor=(1.02, 1), 
                                        fontsize='small',          # Shrinks text
                                        title_fontsize='medium',   # Shrinks title text
                                        labelspacing=0.4,          # Compresses row spacing
                                        markerscale=0.6            # Prevents oversized marker indicators
                                    )   
                                ax.errorbar(df12_group_A['A_df1'],df12_group_A['A_df2'],xerr=(df12_group_A['A_lower_df1'],df12_group_A['A_upper_df1']),yerr=(df12_group_A['A_lower_df2'],df12_group_A['A_upper_df2']),fmt='none',color='lightgrey',ecolor='grey',capsize=0,zorder=1)  
                
                                ax.set_xlabel('A '+validation_dict1['name']+variable)
                                ax.set_ylabel('A '+validation_dict2['name']+variable)   
                                
                                # Set limits and 1:1 line
                                if ( variable == 'LAI' ):
                                    ax.set_xlim(0,3)
                                    ax.set_ylim(0,3)
                                else:
                                    ax.set_xlim(0,.3)
                                    ax.set_ylim(0,.3)
                        
                                # Get current axis limits
                                x_lim = ax.get_xlim()
                                y_lim = ax.get_ylim()
                                
                                # Find the shared range for a perfect 1:1 diagonal
                                limit = [min(x_lim[0], y_lim[0]), max(x_lim[1], y_lim[1])]
                                
                                # Plot and reset limits so the line doesn't expand the plot area
                                ax.plot(limit, limit, color='grey', ls='--')
                                ax.set_xlim(x_lim)
                                ax.set_ylim(y_lim)

                                # Set limits and 1:1 line
                                if ( variable == 'LAI' ):
                                    ax.set_xlim(0,3)
                                    ax.set_ylim(0,3)
                                else:
                                    ax.set_xlim(0,.3)
                                    ax.set_ylim(0,.3)
                        
                                # Get current axis limits
                                x_lim = ax.get_xlim()
                                y_lim = ax.get_ylim()
                                
                                # Find the shared range for a perfect 1:1 diagonal
                                limit = [min(x_lim[0], y_lim[0]), max(x_lim[1], y_lim[1])]
                                
                                # Plot and reset limits so the line doesn't expand the plot area
                                ax.plot(limit, limit, color='grey', ls='--')
                                ax.set_xlim(x_lim)
                                ax.set_ylim(y_lim)


                                # Set limits and 1:1 line
                                if ( variable == 'LAI' ):
                                    ax.set_xlim(0,3)
                                    ax.set_ylim(0,3)
                                else:
                                    ax.set_xlim(0,.3)
                                    ax.set_ylim(0,.3)
                        
                                # Get current axis limits
                                x_lim = ax.get_xlim()
                                y_lim = ax.get_ylim()
                                
                                # Find the shared range for a perfect 1:1 diagonal
                                limit = [min(x_lim[0], y_lim[0]), max(x_lim[1], y_lim[1])]
                                
                                # Plot and reset limits so the line doesn't expand the plot area
                                ax.plot(limit, limit, color='grey', ls='--')
                                ax.set_xlim(x_lim)
                                ax.set_ylim(y_lim)


                            if not df12_group_B.empty:

                                #1:1 plots of bias
                                ax = axs[1,colm]
                                df12_group_B['S_conformity']=df12_group_B['S_conformity_df1']+df12_group_B['S_conformity_df2']
                                sns.scatterplot(
                                    data=df12_group_B[df12_group_B['S_conformity_df1']==df12_group_B['S_conformity_df2']],
                                    x='B_df1',
                                    y='B_df2',
                                    ax=ax,
                                    style='S_conformity',
                                    markers= {"TrueTrue":'o',"FalseFalse":'s', "UnknownUnknown":'d',"TrueFalse":'X',"FalseTrue":'P',
                                                 "UnknownTrue":'^',"UnknownFalse":'v',"TrueUnknown":'<', "FalseUnknown":'>'},
                                    style_order= ["TrueTrue","FalseFalse", "UnknownUnknown","TrueFalse","FalseTrue",
                                                 "UnknownTrue","UnknownFalse","TrueUnknown", "FalseUnknown"],
                                    hue='NLCD_df1',
                                    size='B_N_df1',
                                    sizes=(100,500),
                                    zorder=2,
                                    legend=legend)
                                ax.set_box_aspect(1)

                                if ax.get_legend() :
                                    # --- FIX 2: SHRINK AND CONTROL THE LEGEND ---
                                    # We pass specific kwargs to sns.move_legend to shrink text, markers, and pad space
                                    sns.move_legend(
                                        ax, 
                                        "upper left", 
                                        bbox_to_anchor=(1.02, 1), 
                                        fontsize='small',          # Shrinks text
                                        title_fontsize='medium',   # Shrinks title text
                                        labelspacing=0.4,          # Compresses row spacing
                                        markerscale=0.6            # Prevents oversized marker indicators
                                    )   
                                ax.errorbar(df12_group_B['B_df1'],df12_group_B['B_df2'],xerr=(df12_group_B['B_lower_df1'],df12_group_B['B_upper_df1']),yerr=(df12_group_B['B_lower_df2'],df12_group_B['B_upper_df2']),fmt='none',color='lightgrey',ecolor='grey',capsize=0,zorder=1)  
                
                                ax.set_xlabel('B '+validation_dict1['name']+variable)
                                ax.set_ylabel('B '+validation_dict2['name']+variable)   
                                
                            # Set limits and 1:1 line
                                if ( variable == 'LAI' ):
                                    ax.set_xlim(-3,3)
                                    ax.set_ylim(-0.3,0.3)
                                else:
                                    ax.set_xlim(-3,3)
                                    ax.set_ylim(-0.3,0.3)
                        
                                # Get current axis limits
                                x_lim = ax.get_xlim()
                                y_lim = ax.get_ylim()
                                
                                # Find the shared range for a perfect 1:1 diagonal
                                limit = [min(x_lim[0], y_lim[0]), max(x_lim[1], y_lim[1])]
                                
                                # Plot and reset limits so the line doesn't expand the plot area
                                ax.plot(limit, limit, color='grey', ls='--')
                                ax.set_xlim(x_lim)
                                ax.set_ylim(y_lim)

                                # Set limits and 1:1 line
                                if ( variable == 'LAI' ):
                                    ax.set_xlim(-3,3)
                                    ax.set_ylim(-0.3,0.3)
                                else:
                                    ax.set_xlim(-3,3)
                                    ax.set_ylim(-0.3,0.3)
                        
                                # Get current axis limits
                                x_lim = ax.get_xlim()
                                y_lim = ax.get_ylim()
                                
                                # Find the shared range for a perfect 1:1 diagonal
                                limit = [min(x_lim[0], y_lim[0]), max(x_lim[1], y_lim[1])]
                                
                                # Plot and reset limits so the line doesn't expand the plot area
                                ax.plot(limit, limit, color='grey', ls='--')
                                ax.set_xlim(x_lim)
                                ax.set_ylim(y_lim)


                                # Set limits and 1:1 line
                                if ( variable == 'LAI' ):
                                    ax.set_xlim(-3,3)
                                    ax.set_ylim(-3,3)
                                else:
                                    ax.set_xlim(-.3,.3)
                                    ax.set_ylim(-.3,.3)
                        
                                # Get current axis limits
                                x_lim = ax.get_xlim()
                                y_lim = ax.get_ylim()
                                
                                # Find the shared range for a perfect 1:1 diagonal
                                limit = [min(x_lim[0], y_lim[0]), max(x_lim[1], y_lim[1])]
                                
                                # Plot and reset limits so the line doesn't expand the plot area
                                ax.plot(limit, limit, color='grey', ls='--')
                                ax.set_xlim(x_lim)
                                ax.set_ylim(y_lim)

                            if not df12_group_U.empty:

                                #1:1 plots of uncertainty
                                ax = axs[2,colm]
                                df12_group_U['U_conformity']=df12_group_U['U_conformity_df1']+df12_group_U['U_conformity_df2']
                                sns.scatterplot(
                                    data=df12_group_U[df12_group_U['U_conformity_df1']==df12_group_U['U_conformity_df2']],
                                    x='U_df1',
                                    y='U_df2',
                                    ax=ax,
                                    style='U_conformity',
                                    markers= {"TrueTrue":'o',"FalseFalse":'s', "UnknownUnknown":'d',"TrueFalse":'X',"FalseTrue":'P',
                                                 "UnknownTrue":'^',"UnknownFalse":'v',"TrueUnknown":'<', "FalseUnknown":'>'},
                                    style_order= ["TrueTrue","FalseFalse", "UnknownUnknown","TrueFalse","FalseTrue",
                                                 "UnknownTrue","UnknownFalse","TrueUnknown", "FalseUnknown"],
                                    hue='NLCD_df1',
                                    size='U_N_df1',
                                    sizes=(100,500),
                                    zorder=2,
                                    legend=legend)
                                ax.set_box_aspect(1)

                                if ax.get_legend() :
                                    # --- FIX 2: SHRINK AND CONTROL THE LEGEND ---
                                    # We pass specific kwargs to sns.move_legend to shrink text, markers, and pad space
                                    sns.move_legend(
                                        ax, 
                                        "upper left", 
                                        bbox_to_anchor=(1.02, 1), 
                                        fontsize='small',          # Shrinks text
                                        title_fontsize='medium',   # Shrinks title text
                                        labelspacing=0.4,          # Compresses row spacing
                                        markerscale=0.6            # Prevents oversized marker indicators
                                    )   
                                ax.errorbar(df12_group_U['U_df1'],df12_group_U['U_df2'],xerr=(df12_group_U['U_lower_df1'],df12_group_U['U_upper_df1']),yerr=(df12_group_U['U_lower_df2'],df12_group_U['U_upper_df2']),fmt='none',color='lightgrey',ecolor='grey',capsize=0,zorder=1)  
                
                                ax.set_xlabel('U '+validation_dict1['name']+variable)
                                ax.set_ylabel('U '+validation_dict2['name']+variable)   
                                
                                # Set limits and 1:1 line
                                if ( variable == 'LAI' ):
                                    ax.set_xlim(0,1)
                                    ax.set_ylim(0,1)
                                else:
                                    ax.set_xlim(0,0.1)
                                    ax.set_ylim(0,0.1)
                        
                                # Get current axis limits
                                x_lim = ax.get_xlim()
                                y_lim = ax.get_ylim()
                                
                                # Find the shared range for a perfect 1:1 diagonal
                                limit = [min(x_lim[0], y_lim[0]), max(x_lim[1], y_lim[1])]
                                
                                # Plot and reset limits so the line doesn't expand the plot area
                                ax.plot(limit, limit, color='grey', ls='--')
                                ax.set_xlim(x_lim)
                                ax.set_ylim(y_lim)
                                
                                # Set limits and 1:1 line
                                if ( variable == 'LAI' ):
                                    ax.set_xlim(0,1)
                                    ax.set_ylim(0,1)
                                else:
                                    ax.set_xlim(0,0.1)
                                    ax.set_ylim(0,0.1)
                        
                                # Get current axis limits
                                x_lim = ax.get_xlim()
                                y_lim = ax.get_ylim()
                                
                                # Find the shared range for a perfect 1:1 diagonal
                                limit = [min(x_lim[0], y_lim[0]), max(x_lim[1], y_lim[1])]
                                
                                # Plot and reset limits so the line doesn't expand the plot area
                                ax.plot(limit, limit, color='grey', ls='--')
                                ax.set_xlim(x_lim)
                                ax.set_ylim(y_lim)

                            if not df12_group_S.empty:

                                #1:1 plots of stability
                                ax = axs[3,colm]
                                df12_group_S['S_conformity']=df12_group_S['S_conformity_df1']+df12_group_S['S_conformity_df2']
                                sns.scatterplot(
                                    data=df12_group_S[df12_group_S['S_conformity_df1']==df12_group_S['S_conformity_df2']],
                                    x='S_df1',
                                    y='S_df2',
                                    ax=ax,
                                    style='S_conformity',
                                    markers= {"TrueTrue":'o',"FalseFalse":'s', "UnknownUnknown":'d',"TrueFalse":'X',"FalseTrue":'P',
                                                 "UnknownTrue":'^',"UnknownFalse":'v',"TrueUnknown":'<', "FalseUnknown":'>'},
                                    style_order= ["TrueTrue","FalseFalse", "UnknownUnknown","TrueFalse","FalseTrue",
                                                 "UnknownTrue","UnknownFalse","TrueUnknown", "FalseUnknown"],
                                    hue='NLCD_df1',
                                    size='S_N_df1',
                                    sizes=(100,500),
                                    zorder=2,
                                    legend=legend)
                                ax.set_box_aspect(1)

                                if ax.get_legend() :
                                    # --- FIX 2: SHRINK AND CONTROL THE LEGEND ---
                                    # We pass specific kwargs to sns.move_legend to shrink text, markers, and pad space
                                    sns.move_legend(
                                        ax, 
                                        "upper left", 
                                        bbox_to_anchor=(1.02, 1), 
                                        fontsize='small',          # Shrinks text
                                        title_fontsize='medium',   # Shrinks title text
                                        labelspacing=0.4,          # Compresses row spacing
                                        markerscale=0.6            # Prevents oversized marker indicators
                                    )   
                                ax.errorbar(df12_group_S['S_df1'],df12_group_S['S_df2'],xerr=(df12_group_S['S_lower_df1'],df12_group_S['S_upper_df1']),yerr=(df12_group_S['S_lower_df2'],df12_group_S['S_upper_df2']),fmt='none',color='lightgrey',ecolor='grey',capsize=0,zorder=1)  
                
                                ax.set_xlabel('S '+validation_dict1['name']+variable)
                                ax.set_ylabel('S '+validation_dict2['name']+variable)   
                                
                                # Set limits and 1:1 line
                                if ( variable == 'LAI' ):
                                    ax.set_xlim(-0.4,0.4)
                                    ax.set_ylim(-0.4,0.4)

                                else:
                                    ax.set_xlim(-0.05,0.05)
                                    ax.set_ylim(-0.05,0.05)
                        
                                # Get current axis limits
                                x_lim = ax.get_xlim()
                                y_lim = ax.get_ylim()
                                
                                # Find the shared range for a perfect 1:1 diagonal
                                limit = [min(x_lim[0], y_lim[0]), max(x_lim[1], y_lim[1])]
                                
                                # Plot and reset limits so the line doesn't expand the plot area
                                ax.plot(limit, limit, color='grey', ls='--')
                                ax.set_xlim(x_lim)
                                ax.set_ylim(y_lim)

                                # Set limits and 1:1 line
                                if ( variable == 'LAI' ):
                                    ax.set_xlim(-0.4,0.4)
                                    ax.set_ylim(-0.4,0.4)

                                else:
                                    ax.set_xlim(-0.05,0.05)
                                    ax.set_ylim(-0.05,0.05)

                            # #show the univariate plots
                            # plt.tight_layout(w_pad=0.5)
                            # 2. Save the plot to a file BEFORE calling plt.show()
                            plt.savefig('D:/CEOS/OutputData/plot_'+validation_dict1['name']+'_'+validation_dict2['name']+variable+'.png', dpi=300, bbox_inches='tight')
                        plt.show()  

                        fig3,axs = plt.subplots(4,ngroups,figsize=(40,40),layout="constrained")

                        colm = -1
                        for group in groups: 
                            colm = colm+1


                            df1_group = df1[df1['NLCD_group']==group]      
                            df1_group_A = df1_group[(df1_group['A_N']>=minimum_df)]   
                            df1_group_B = df1_group[(df1_group['B_N']>=minimum_df)]   
                            df1_group_U = df1_group[(df1_group['U_N']>=minimum_df)]   
                            df1_group_S = df1_group[(df1_group['S_N']>=minimum_df)]   


                            df2_group = df2[df2['NLCD_group']==group]      
                            df2_group_A = df2_group[(df2_group['A_N']>=minimum_df)]   
                            df2_group_B = df2_group[(df2_group['B_N']>=minimum_df)]   
                            df2_group_U = df2_group[(df2_group['U_N']>=minimum_df)]   
                            df2_group_S = df2_group[(df2_group['S_N']>=minimum_df)] 



                            if not df12_group_A.empty:

                                # plots of accuracy conditional on variable estimate
                                ax = axs[0,colm]
                                #A
                                reg = thielsen.TheilSenRegressor(random_state=42).fit(chebvander(df1_group_A['x'].to_numpy(),4), df1_group_A['A'].to_numpy(), w=1/np.power(df1_group_A['A_ci'].to_numpy(),2))
                                ax.plot(x_linspace,reg.predict( chebvander(x_linspace,4)),zorder=4, c='r',label=validation_dict1['name'])                 
                                ax.plot(df1_group_A['x'].to_numpy(), df1_group_A['A'].to_numpy(),'r.')
                                reg = thielsen.TheilSenRegressor(random_state=42).fit(chebvander(df2_group_A['x'].to_numpy(),4), df2_group_A['A'].to_numpy(), w=1/np.power(df2_group_A['A_ci'].to_numpy(),2))
                                ax.plot(x_linspace,reg.predict( chebvander(x_linspace,4)),zorder=4, c='b',label=validation_dict2['name'])     
                                ax.plot(df2_group_A['x'].to_numpy(), df2_group_A['A'].to_numpy(),'b.')


                                if legend:
                                    plt.legend()
                                    
                                ax.set_xlabel(validation_dict2['name']+variable)
                                ax.set_ylabel("Conditional A "+variable)   
                                
                                # Set limits and 0 line
                                if ( variable == 'LAI' ):
                                    ax.set_xlim(0,5)
                                    ax.set_ylim(0,3)
                                else:
                                    ax.set_xlim(0,1)
                                    ax.set_ylim(0,0.5)
                        
                                # Get current axis limits
                                x_lim = ax.get_xlim()
                                y_lim = ax.get_ylim()
                                
                                #plot 0 line
                                x_lim = ax.get_xlim()
                                ax.plot(x_lim, [0,0], color='grey', ls='--')
                                    
                                # Plot and reset limits so the line doesn't expand the plot area
                                ax.set_xlim(x_lim)
                                ax.set_ylim(y_lim)

                            if not df12_group_B.empty:

                                # plots of bias conditional on variable estimate
                                ax = axs[1,colm]
                                #U
                                reg = thielsen.TheilSenRegressor(random_state=42).fit(chebvander(df1_group_B['x'].to_numpy(),4), df1_group_B['B'].to_numpy(), w=1/np.power(df1_group_B['B_ci'].to_numpy(),2))
                                ax.plot(x_linspace,reg.predict( chebvander(x_linspace,4)),zorder=4,c='r', label=validation_dict1['name'])               
                                ax.plot(df1_group_A['x'].to_numpy(), df1_group_A['B'].to_numpy(),'r.')

            
                                reg = thielsen.TheilSenRegressor(random_state=42).fit(chebvander(df2_group_B['x'].to_numpy(),4), df2_group_B['B'].to_numpy(), w=1/np.power(df2_group_B['B_ci'].to_numpy(),2))
                                ax.plot(x_linspace,reg.predict( chebvander(x_linspace,4)),zorder=4,c= 'b',label=validation_dict2['name'])                  
                                ax.plot(df2_group_A['x'].to_numpy(), df2_group_A['B'].to_numpy(),'b.')

                                if legend:
                                    plt.legend()
                                    
                                ax.set_xlabel(validation_dict2['name']+variable)
                                ax.set_ylabel("Conditional B "+variable)   
                                
                                # Set limits and 0 line
                                if ( variable == 'LAI' ):
                                    ax.set_xlim(0,5)
                                    ax.set_ylim(-3,3)
                                else:
                                    ax.set_xlim(0,1)
                                    ax.set_ylim(-0.3,0.3)
                        
                                # Get current axis limits
                                x_lim = ax.get_xlim()
                                y_lim = ax.get_ylim()
                                
                                #plot 0 line
                                x_lim = ax.get_xlim()
                                ax.plot(x_lim, [0,0], color='grey', ls='--')
                                    
                                # Plot and reset limits so the line doesn't expand the plot area
                                ax.set_xlim(x_lim)
                                ax.set_ylim(y_lim)

                            if not df12_group_U.empty:

                                # plots of uncertainty conditional on variable estimate
                                ax = axs[2,colm]
                                #U
                                reg = thielsen.TheilSenRegressor(random_state=42).fit(chebvander(df1_group_U['x'].to_numpy(),4), df1_group_U['U'].to_numpy(), w=1/np.power(df1_group_U['U_ci'].to_numpy(),2))
                                ax.plot(x_linspace,reg.predict( chebvander(x_linspace,4)),zorder=4, c='r', label=validation_dict1['name'])                               
                                ax.plot(df1_group_A['x'].to_numpy(), df1_group_A['U'].to_numpy(),'r.')

            
                                reg = thielsen.TheilSenRegressor(random_state=42).fit(chebvander(df2_group_U['x'].to_numpy(),4), df2_group_U['U'].to_numpy(), w=1/np.power(df2_group_U['U_ci'].to_numpy(),2))
                                ax.plot(x_linspace,reg.predict( chebvander(x_linspace,4)),zorder=4, c='b', label=validation_dict2['name'])     
                                ax.plot(df1_group_A['x'].to_numpy(), df1_group_A['U'].to_numpy(),'b.')

                                if legend:
                                    plt.legend()
                                    
                                ax.set_xlabel(validation_dict2['name']+variable)
                                ax.set_ylabel("Conditional U "+variable)   
                                
                                # Set limits and 0 line
                                if ( variable == 'LAI' ):
                                    ax.set_xlim(0,5)
                                    ax.set_ylim(0,3)
                                else:
                                    ax.set_xlim(0,1)
                                    ax.set_ylim(0,0.5)
                        
                                # Get current axis limits
                                x_lim = ax.get_xlim()
                                y_lim = ax.get_ylim()
                                
                                #plot 0 line
                                x_lim = ax.get_xlim()
                                ax.plot(x_lim, [0,0], color='grey', ls='--')
                                    
                                # Plot and reset limits so the line doesn't expand the plot area
                                ax.set_xlim(x_lim)
                                ax.set_ylim(y_lim)

                            if not df12_group_S.empty:

                            # plots of stability conditional on variable estimate
                                ax = axs[3,colm]
                                #U
                                df = df1_group_S.dropna(subset=['S','S_ci'])
                                reg = thielsen.TheilSenRegressor(random_state=42).fit(chebvander(df1_group_S['x'].to_numpy(),4), df1_group_S['S'].to_numpy(), w=1/np.power(df1_group_S['S_ci'].to_numpy(),2))
                                ax.plot(x_linspace,reg.predict( chebvander(x_linspace,4)),zorder=4,c='r', label=validation_dict1['name'])                  
                                ax.plot(df1_group_A['x'].to_numpy(), df1_group_A['S'].to_numpy(),'r.')

                                df = df2_group_S.dropna(subset=['S','S_ci'])
                                reg = thielsen.TheilSenRegressor(random_state=42).fit(chebvander(df2_group_S['x'].to_numpy(),4), df2_group_S['S'].to_numpy(), w=1/np.power(df2_group_S['S_ci'].to_numpy(),2))
                                ax.plot(x_linspace,reg.predict( chebvander(x_linspace,4)),zorder=4, c='b', label=validation_dict2['name'])     
                                ax.plot(df1_group_A['x'].to_numpy(), df1_group_A['S'].to_numpy(),'b.')

                                if legend:
                                    plt.legend()
                                    
                                ax.set_xlabel(validation_dict2['name']+variable)
                                ax.set_ylabel("Conditional S "+variable)   
                                
                                # Set limits and 0 line
                                if ( variable == 'LAI' ):
                                    ax.set_xlim(0,5)
                                    ax.set_ylim(-0.4,0.4)
                                else:
                                    ax.set_xlim(0,1)
                                    ax.set_ylim(-0.1,0.1)
                        
                                # Get current axis limits
                                x_lim = ax.get_xlim()
                                y_lim = ax.get_ylim()
                                
                                #plot 0 line
                                x_lim = ax.get_xlim()
                                ax.plot(x_lim, [0,0], color='grey', ls='--')
                                    
                                # Plot and reset limits so the line doesn't expand the plot area
                                ax.set_xlim(x_lim)
                                ax.set_ylim(y_lim)

    
                        # #show the univariate plots
                        # plt.tight_layout(w_pad=0.5)
                        # 2. Save the plot to a file BEFORE calling plt.show()
                        plt.savefig('D:/CEOS/OutputData/plot_conditional_'+validation_dict1['name']+'_'+validation_dict2['name']+variable+'.png', dpi=300, bbox_inches='tight')
                        plt.show()           

    return


def readCSV(path) :
    """
    Reads all CSV files from a directory into a pandas dataframe
    
    Args:
        path (path): Windos or POSIX pathname
        
    Returns:
        df (dataframe):  Dataframe with all CSV files contaatented by row
    """
    # Use glob to get all csv files in that folder
    all_files = glob.glob(os.path.join(path, "*.csv"))
    
    # Read each file and store them in a list
    li = []
    for filename in all_files:
        print(filename)
        df = pd.read_csv(filename, index_col=None, header=0)
        df['source_file'] = os.path.basename(filename) # Adds the filename as a column
        li.append(df)
    
    # Concatenate all DataFrames into
    df = pd.concat(li, axis=0, ignore_index=True)

    return df


def readCSV_gcloud(bucket) :
    """
    Reads all CSV files from a google cloud bucket into a pandas dataframe
    
    Args:
        bucket (string): Google Cloud Bucket
        
    Returns:
        df (dataframe):  Dataframe with all CSV files contaatented by row
    """
    # Use glob to get all csv files in that folder
    blob_list = bucket.list_blobs()
    csv_blob_list = [blob.name for blob in blob_list if blob.name.lower().endswith('.csv')]

    # Read each file and store them in a list
    li = []
    for csv_blobname in csv_blob_list:
        print(csv_blobname)
        csv_blob = bucket.get_blob(csv_blobname)
        content = csv_blob.download_as_bytes()
        df = pd.read_csv(BytesIO(content), index_col=None, header=0)
        df['source_file'] = csv_blobname # Adds the filename as a column
        li.append(df)
    
    # Concatenate all DataFrames into
    df = pd.concat(li, axis=0, ignore_index=True)

    return df

def expand_Fmask(df):
    """
    Expands a column in a dataframe corresponding to the GEE HLS Fmask into
    separate columns.  

    See https://developers.google.com/earth-engine/datasets/catalog/NASA_HLS_HLSL30_v002
    and https://developers.google.com/earth-engine/datasets/catalog/NASA_HLS_HLSS30_v002#bands
    
    Args:
        df (dataframe): Dataframe with a column labelled 'Fmask' containing Fmask values
        
    Returns:
        df (dataframe):  Dataframe with columnds added for cloud, shadow; adjacent shadow, water and snow flags
    """
    
    # df['cloud'] = np.bitwise_and(np.right_shift(df['Fmask'],1),1)
    # df['shadow'] = np.bitwise_and(np.right_shift(df['Fmask'],2),1) 
    # df['adjacent'] = np.bitwise_and(np.right_shift(df['Fmask'],3),1) 
    # df['snow'] = np.bitwise_and(np.right_shift(df['Fmask'],4),1) 
    # df['water'] = np.bitwise_and(np.right_shift(df['Fmask'],5),1)
    # df['aerosolmoderate'] = np.bitwise_and(np.right_shift(df['Fmask'],7),1)
    # df['aerosolhigh'] = np.bitwise_and(np.right_shift(df['Fmask'],7),3)
    # df['aerosollow'] = np.bitwise_and(np.right_shift(df['Fmask'],6),1)
    # df['aerosolclim'] = 1-np.bitwise_and(np.right_shift(df['Fmask'],6),1)
    df['noflags'] = (df['Fmask']==64).astype(int)
    return df

def filter_clearsky(df,mask,groupby_list):
    """
    Returns only rows of dataframe where there are only zero values for all other rows in the group.  


    Args:
        df (dataframe): Dataframe with columns coresponding to the mask and the groupby_list
        mask (string): Mask column name, assumed 0 values are not masked
        groupby_list: List to group rows when checking mask
        
    Returns:
        df (dataframe):  Dataframe with only rows with zero values for all other rows in group
    """
    
    filtered_df = df.groupby(groupby_list).filter(lambda x: (x[mask] == 1).all())
    return filtered_df

def NDVI(df,red_name,nir_name):
    """
    Adds NDVI to a df


    Args:
        df (dataframe): Dataframe with columns coresponding that include a red and nir band
        red_name (string): Name of red band column
        nir_name (string): Name of nir band column
    Returns:
        df (dataframe):  Dataframe with ndvi added
    """
    
    ndvi= ((df[nir_name] - df[red_name])/(df[nir_name] - df[red_name])).to_numpy()
    return ndvi

def match_FRM(df_satellite,df_FRM,time_interval='1day'):
    # match within k day interval
    return pd.merge_asof(df_satellite.sort_values('Date'),df_FRM.sort_values('Date'),on='Date', 
        by='Plot', 
        direction='nearest',
         suffixes = ('','_FRM'),
        tolerance = pd.Timedelta(time_interval)).dropna(subset=['date_FRM'])



def filter_quantile(df,y,lower_q,upper_q):
    """
    Filters rows where y variable falls outside quatile bounds for each group of x variables


    Args:
        df (dataframe): Dataframe to be filtered
        y (string): y variable to for filter
        lower_q (float): quantile between 0 and 1 inclusive
        upper_q (float): quantile between 0 and 1 inclusive

    Returns:
        df (dataframe):  Dataframe filtered
    """
    # Calculate the bounds per group 'x'
    lower_bound = df[y].quantile(lower_q)
    upper_bound = df[y].quantile(upper_q)

    # Filter the original DataFrame
    filtered_df = df[(df[y] >= lower_bound) & (df[y] <= upper_bound)]

    return filtered_df

def apply_quantile_filter_toESU(df,x,y,lower_q,upper_q):
    df_result = pd.DataFrame()
    for sensor in df['sensor'].unique():
        df_sensor = df.loc[df['sensor']==sensor]
        for ESU in df_sensor['Plot'].unique():
            filtered_df = df_sensor.loc[df_sensor['Plot']==plot].groupby(x) \
                            .apply(lambda group_df: filter_quantile(group_df,y,lower_q,upper_q),include_groups=False).reset_index(drop=True)
            filtered_df['Plot']=ESU
            df_result = pd.concat([df_result.reset_index(drop=True),filtered_df],axis=0).reset_index(drop=True)
    return df_result
    
def robust_gpr_fit(df,variable,trim):
    """
    Fits a robust gpr interpolator for data after trimming date for each date 


    Args:
        df (dataframe):  dataframe
        variable (string): Name of variable to fit
        trim (list): Tuple [low trim percentile,high trim percentile]
        
    Returns:
        gpr model:  gpr model
    """
    # trim data for each unique date
    return

def applySL2PtoHLS(df,algorithm):

    os.chdir('D:/CEOS/modules') 

    S2inputs = df
    S2inputs['RAA'] =  np.abs(S2inputs['SAA']-S2inputs['VAA'])
    S2inputs[['cosSZA','cosVZA','cosRAA']] = np.cos(np.radians(S2inputs[['SZA','VZA','RAA']]))
    if (algorithm=='HLSL30'):
        S2inputs = S2inputs[['B3','B4','B8A','B11','B12','cosSZA','cosVZA','cosRAA']].rename(columns={'B8A':'B5','B11':'B6','B12':'B7'}).reset_index()
    else:
        S2inputs = S2inputs[['B3','B4','B5','B6','B7','B8A','B11','B12','cosSZA','cosVZA','cosRAA']].rename(columns={'B3':'B03','B4':'B04','B5':'B05','B6':'B06','B7':'B07'}).reset_index()
    
    df= pd.concat([df.reset_index(),Algos.SL2PINRA(S2inputs,'fAPAR',algorithm).rename(columns={'networkID':'fAPAR_networkID','QC_input':'fAPAR_QC_input','QC_output':'fAPAR_QC_output'}).reset_index(),\
                        Algos.SL2PINRA(S2inputs,'LAI',algorithm).rename(columns={'networkID':'LAI_networkID','QC_input':'LAI_QC_input','QC_output':'LAI_QC_output'}).reset_index()],axis=1)
    
    #Only select samples with valid range and domain for both LAI and FAPAR and NDVI
    df= df.loc[(df['fAPAR_QC_input']==0) & (df['LAI_QC_input']==0) & (df['fAPAR_QC_output']==0) & (df['LAI_QC_output']==0)  ]
    
    #Filter unreasonable NDVI values
    df.loc[:,'NDVI'] = (df['B8A'] - df['B4'])/(df['B8A'] + df['B4'])
    df  = df.loc[(df['NDVI']>0.01) & (df['NDVI']<0.99) ]

    return df.drop(columns=['index'])


# def get_Matchups(df_SL2P,FRM,variable,time_interval):
    
#     df = pd.merge_asof(df_SL2P.sort_values('Date'),FRM.sort_values('Date'),on='Date', 
#         by='Plot', 
#         direction='nearest',
#          suffixes = ('','_FRM'),
#         tolerance = pd.Timedelta(time_interval)).dropna(subset=['date_FRM'])
    
#     #only keep samples with same land cover
#     df = df[df['NLCD'] == df['NLCD_FRM']]
    
#     #get the  median estimated variable for each unique plot,date
#     df[['medianestimate'+variable,'medianestimate'+variable+'_ci']] = np.abs(df.groupby(['Plot', 'Date'],as_index=False)[['estimate'+variable,'error'+variable]].transform('median'))
    
#     #only retain the median estimate 
#     df_median = df.drop_duplicates(subset=['Plot','Date'])

#     # # # estimate residual for LAI and fAPAR
#     df_median['medianresidual'+variable] = df_median['medianestimate'+variable] - df_median[variable]
#     df_median['absmedianresidual'+variable] = np.abs(df_median['medianresidual'+variable])
#     df_median['relabsmedianresidual'+variable] = df_median['absmedianresidual'+variable]/ df_median['medianestimate'+variable] 
      


#     # #estimate the uncerainty of the estimated error
#     if (variable=='LAI'):
#         df_median['medianestimate'+variable+'_ci_ci'] = np.minimum(0.5,0.2 * df_median['medianestimate'+variable+'_ci'])
#     elif (variable=='fAPAR'): 
#         df_median['medianestimate'+variable+'_ci_ci'] = np.minimum(0.05,0.2 * df_median['medianestimate'+variable+'_ci'])
    
#     #return the date with the lowest absolute median residual for each FRM
#     return df_median[df_median.groupby(['Plot', 'date_FRM'],as_index=False)['absmedianresidual'+variable].transform('min') == df_median['absmedianresidual'+variable]].drop_duplicates("absmedianresidual"+variable)
    

def get_Matchups(df_SL2P,df_FRM,variable,time_interval):

    #ensure same data types of merge keys
    df_SL2P['Plot']= df_SL2P['Plot'].astype(str)
    df_FRM['Plot']= df_FRM['Plot'].astype(str)
    df_SL2P['Date']= df_SL2P['Date'].astype('datetime64[ns]')
    df_FRM['Date']= df_FRM['Date'].astype('datetime64[ns]')
    
    df = pd.merge_asof(df_SL2P.sort_values('Date'),df_FRM.sort_values('Date'),on='Date', 
        by='Plot', 
        direction='nearest',
         suffixes = ('','_FRM'),
        tolerance = pd.Timedelta(time_interval)).dropna(subset=['date_FRM'])

    #only keep samples with same land cover
    df = df[df['NLCD'] == df['NLCD_FRM']]

      
    #estimate residual for LAI and fAPAR
    df['residual'+variable] = df['estimate'+variable] - df[variable+'_FRM']
    df['absresidual'+variable] = np.abs(df['residual'+variable])
    df['relabsresidual'+variable] = df['absresidual'+variable]/ df.groupby(['Plot','date'],as_index=False)['absresidual'+variable].transform('median')
      
    #get theabsolute  median residual for each unique plot,date
    df['medianresidual'+variable] = (df.groupby(['Plot', 'Date'],as_index=False)['residual'+variable].transform('median'))
    df['absmedianresidual'+variable] = np.abs(df['medianresidual'+variable] )
    df['absmedianerror'+variable] = np.abs(df.groupby(['Plot', 'Date'],as_index=False)['error'+variable].transform('median'))


    #get the  median estimated variable for each unique plot,date
    df['medianestimate'+variable] = np.abs(df.groupby(['Plot', 'Date'],as_index=False)['estimate'+variable].transform('median'))


    #get the  median estimated variable confidence interval  for each unique plot,date
    df['medianestimate'+variable+'_ci'] = np.abs(df.groupby(['Plot', 'Date'],as_index=False)['error'+variable].transform('median'))

    
    #estimate the uncerainty of the estimated error
    df['medianestimate'+variable+'_ci_ci'] = np.maximum(0.05,0.2 * df['medianestimate'+variable+'_ci'])

        
    #get the relative absolute median residuals
    df['relabsmedianresidual'+variable] = df['absmedianresidual'+variable]/df['medianestimate'+variable] 

    
    #return the date with the lowest absolute median residual for each FRM
    return df[df.groupby(['Plot', 'date_FRM'],as_index=False)['absmedianresidual'+variable].transform('min') == df['absmedianresidual'+variable]].drop_duplicates("absmedianresidual"+variable)
    
# parse grounded EO FRM 
def parse_FRM(df):
    """
    Parse GroundedEO Fiducial Reference Measurements into a data frame 


    Args:
        df (dataframe):  input dataframe

        
    Returns:
        df (dataframe):  output dataframe
    """
    
    for colName in ['PAIe_up','PAIe_down', 'PAI_up','PAI_down','PAIe_Miller_up','PAIe_Miller_down', 'PAI_Miller_down', \
                    'FCOVER_down2', 'WAI_up','Alpha','LAIe_up','LAI_up','LAIe_Miller_up' ,'FAPAR_up','LAIe', \
                    'LAI', 'LAIe_Miller','LAI_Miller','FAPAR','FCOVER','FIPAR_up']:
        df[colName]=df[colName].replace('-999','0+/-0')
        df[[colName,'temp1']]=df[colName].str.split('/', expand=True)
        df[[colName,'temp']]=df[colName].str.split('+', expand=True)
        
        df['temp1'] = df['temp1'].fillna('-None')
        df2 =df['temp1'].str.split('-', expand=True)
        if ( df2.shape[1] == 2):
            df[['temp2',colName+'_ci']]=df2
        else:
            df[['temp2',colName+'_ci','temp3']]=df2
        df[[colName,colName+'_ci']] = df[[colName,colName+'_ci']].replace('',np.nan)
        df[[colName,colName+'_ci']] = df[[colName,colName+'_ci']].replace(to_replace=r'.*e.*', value='0', regex=True)

        df[[colName,colName+'_ci']]=df[[colName,colName+'_ci']].astype(float)
    df=df.replace(-999,0)
    df['PAIe'] = df['PAIe_up']+df['PAIe_down']
    df['PAI'] = df['PAI_up']+df['PAI_down']
    df['FAPAR_down']  = df['FAPAR']- df['FAPAR_up']
    
    # FRM[['FAPAR_ci','LAI_ci']] = FRM[['FAPAR_ci','LAI_ci']].astype(float)
    df['LAI_down'] = df['LAI']-df['LAI_up']

    #minimum uncertaint for WAI_up
    df['WAI_up_ci']=np.maximum(0.05,df['WAI_up_ci'].to_numpy())
                               
    df['date'] = (pd.to_datetime(df['Date'],format='mixed'))
    df['year'] = df['date'].dt.year
    df['site_id']= pd.factorize(df['Site'])[0]
    df['site_id_ci']= 0.00001
    
    df['NLCD_group'] = df['NLCD'].map({'evergreenForest': 'needleleafForest',
                           'deciduousForest': 'broadleafForest',
                           'mixedForest': 'broadleafForest',
                           'woodyWetlands': 'broadleafForest',
                           'grasslandHerbaceous': 'nonForest',
                           'emergentHerbaceousWetlands': 'nonForest',
                           'dwarfScrub': 'nonForest',
                           'sedgeHerbaceous': 'nonForest',

                           'shrubScrub': 'nonForest',
                           'pastureHay': 'nonForest',
                           'cultivatedCrops': 'nonForest'})
    return(df)