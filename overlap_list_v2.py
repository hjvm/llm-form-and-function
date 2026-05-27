#!/usr/bin/python3
import sys
import os
import math
import numpy as np
from nltk.lm import NgramCounter
from nltk.util import bigrams
from scipy.stats import linregress
from typing import List, Tuple


def readindata(fname):

    # Get ngram counts from the file.
    ngrams = NgramCounter()
    with open(fname, mode="r") as f:
        # Read the first line in the file.
        next_line = f.readline()
        # Keep going while there are more lines to read.
        while next_line:
            tokens = next_line.strip().split()
            next_line = f.readline()
            if len(tokens) != 2:
                #print(tokens)
                continue
            bgrms = bigrams(tokens)#, min_len=2, max_len=2)
            ngrams.update([bgrms])
        
    # Eliminate any empty keys.
    for key in ngrams[2].conditions():
        if len(ngrams[2][key]) == 0:
            ngrams[2].pop(key)

    #print(ngrams.N())    
    return ngrams

def parse_data_from_file(fname):
    # Get ngrams from given file.
    ngrams = readindata(fname)
    # Now convert NLTK ngram object into a numpy matrix.

    # Closed class category.
    c = ngrams[2].conditions()
    c2idx = {k:v for v, k in enumerate(c)}
    #idx2c = {v:k for k, v in c2idx.items()}

    # Open class category.
    # merge the keys of the dictionaries from each item of the closed class.
    o = set().union(*ngrams[2].values()) 
    o2idx = {k:v for v, k in enumerate(o)}
    #idx2o = {v:k for k, v in o2idx.items()}

    # Create matrix of counts.
    co_counts = np.zeros((len(c), len(o)))
    for c_type in c:
        for o_type in o:
            c_idx = c2idx[c_type]
            o_idx = o2idx[o_type]
            co_counts[c_idx, o_idx] = ngrams[2][c_type].get(o_type, 0)
    
    return co_counts, c2idx, o2idx

def build_ngram_counter_from_pairs(pairs: List[str]) -> NgramCounter:
    ngrams = NgramCounter()
    for pair in pairs:
        tokens = pair.strip().split()
        if len(tokens) != 2:
            continue
        bgrms = bigrams(tokens)
        ngrams.update([bgrms])
    return ngrams


def parse_data_from_pairs(pairs: List[str]):
    ngrams = build_ngram_counter_from_pairs(pairs)

    c = ngrams[2].conditions()
    c2idx = {k: v for v, k in enumerate(c)}

    o = set().union(*ngrams[2].values())
    o2idx = {k: v for v, k in enumerate(o)}

    co_counts = np.zeros((len(c), len(o)))
    for c_type in c:
        for o_type in o:
            c_idx = c2idx[c_type]
            o_idx = o2idx[o_type]
            co_counts[c_idx, o_idx] = ngrams[2][c_type].get(o_type, 0)

    return co_counts, c2idx, o2idx


def base_overlap_stats(co_matrix):
    N = co_matrix.shape[1]
    S = co_matrix.sum(dtype=int)

    bias = find_bias(co_matrix)
    r = float(S)/float(N)

    return (N, S, bias, r)


def Harmonic(n, a=1):
    s=0
    for i in range(1, n+1):
        s+=1.0/math.pow(i, a)
    return s

def expected_overlap(N, S, r, a=1, b=0.82):
    hN = Harmonic(N, a)
    p = 1.0/(math.pow(r,a)*hN)
    eo = 1 - sum( [ math.pow((p*di+1.0-p), S) for di in [b, 1.0-b] ] ) + math.pow(1-p, S)
    assert eo>0, 'r=%d, predicted=%.6f'%(r, eo)
    return eo

def average_expected_overlap(N, S, a=1, b=0.82):
    sumo = 0
    for r in range(1, N+1):
        sumo += expected_overlap(N, S, r, a, b)
    return sumo/N

def simple_expected_overlap(co_matrix, b=0.82):
    '''Use the empirical frequencies of the open class items,
    and use the universal bias value of 0.82 (b=0.82), to do the prediction. 
    This is very simple. If a noun has frequency of f, 
    its expected overlap is simple 1-0.82^f-0.18^f
    '''
    # Convert counts to frequencies.
    #print(co_matrix.shape)
    freq_matrix = co_matrix.sum(axis=0) #/ co_matrix.sum()
    #print(freq_matrix.shape)
    exp_overlap = 1 - np.power(b, freq_matrix) - np.power(1-b, freq_matrix)
    #print(exp_overlap.shape)
    return exp_overlap.mean()

def empirical_overlap(co_matrix): # returns s, n, and empirical overlap
    # If only one determiner appears, no noun can co-occur with both,
    # so overlap is 0. Without this guard, a 1-row matrix gives a
    # spurious 1.0 because all(axis=0) is trivially True.
    if co_matrix.shape[0] < 2:
        return 0.0
    emp_overlap = co_matrix.all(axis=0).sum()/co_matrix.shape[1]
    return emp_overlap


def find_bias(co_matrix):

    bias = co_matrix.max(axis=0).sum()/co_matrix.sum()    
    return bias
        
def zipfian_fit(co_matrix):
    '''Check how well a given dataset is Zipfian by checking the
       fit between the word frequencies and their rank.'''
    # Create a rank and a log(frequency) array sorted in descending order.
    x = np.log(np.arange(start=co_matrix.shape[1], stop=0, step=-1))
    x = np.sort(x)
    y = -np.sort(-np.log(co_matrix.sum(axis=0)))
    
    # Perform linear regression
    slope, intercept, r_value, p_value, std_err = linregress(x, y)
    
    # Print results
    return slope

def pred_vs_emp_overlap(fname, verbose=True):
    # Matrix of bigram (C-O) counts.
    co_counts, _, _ = parse_data_from_file(fname)

    # Calculate unique nouns (N), sample size (S), 
    # category bias, and token/type ration (r).
    (N, S, bias, r) = base_overlap_stats(co_counts)

    emp_overlap = empirical_overlap(co_counts)
    avg_pred_overlap = average_expected_overlap(N, S, bias)
    if verbose:
        print("N=", N, "S=", S, "bias=", "%.4f" % bias, "r=", "%.4f" % (r),  "Empirical=",  "%.4f" % (emp_overlap),  "Predicted=", "%.4f" % (avg_pred_overlap))
    
    return {"N":N, "S":S, "bias":bias, "r":r, "empirical":emp_overlap, "predicted":avg_pred_overlap}

def simple_vs_emp_overlap(fname, b=0.82, verbose=True):
    '''Calculate empirical overlap vs. simple predicted overlap, which
    relies on a universal bias value of b=0.82.'''
    # Matrix of bigram (C-O) counts.
    co_counts, _, _ = parse_data_from_file(fname)

    # Check whether the distribution is zipfian.
    slope = zipfian_fit(co_counts)

    # Calculate unique nouns (N), sample size (S), 
    # category bias, and token/type ration (r).
    (N, S, bias, r) = base_overlap_stats(co_counts)

    emp_overlap = empirical_overlap(co_counts)
    simple_overlap = simple_expected_overlap(co_counts, b=b)
    if verbose:
        print("N=", N, "S=", S, "emp_bias=", "%.4f" % bias,"univ_bias=", "%.4f" % b, "r=", "%.4f" % (r),  "Empirical=",  "%.4f" % (emp_overlap),  "Predicted=", "%.4f" % (simple_overlap))
    
    return {"N":N, "S":S, "bias":b, "r":r, "empirical":emp_overlap, "predicted":simple_overlap, "linear_fit":slope}

def all_overlap_stats(source, a=None, b=None, verbose=True):
    """
    Calculate all overlap metrics. Can accept either a filename (str)
    or a list of determiner-noun strings.

    Parameters
    ----------
    source : str or List[str]
        If str and is a valid path, will read pairs from file.
        If list, will treat as list of "Det Noun" pairs.
    """
    if isinstance(source, str) and os.path.isfile(source):
        co_counts, _, _ = parse_data_from_file(source)
    elif isinstance(source, list):
        co_counts, _, _ = parse_data_from_pairs(source)
    else:
        raise ValueError(
            "Invalid input to all_overlap_stats: expected a valid filename (str) or a list of strings."
        )

    # Check whether the distribution is zipfian.
    slope = zipfian_fit(co_counts)
    if a is None:
        a = slope
        
    # Calculate unique nouns (N), sample size (S), 
    # empirical category bias, and token/type ration (r).
    (N, S, bias, r) = base_overlap_stats(co_counts)
    if b is None:
        # Use the empirical bias value.
        b = bias

    emp_overlap = empirical_overlap(co_counts)
    
    try:
        naive_pred_overlap = average_expected_overlap(N, S, b=bias)
    except AssertionError:
        naive_pred_overlap = float('nan')
        
    try:
        # NOTE: We do not want to correct for non-zipfian slope.
        #adj_pred_overlap = average_expected_overlap(N, S, a=a, b=b)
        adj_pred_overlap = average_expected_overlap(N, S, b=b)
    except AssertionError:
        adj_pred_overlap = float('nan')

    if verbose:
        print("N=", N, "S=", S, "emp_bias=", "%.4f" % bias,
              "univ_bias=", "%.4f" % b, "r=", "%.4f" % (r),  
              "Empirical=",  "%.4f" % (emp_overlap),  
              "Naive Predicted=", "%.4f" % (naive_pred_overlap) if not np.isnan(naive_pred_overlap) else "NaN", 
              "Linear fit=","%.4f" % (slope),
              "Adjusted Predicted=", "%.4f" % (adj_pred_overlap) if not np.isnan(adj_pred_overlap) else "NaN")
    
    return {"N":N, "S":S, "emp_bias":bias, "univ_bias":b, "r":r, 
            "empirical":emp_overlap, 
            "naive_predicted":naive_pred_overlap, 
            "linear_fit":slope,
            "adj_predicted":adj_pred_overlap}
    
    
def main():

    if len(sys.argv) > 1:
        fname = sys.argv[1]
        pred_vs_emp_overlap(fname, verbose=True)
    

    
if __name__ == "__main__":
    main()