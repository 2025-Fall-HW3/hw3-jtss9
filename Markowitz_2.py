"""
Package Import
"""
import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import quantstats as qs
import gurobipy as gp
import warnings
import argparse
import sys

"""
Project Setup
"""
warnings.simplefilter(action="ignore", category=FutureWarning)

assets = [
    "SPY",
    "XLB",
    "XLC",
    "XLE",
    "XLF",
    "XLI",
    "XLK",
    "XLP",
    "XLRE",
    "XLU",
    "XLV",
    "XLY",
]

# Initialize Bdf and df
Bdf = pd.DataFrame()
for asset in assets:
    raw = yf.download(asset, start="2012-01-01", end="2024-04-01", auto_adjust = False)
    Bdf[asset] = raw['Adj Close']

df = Bdf.loc["2019-01-01":"2024-04-01"]

"""
Strategy Creation

Create your own strategy, you can add parameter but please remain "price" and "exclude" unchanged
"""

def my_opt_helper(mu, Sigma, gamma=0):
    n = len(mu)
    model = gp.Model("Markowitz Portfolio Optimization")
    model.setParam('OutputFlag', 0)  # Suppress Gurobi output

    # Decision Variables: Portfolio Weights
    w = model.addMVar(n, name="w", lb=0)

    # Objective: Maximize Risk-Adjusted Return
    linear_term = w @ mu
    quadratic_term = w @ Sigma @ w
    objective = linear_term - (gamma/2) * quadratic_term
    model.setObjective(objective, gp.GRB.MAXIMIZE)

    # Constraint: Weights Sum to 1
    model.addConstr(w.sum() == 1, name="budget")

    # Optimize the Model
    model.optimize()

    # Extract Optimal Weights
    opt_weights = w.X

    return opt_weights

class MyPortfolio:
    """
    NOTE: You can modify the initialization function
    """

    def __init__(self, price, exclude, lookback=50, gamma=0, momentum_window=20):
        self.price = price
        self.returns = price.pct_change().fillna(0)
        self.exclude = exclude
        self.lookback = lookback
        self.gamma = gamma
        self.momentum_window = momentum_window


    def calculate_weights(self):
        # Get the assets by excluding the specified column
        assets = self.price.columns[self.price.columns != self.exclude]
        asset_list = assets.tolist()

        # Calculate the portfolio weights
        self.portfolio_weights = pd.DataFrame(
            index=self.price.index, columns=self.price.columns
        )

        """
        TODO: Complete Task 4 Below
        """
        momentum_returns = self.returns[assets].rolling(window=self.momentum_window).sum().fillna(0)
        for i in range(self.lookback + 1, len(self.price)):
            R_n = self.returns[assets].iloc[i - self.lookback : i]
            mu_base = R_n.mean().values
            current_momentum = momentum_returns.iloc[i].values
            adjusted_mu = mu_base + 0.1*current_momentum
            mu = adjusted_mu

            if(i%5==0) or (i==self.lookback + 1):
                Sigma = R_n.cov().values
                last_Sigma = Sigma
            else:
                Sigma = last_Sigma
        
            opt_weights = my_opt_helper(mu, Sigma, self.gamma)
            self.portfolio_weights.loc[self.price.index[i], assets] = opt_weights
        self.portfolio_weights[self.exclude] = 0
        """
        TODO: Complete Task 4 Above
        """

        self.portfolio_weights.ffill(inplace=True)
        self.portfolio_weights.fillna(0, inplace=True)

    def calculate_portfolio_returns(self):
        # Ensure weights are calculated
        if not hasattr(self, "portfolio_weights"):
            self.calculate_weights()

        # Calculate the portfolio returns
        self.portfolio_returns = self.returns.copy()
        assets = self.price.columns[self.price.columns != self.exclude]
        self.portfolio_returns["Portfolio"] = (
            self.portfolio_returns[assets]
            .mul(self.portfolio_weights[assets])
            .sum(axis=1)
        )

    def get_results(self):
        # Ensure portfolio returns are calculated
        if not hasattr(self, "portfolio_returns"):
            self.calculate_portfolio_returns()

        return self.portfolio_weights, self.portfolio_returns


if __name__ == "__main__":
    # Import grading system (protected file in GitHub Classroom)
    from grader_2 import AssignmentJudge
    
    parser = argparse.ArgumentParser(
        description="Introduction to Fintech Assignment 3 Part 12"
    )

    parser.add_argument(
        "--score",
        action="append",
        help="Score for assignment",
    )

    parser.add_argument(
        "--allocation",
        action="append",
        help="Allocation for asset",
    )

    parser.add_argument(
        "--performance",
        action="append",
        help="Performance for portfolio",
    )

    parser.add_argument(
        "--report", action="append", help="Report for evaluation metric"
    )

    parser.add_argument(
        "--cumulative", action="append", help="Cumulative product result"
    )

    args = parser.parse_args()

    judge = AssignmentJudge()
    
    # All grading logic is protected in grader_2.py
    judge.run_grading(args)
