import torch
import torch.nn as nn


def non_linearity_guard(activation):
    activation = activation.lower()
    allowed = {'tanh', 'relu', 'gelu', 'none'}
    assert activation in allowed, "The non-linearity must be in 'tanh', 'relu', 'gelu', 'none'"

    if activation == 'tanh':
        return nn.Tanh()
    elif activation == 'relu':
        return nn.ReLU()
    elif activation == 'gelu':
        return nn.GELU()
    elif activation == 'none':
        return nn.Identity()
    
def shape_guard(tensor1, tensor2):
    assert tensor1.shape == tensor2.shape, f"tensor 1 has shape: {tensor1.shape}, whilst tensor 2 has shape: {tensor2.shape}"

def stock_ticker_guard(ticker_symbol):
    ticker_symbol = ticker_symbol.upper()
    allowed = {"AAPL", "GOOGL"}
    assert ticker_symbol in allowed, "The ticker symbol must be in AAPL, "