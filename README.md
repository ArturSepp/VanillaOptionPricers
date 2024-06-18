# VanillaOptionPricers
 Fast and vectorised pricer and implied volatility fitters for Black-Scholes and Merton models

Minimum dependencies on higher level packages

Core dependencies:

    python = ">=3.8,<3.11"
    numba = ">=0.59.0"
    numpy = ">=1.26.4"


Installation

    pip install vanilla_option_pricers

Update

    pip install --upgrade vanilla_option_pricers



Supported Option types (passed as string):

    CALL = 'C'
    PUT = 'P'
    INVERSE_CALL = 'IC'
    INVERSE_PUT = 'IP'
