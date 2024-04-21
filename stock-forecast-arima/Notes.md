### V1 Model

## Issues

- Uses sequence of 1
- Not a sequence
- Can't beat naive model (random walk theory)
- Min-max scaling: There is no hypothetical "high". What even is the "low"? Do you need 50 years of data for that?

## Fixes

- Use stock returns instead of price (more stationary)
- Use standardization instead, since distribution is closer to t-distribution/gaussian
- Models to use: ARIMA, LSTM(?)
