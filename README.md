Intermediate Deep Learning Final Projection by @rithums and @aLyonsGH

The Black-Scholes Equation is a differential equation that represents a call option
pricing model. It computes call option prices by taking in the strike price, stock
price, time maturity, interest rate, and volatility. In practice, implied volatility is
used since volatility can’t be directly observed, rather, can only be derived inversely
by observing option prices from the market. Previous work uses feedforward neural
networks to predict call option pricing, and various architectures to compute the
implied volatility needed as input. Our work uses an encoder-only transformer
to compute the implied volatility given previous strike prices, stock prices, time
maturities, and interest rates in the form of a time series transformed into embeddings, and we input the volatility into a forward neural network along with the
other Black-Scholes inputs to predict call option pricing. Using the transformer in
addition to the feedforward model yielded performance improvement, providing
evidence that transformers are useful for computing useful implied volatilities.
