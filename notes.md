- SPY: Weighted average (float-adjusted) of top 500 S&P companies
- Float-adjusted: only counmts the share that are realistically available for public investors to trade
- The ones not counted can be

+ Founder/insider-owned share
+ Government
+ Parent-company
+ ...

- S&P 500 -> just an index, not tradable
- SPY: tries to mimic S&P 500, work as a tradable stock

Nasdaq -> 100 largest non-financial companies

- Non-financial companies behave differnt - they lent money, manage money, insure risk, trading financial assets
- Normal company sell product, and revenue - cost = profit
  -> Different things to track

Stats

- Total return: how much strat made over full test
- Annual return: average compounded return per year
- Annual volatility: how much strat's daily returns bounce around (annualized))
  - Calculated: Standard deviation of the daily returns, multiply by sqrt(252) - number of days of trading per year
- Sharpe (assume risk_free = 0, which might not be the case)
- Max drawdown
  - For each point of time compare to the highest portfolio value seen so far. Of course it is worse (or equal)
  - Drawdown = worst one
- calmar = annual return / abs(max_drawdown)
- directional accuracy (only for spome)
  - How often model correctly predicted up vs down
- psitive_day_rate
  - How many days do we make money (strat return > 0)
- avg_exposure
  - How invested strat is on average
- annual_turnover
  - How much we invest in total? How much does the strat changes on average every day * 252

What models actually do

- Best model: allin cash / allin SPY,
