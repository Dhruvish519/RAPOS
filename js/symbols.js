/**
 * Popular US-listed stocks, ETFs and crypto, for instant suggestions in the
 * "Add a Holding" search. Anything not listed here is found through the
 * backend's live search (/api/search-stocks).
 */
window.RAPOS_SYMBOLS = [
    // Stocks
    ["AAPL", "Apple Inc.", "Stock"], ["MSFT", "Microsoft Corporation", "Stock"],
    ["NVDA", "NVIDIA Corporation", "Stock"], ["GOOGL", "Alphabet Inc. (Google) Class A", "Stock"],
    ["AMZN", "Amazon.com, Inc.", "Stock"], ["META", "Meta Platforms, Inc. (Facebook)", "Stock"],
    ["TSLA", "Tesla, Inc.", "Stock"], ["BRK-B", "Berkshire Hathaway Inc. Class B", "Stock"],
    ["AVGO", "Broadcom Inc.", "Stock"], ["JPM", "JPMorgan Chase & Co.", "Stock"],
    ["V", "Visa Inc.", "Stock"], ["MA", "Mastercard Incorporated", "Stock"],
    ["UNH", "UnitedHealth Group Incorporated", "Stock"], ["JNJ", "Johnson & Johnson", "Stock"],
    ["LLY", "Eli Lilly and Company", "Stock"], ["XOM", "Exxon Mobil Corporation", "Stock"],
    ["WMT", "Walmart Inc.", "Stock"], ["PG", "Procter & Gamble Company", "Stock"],
    ["KO", "The Coca-Cola Company", "Stock"], ["PEP", "PepsiCo, Inc.", "Stock"],
    ["COST", "Costco Wholesale Corporation", "Stock"], ["HD", "The Home Depot, Inc.", "Stock"],
    ["DIS", "The Walt Disney Company", "Stock"], ["NFLX", "Netflix, Inc.", "Stock"],
    ["ADBE", "Adobe Inc.", "Stock"], ["CRM", "Salesforce, Inc.", "Stock"],
    ["ORCL", "Oracle Corporation", "Stock"], ["AMD", "Advanced Micro Devices, Inc.", "Stock"],
    ["INTC", "Intel Corporation", "Stock"], ["QCOM", "QUALCOMM Incorporated", "Stock"],
    ["CSCO", "Cisco Systems, Inc.", "Stock"], ["IBM", "International Business Machines", "Stock"],
    ["BAC", "Bank of America Corporation", "Stock"], ["GS", "The Goldman Sachs Group, Inc.", "Stock"],
    ["NKE", "NIKE, Inc.", "Stock"], ["MCD", "McDonald's Corporation", "Stock"],
    ["SBUX", "Starbucks Corporation", "Stock"], ["BA", "The Boeing Company", "Stock"],
    ["PFE", "Pfizer Inc.", "Stock"], ["UBER", "Uber Technologies, Inc.", "Stock"],
    ["PYPL", "PayPal Holdings, Inc.", "Stock"], ["SHOP", "Shopify Inc.", "Stock"],
    ["PLTR", "Palantir Technologies Inc.", "Stock"], ["INFY", "Infosys Limited (ADR)", "Stock"],
    ["HDB", "HDFC Bank Limited (ADR)", "Stock"], ["IBN", "ICICI Bank Limited (ADR)", "Stock"],
    ["WIT", "Wipro Limited (ADR)", "Stock"], ["TSM", "Taiwan Semiconductor (ADR)", "Stock"],
    // ETFs
    ["SPY", "SPDR S&P 500 ETF Trust", "ETF"], ["VOO", "Vanguard S&P 500 ETF", "ETF"],
    ["QQQ", "Invesco QQQ Trust (Nasdaq-100)", "ETF"], ["VTI", "Vanguard Total Stock Market ETF", "ETF"],
    ["DIA", "SPDR Dow Jones Industrial Average ETF", "ETF"], ["IWM", "iShares Russell 2000 ETF", "ETF"],
    ["VEA", "Vanguard FTSE Developed Markets ETF", "ETF"], ["VWO", "Vanguard FTSE Emerging Markets ETF", "ETF"],
    ["INDA", "iShares MSCI India ETF", "ETF"], ["BND", "Vanguard Total Bond Market ETF", "ETF"],
    ["AGG", "iShares Core U.S. Aggregate Bond ETF", "ETF"], ["TLT", "iShares 20+ Year Treasury Bond ETF", "ETF"],
    ["IEF", "iShares 7-10 Year Treasury Bond ETF", "ETF"], ["SHY", "iShares 1-3 Year Treasury Bond ETF", "ETF"],
    ["LQD", "iShares iBoxx Investment Grade Corporate Bond ETF", "ETF"], ["TIP", "iShares TIPS Bond ETF", "ETF"],
    ["GLD", "SPDR Gold Shares", "ETF"], ["IAU", "iShares Gold Trust", "ETF"],
    ["SLV", "iShares Silver Trust", "ETF"], ["VNQ", "Vanguard Real Estate ETF", "ETF"],
    ["XLK", "Technology Select Sector SPDR Fund", "ETF"], ["XLF", "Financial Select Sector SPDR Fund", "ETF"],
    ["XLE", "Energy Select Sector SPDR Fund", "ETF"], ["XLV", "Health Care Select Sector SPDR Fund", "ETF"],
    ["SCHD", "Schwab U.S. Dividend Equity ETF", "ETF"], ["ARKK", "ARK Innovation ETF", "ETF"],
    // Crypto
    ["BTC", "Bitcoin", "Crypto"], ["ETH", "Ethereum", "Crypto"], ["SOL", "Solana", "Crypto"],
    ["XRP", "XRP", "Crypto"], ["ADA", "Cardano", "Crypto"], ["DOGE", "Dogecoin", "Crypto"],
    ["BNB", "BNB", "Crypto"], ["AVAX", "Avalanche", "Crypto"], ["DOT", "Polkadot", "Crypto"],
    ["LTC", "Litecoin", "Crypto"],
].map(([symbol, name, type]) => ({ symbol, name, type }));
