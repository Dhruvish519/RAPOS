/**
 * Live price ticker: real quotes from the RAPOS backend (/api/live-price,
 * which calls Finnhub). Fills every [data-live-ticker] element and reveals
 * its container ([data-live-ticker-box]); if no quotes come back (backend
 * asleep, no API key, market data down) the ticker simply stays hidden, so
 * it never shows made-up prices. Needs js/rapos-config.js loaded first.
 */
(function () {
    var SYMBOLS = ['AAPL', 'MSFT', 'NVDA', 'TSLA', 'AMZN', 'GOOGL'];
    function load() {
        var targets = document.querySelectorAll('[data-live-ticker]');
        if (!targets.length || !window.RAPOS_API_BASE) return;
        Promise.all(SYMBOLS.map(function (t) {
            return fetch(window.RAPOS_API_BASE + '/api/live-price?ticker=' + t)
                .then(function (r) { return r.json(); })
                .catch(function () { return null; });
        })).then(function (quotes) {
            var ok = quotes.filter(function (q) { return q && !q.error && q.current_price > 0; });
            if (!ok.length) return;
            var items = ok.map(function (q) {
                var up = q.percent_change >= 0;
                return '<div class="ticker-item"><span>' + q.ticker + '</span> <span class="' +
                    (up ? 'trend-up ticker-up' : 'trend-down ticker-down') + '">$' +
                    q.current_price.toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 }) +
                    ' (' + (up ? '+' : '') + q.percent_change.toFixed(2) + '%)</span></div>';
            }).join('');
            var label = '<div class="ticker-item" style="opacity:.6;">LIVE · FINNHUB</div>';
            targets.forEach(function (el) {
                el.innerHTML = label + items + label + items;   // doubled for the seamless scroll loop
                var box = el.closest('[data-live-ticker-box]') || el;
                box.style.display = '';
            });
        });
    }
    if (document.readyState === 'loading') document.addEventListener('DOMContentLoaded', load);
    else load();
})();
