/**
 * RAPOS API Service
 * Centralized logic for all data fetching and state mutations.
 */

// Set once in js/rapos-config.js. `var` so it can coexist with supabase-client.js.
var API_BASE_URL = window.RAPOS_API_BASE || 'https://rapos-api.onrender.com';

const RAPOS_API = {
    /**
     * Auth & Session
     */
    async getSession() {
        return await supabase.auth.getSession();
    },

    async logout() {
        return await supabase.auth.signOut();
    },

    /**
     * Portfolio Management
     */
    async fetchHoldings() {
        try {
            const { data, error } = await supabase
                .from('portfolios')
                .select('*')
                .order('created_at', { ascending: false });
            if (error) throw error;
            const base = data || [];
            const quotes = await this.fetchQuotes(base.map(a => a.ticker));
            // Live prices from the backend (one batched call). No made-up
            // fallback: if a price can't be fetched the holding says so.
            return base.map(asset => {
                const t = (asset.ticker || '').toUpperCase();
                if (t === 'CASH') {
                    return { ...asset, current_price: 1, live_value: asset.quantity || 0,
                             performance_pct: 0, is_live: true };
                }
                const q = quotes[t];
                if (q && !q.error) {
                    return { ...asset, current_price: q.price, live_value: (asset.quantity || 0) * q.price,
                             performance_pct: +(q.percent_change || 0).toFixed(2), is_live: true,
                             price_source: q.source, stale: !!q.stale };
                }
                return { ...asset, current_price: null, live_value: 0, performance_pct: 0,
                         is_live: false, price_error: true };
            });
        } catch (e) {
            console.error('fetchHoldings error:', e);
            return [];
        }
    },

    /** Batched live prices: {AAPL: {price, percent_change, source}, ...} */
    async fetchQuotes(tickers) {
        const list = [...new Set((tickers || []).map(t => (t || '').toUpperCase()).filter(t => t && t !== 'CASH'))];
        if (!list.length) return {};
        try {
            const res = await fetch(`${API_BASE_URL}/api/quotes?tickers=${encodeURIComponent(list.join(','))}`);
            if (res.ok) return (await res.json()).quotes || {};
        } catch (e) { console.warn('Quote service unreachable', e); }
        return {};
    },

    /** Check a ticker exists and get its price before saving it. */
    async checkTicker(ticker) {
        const res = await fetch(`${API_BASE_URL}/api/live-price?ticker=${encodeURIComponent(ticker)}`);
        if (!res.ok) return null;
        return await res.json();
    },

    /** Real-portfolio risk analysis + optimisation (backend /api/portfolio/analyze). */
    async analyzePortfolio(holdings, profile) {
        const res = await fetch(`${API_BASE_URL}/api/portfolio/analyze`, {
            method: 'POST', headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                profile,
                holdings: holdings.filter(h => (h.quantity || 0) > 0)
                                  .map(h => ({ ticker: h.ticker, quantity: h.quantity }))
            })
        });
        const data = await res.json();
        if (!res.ok) {
            const d = data.detail;
            throw new Error(typeof d === 'string' ? d : (d && d.message) || 'Analysis failed');
        }
        return data;
    },

    async updateQuantity(id, quantity) {
        const { error } = await supabase.from('portfolios').update({ quantity }).eq('id', id);
        if (error) throw error;
    },

    async saveAsset(asset) {
        const { data, error } = await supabase
            .from('portfolios')
            .insert([asset]);
        if (error) throw error;
        return data;
    },

    async deleteAsset(id) {
        const { error } = await supabase
            .from('portfolios')
            .delete()
            .eq('id', id);
        if (error) throw error;
    },

    /**
     * AI & Analytics
     */
    async fetchAIPredictions(ticker) {
        try {
            const res = await fetch(`${API_BASE_URL}/api/predict-risk?ticker=${ticker}`);
            if (res.ok) return await res.json();
        } catch (e) { console.warn('Forecast service unreachable', e); }
        return { error: 'Forecast service unreachable' };
    },

    async fetchMarketDiscover() {
        try {
            const res = await fetch(`${API_BASE_URL}/api/market-discover`);
            if (res.ok) return await res.json();
        } catch (e) { console.warn('Market movers unavailable', e); }
        return { discovery: [] };
    },

    async fetchNews() {
        try {
            const res = await fetch(`${API_BASE_URL}/api/market-news`);
            if (res.ok) return await res.json();
        } catch (e) { console.warn('News unavailable', e); }
        return { news: [] };
    },

    /**
     * History & Snapshots
     */
    async saveDailySnapshot(userId, totalVal) {
        const today = new Date().toISOString().split('T')[0];
        try {
            const { error } = await supabase.from('portfolio_history').upsert(
                [{ user_id: userId, total_val: totalVal, recorded_at: today }],
                { onConflict: 'user_id, recorded_at' }
            );
            return { success: !error, error };
        } catch (e) {
            return { success: false, error: e };
        }
    }
};
