/**
 * RAPOS UI Engine
 * Orchestrates all DOM updates, chart renderings, and visual state.
 */

const RAPOS_UI = {
    state: {
        holdings: [],
        allocationChart: null,
        targetChart: null,
        currentUser: null,
        activeTab: 'portfolioTab'
    },

    /**
     * Initialization
     */
    init(user) {
        this.state.currentUser = user;
        this.initCharts();
        this.setupEventListeners();
    },

    initCharts() {
        const allocCtx = document.getElementById('allocationChart');
        if (allocCtx) {
            this.state.allocationChart = new Chart(allocCtx, {
                type: 'doughnut',
                data: {
                    labels: [],
                    datasets: [{
                        data: [],
                        backgroundColor: ['#3b82f6', '#6366f1', '#10b981', '#f59e0b', '#ef4444'],
                        borderWidth: 0,
                        hoverOffset: 15
                    }]
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    cutout: '75%',
                    plugins: {
                        legend: { position: 'bottom', labels: { usePointStyle: true, padding: 20, font: { family: 'Inter', weight: 600 } } }
                    }
                }
            });
        }
    },

    setupEventListeners() {
        // Click outside modal to close
        const overlay = document.getElementById('modalOverlay');
        if (overlay) {
            overlay.addEventListener('click', (e) => {
                if (e.target === overlay) this.closeModal();
            });
        }
        // ESC key closes modal
        document.addEventListener('keydown', (e) => {
            if (e.key === 'Escape') this.closeModal();
        });
    },

    /**
     * View Controllers
     */
    switchTab(tabId, element) {
        // Update active class on sidebar items
        document.querySelectorAll('.sidebar-item').forEach(i => i.classList.remove('active'));
        if (element) {
            element.classList.add('active');
        } else {
            // Find sidebar-item with matching onclick
            const matching = Array.from(document.querySelectorAll('.sidebar-item')).find(el => {
                const attr = el.getAttribute('onclick') || '';
                return attr.includes(tabId);
            });
            if (matching) matching.classList.add('active');
        }

        // Hide all views instantly
        const views = document.querySelectorAll('.tab-view');
        views.forEach(v => {
            v.style.display = 'none';
            v.classList.remove('active');
        });

        // Show active view
        const activeView = document.getElementById(tabId);
        if (activeView) {
            activeView.style.display = 'block';
            activeView.classList.add('active');
            activeView.style.opacity = '1';
        }
        this.state.activeTab = tabId;

        // Tab-specific initializations
        if (tabId === 'marketTab') {
            this.initTradingView();
        }
    },

    /**
     * Portfolio Rendering
     */
    updatePortfolio(holdings) {
        this.state.holdings = holdings || [];
        this.renderHoldingsList();
        this.updateStats();
        this.updateCharts();
    },

    renderHoldingsList() {
        const list = document.getElementById('holdingsList');
        if (!list) return;

        if (this.state.holdings.length === 0) {
            list.innerHTML = `
                <div style="text-align:center; padding:60px; color:var(--slate-400);">
                    <i class="fas fa-folder-open" style="font-size:32px; margin-bottom:16px; opacity:0.5;"></i>
                    <p style="font-weight:600; font-size:16px; color:var(--slate-700);">No assets currently registered.</p>
                    <p style="font-size:13px; margin-top:4px;">Click <b>+ New Investment</b> above to track your first asset.</p>
                </div>`;
            return;
        }

        list.innerHTML = this.state.holdings.map(h => `
            <div class="list-item animate-fade-in">
                <div style="display:flex; align-items:center; gap:16px;">
                    <div style="width:48px; height:48px; background:var(--slate-100); border-radius:12px; display:flex; align-items:center; justify-content:center; font-weight:800; color:var(--brand-primary); font-size:12px;">${(h.ticker || 'AST').substring(0,4)}</div>
                    <div>
                        <div style="font-weight:700; color:var(--slate-900);">${h.asset_name || h.ticker}</div>
                        <div style="font-size:12px; color:var(--slate-500);">${h.quantity || 1} units @ $${(h.current_price || 0).toLocaleString()}</div>
                    </div>
                </div>
                <div style="display:flex; align-items:center; gap:24px;">
                    <div style="text-align:right;">
                        <div style="font-weight:800; font-size:16px; color:var(--slate-900);">$${(h.live_value || 0).toLocaleString()}</div>
                        <div class="stat-trend ${(h.performance_pct || 0) >= 0 ? 'trend-up' : 'trend-down'}">
                            ${(h.performance_pct || 0) >= 0 ? '+' : ''}${h.performance_pct || 0}%
                        </div>
                    </div>
                    <button onclick="handleDeleteAsset('${h.id}')" style="background:none; border:none; color:var(--slate-300); cursor:pointer; font-size:18px; transition:color 0.2s;" title="Delete Asset">
                        <i class="fas fa-trash-alt"></i>
                    </button>
                </div>
            </div>
        `).join('');
    },

    updateStats() {
        let totalVal = 0, totalPrevVal = 0;
        this.state.holdings.forEach(h => {
            const val = h.live_value || 0;
            totalVal += val;
            const prevPrice = (h.current_price || 0) / (1 + (h.performance_pct || 0) / 100);
            totalPrevVal += (h.quantity || 1) * prevPrice;
        });

        const totalGain = totalVal - totalPrevVal;
        const velocity = totalPrevVal > 0 ? (totalGain / totalPrevVal) * 100 : 0;

        const valEl = document.getElementById('portfolioValue');
        if (valEl) valEl.textContent = '$' + totalVal.toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 });

        const changeEl = document.getElementById('portfolioDailyChange');
        if (changeEl) {
            changeEl.textContent = `${totalGain >= 0 ? '+' : ''}$${totalGain.toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 })} (${velocity.toFixed(2)}%)`;
            changeEl.className = `stat-trend ${totalGain >= 0 ? 'trend-up' : 'trend-down'}`;
        }

        const countEl = document.getElementById('assetCount');
        if (countEl) countEl.textContent = this.state.holdings.length;
    },

    updateCharts() {
        if (this.state.allocationChart) {
            const groups = {};
            this.state.holdings.forEach(h => { 
                const type = h.asset_type || 'Others';
                groups[type] = (groups[type] || 0) + (h.live_value || 0); 
            });
            const labels = Object.keys(groups);
            const data = Object.values(groups);

            if (labels.length === 0) {
                this.state.allocationChart.data.labels = ['Empty Portfolio'];
                this.state.allocationChart.data.datasets[0].data = [1];
                this.state.allocationChart.data.datasets[0].backgroundColor = ['#e2e8f0'];
            } else {
                this.state.allocationChart.data.labels = labels;
                this.state.allocationChart.data.datasets[0].data = data;
                this.state.allocationChart.data.datasets[0].backgroundColor = ['#3b82f6', '#6366f1', '#10b981', '#f59e0b', '#ef4444'];
            }
            this.state.allocationChart.update();
        }
    },

    /**
     * Intelligence Hub Rendering
     */
    renderAIPrediction(data) {
        if (!data) return;
        const pClose = document.getElementById('predictedCloseValue');
        const pScore = document.getElementById('aiRiskScoreValue');
        const pConf  = document.getElementById('modelConfidence');
        const pSent  = document.getElementById('predictedSentiment');

        if (data.error || data.predicted_next_close == null) {
            // No model output for this asset: say so rather than show a number
            if (pClose) pClose.textContent = 'n/a';
            if (pScore) pScore.textContent = '---';
            if (pConf)  pConf.textContent  = '--';
            if (pSent)  { pSent.textContent = 'No data'; pSent.className = 'stat-trend'; }
            return;
        }
        if (pClose) pClose.textContent = '$' + data.predicted_next_close;
        if (pScore) pScore.textContent = data.ai_risk_score;
        if (pConf)  pConf.textContent  = '$' + data.last_close_price;
        if (pSent) {
            const isBull = (data.predicted_daily_return_pct || 0) >= 0;
            pSent.textContent = isBull ? 'Bullish' : 'Bearish';
            pSent.className = `stat-trend ${isBull ? 'trend-up' : 'trend-down'}`;
            pSent.style.background = isBull ? 'var(--success-bg)' : 'var(--danger-bg)';
            pSent.style.padding = '4px 12px';
            pSent.style.borderRadius = '8px';
        }
    },

    renderDiscovery(items) {
        const div = document.getElementById('discoveryList');
        if (!div || !items) return;
        div.innerHTML = items.map(item => `
            <div class="list-item" style="padding:12px 16px; cursor:pointer;" onclick="document.getElementById('tickerSelector').value='${item.ticker}'; fetchAIPredictions('${item.ticker}');">
                <div>
                    <div style="font-weight:800; font-size:15px; color:var(--slate-900);">${item.ticker}</div>
                    <div style="font-size:12px; color:var(--slate-500);">${item.name || ''}</div>
                </div>
                <div class="stat-trend ${item.change >= 0 ? 'trend-up' : 'trend-down'}" style="font-weight:800; font-size:14px;">
                    ${item.change >= 0 ? '+' : ''}${item.change}%
                </div>
            </div>
        `).join('');
    },

    renderNews(news) {
        const div = document.getElementById('newsFeed');
        if (!div || !news) return;
        div.innerHTML = news.map(item => `
            <div style="padding:20px 24px; border-bottom:1px solid var(--slate-100); cursor:pointer; transition:background 0.2s;" onclick="window.open('${item.url}', '_blank')">
                <div style="font-size:14px; font-weight:700; color:var(--slate-800); margin-bottom:6px; line-height:1.4;">${item.headline}</div>
                <div style="display:flex; justify-content:space-between; align-items:center;">
                    <span style="font-size:11px; color:var(--slate-400); font-weight:700; text-transform:uppercase;">${item.source}</span>
                    <span style="font-size:11px; color:var(--slate-400);">${new Date(item.datetime * 1000).toLocaleTimeString([], {hour: '2-digit', minute:'2-digit'})}</span>
                </div>
            </div>
        `).join('');
    },

    /**
     * Overlay Systems
     */
    showModal(title, body, confirmText, onConfirm) {
        document.getElementById('modalTitle').textContent = title;
        document.getElementById('modalBody').innerHTML = body;
        const btn = document.getElementById('modalConfirmBtn');
        if (btn) {
            btn.textContent = confirmText || 'Confirm';
            btn.onclick = onConfirm;
        }
        const overlay = document.getElementById('modalOverlay');
        if (overlay) {
            overlay.style.display = 'flex';
            overlay.classList.add('active');
        }
    },

    closeModal() {
        const overlay = document.getElementById('modalOverlay');
        if (overlay) {
            overlay.style.display = 'none';
            overlay.classList.remove('active');
        }
    },

    showToast(message, type = 'info') {
        const toast = document.getElementById('toast');
        const msg = document.getElementById('toastMsg');
        if (!toast || !msg) return;

        msg.textContent = message;
        toast.style.display = 'flex';
        toast.style.opacity = '1';

        setTimeout(() => {
            toast.style.opacity = '0';
            setTimeout(() => { toast.style.display = 'none'; }, 300);
        }, 3000);
    },

    /**
     * Market Visuals
     */
    initTradingView() {
        const container = document.getElementById('tradingview_widget');
        if (!container) return;

        // Only initialize once
        if (container.dataset.loaded === 'true') return;

        if (typeof TradingView !== 'undefined') {
            container.innerHTML = '';
            new TradingView.widget({
                "autosize": true,
                "symbol": "NASDAQ:AAPL",
                "interval": "D",
                "timezone": "Etc/UTC",
                "theme": "light",
                "style": "1",
                "locale": "en",
                "toolbar_bg": "#f1f5f9",
                "enable_publishing": false,
                "allow_symbol_change": true,
                "container_id": "tradingview_widget"
            });
            container.dataset.loaded = 'true';
        } else {
            // High-reliability embed fallback
            container.innerHTML = `
                <iframe 
                    src="https://s.tradingview.com/widgetembed/?symbol=NASDAQ%3AAAPL&interval=D&symboledit=1&saveimage=0&toolbarbg=f1f5f9&studies=%5B%5D&theme=light&style=1&timezone=Etc%2FUTC" 
                    style="width:100%; height:100%; border:none; border-radius:12px;"
                    title="TradingView Chart">
                </iframe>
            `;
            container.dataset.loaded = 'true';
        }
    }
};
