/**
 * Searchable instrument picker (company name or ticker -> listing).
 * Instant matches from js/symbols.js, then live results from the backend
 * (/api/search-stocks: US stocks/ETFs, Indian NSE/BSE stocks, major crypto).
 *
 *   RAPOS_attachSymbolSearch(inputEl, listEl, item => { ... })
 *   item = { symbol, name, type: "Stock" | "ETF" | "Crypto", exchange?, currency? }
 */
(function () {
    const TYPE = { Stock: "Stock", ETF: "ETF / Fund", Crypto: "Crypto" };
    const esc = t => String(t).replace(/[&<>"]/g, c => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));
    const api = () => window.RAPOS_API_BASE || "";

    window.RAPOS_symbolLabel = it => {
        const where = it.exchange || (it.symbol.endsWith(".NS") ? "NSE" : it.symbol.endsWith(".BO") ? "BSE" : "");
        return `${TYPE[it.type] || it.type}${where ? " · " + where : ""}`;
    };
    window.RAPOS_currencySymbol = cur => (cur === "INR" ? "₹" : "$");
    window.RAPOS_currencyOf = sym => (/\.(NS|BO)$/i.test(sym || "") ? "INR" : "USD");

    function localMatches(q) {
        q = q.toLowerCase();
        return (window.RAPOS_SYMBOLS || [])
            .filter(it => it.symbol.toLowerCase().startsWith(q) || it.name.toLowerCase().includes(q))
            .sort((a, b) => (b.symbol.toLowerCase().replace(/\.(ns|bo)$/, "") === q) - (a.symbol.toLowerCase().replace(/\.(ns|bo)$/, "") === q))
            .slice(0, 8);
    }

    window.RAPOS_attachSymbolSearch = function (input, list, onPick) {
        let timer = null, seq = 0, items = [];
        list.style.cssText += ";display:none; position:absolute; left:0; right:0; top:100%; margin-top:4px; background:white; border:1px solid var(--slate-200); border-radius:12px; box-shadow:0 12px 30px rgba(15,23,42,.12); max-height:280px; overflow-y:auto; z-index:50;";

        function render(found, note) {
            items = found;
            if (!found.length) {
                list.innerHTML = `<div style="padding:12px 14px; font-size:13px; color:var(--slate-400);">${note || "No matches"}</div>`;
            } else {
                list.innerHTML = found.map((it, i) => `<div class="sym-opt" data-i="${i}" style="padding:10px 14px; cursor:pointer; display:flex; justify-content:space-between; gap:12px; border-bottom:1px solid var(--slate-100);">
                    <div><div style="font-weight:700; color:var(--slate-900);">${esc(it.name)}</div><div style="font-size:12px; color:var(--slate-500);">${esc(it.symbol)}</div></div>
                    <span style="font-size:11px; font-weight:700; color:var(--brand-primary); align-self:center; white-space:nowrap;">${esc(RAPOS_symbolLabel(it))}</span></div>`).join("") +
                    (note ? `<div style="padding:8px 14px; font-size:12px; color:var(--slate-400);">${note}</div>` : "");
            }
            list.style.display = "";
            list.querySelectorAll(".sym-opt").forEach(el => {
                el.onmouseenter = () => el.style.background = "var(--slate-50)";
                el.onmouseleave = () => el.style.background = "";
                el.onmousedown = e => { e.preventDefault(); choose(items[+el.dataset.i]); };
            });
        }
        function choose(it) {
            list.style.display = "none";
            input.value = `${it.name} (${it.symbol})`;
            onPick(it);
        }
        input.addEventListener("input", () => {
            const q = input.value.trim();
            if (!q) { list.style.display = "none"; return; }
            const local = localMatches(q);
            render(local, "Searching US, Indian (NSE/BSE) and crypto markets...");
            clearTimeout(timer);
            const my = ++seq;
            timer = setTimeout(async () => {
                let remote = [];
                try {
                    const r = await fetch(`${api()}/api/search-stocks?q=${encodeURIComponent(q)}`);
                    if (r.ok) remote = (await r.json()).results || [];
                } catch (e) { /* offline: keep local matches */ }
                if (my !== seq) return;                       // a newer search is running
                const seen = new Set(local.map(i => i.symbol));
                const all = [...local, ...remote.filter(i => !seen.has(i.symbol))].slice(0, 12);
                render(all, all.length ? "" : "No matches. Try the company name or ticker (e.g. Infosys, TCS.NS).");
            }, 250);
        });
        input.addEventListener("focus", () => { if (input.value.trim() && items.length) list.style.display = ""; });
        input.addEventListener("blur", () => setTimeout(() => list.style.display = "none", 150));
        input.addEventListener("keydown", e => {
            if (e.key === "Enter") { e.preventDefault(); if (items[0]) choose(items[0]); }
            if (e.key === "Escape") list.style.display = "none";
        });
    };
})();
