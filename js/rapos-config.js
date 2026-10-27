/**
 * RAPOS backend URL: the ONE place to change it.
 *
 * PROD_API is the deployed FastAPI service (Render). After deploying the
 * backend, paste its URL here and push; every page picks it up.
 * - Pages opened from localhost / file:// use a local backend on port 8000
 *   (`uvicorn main:app --reload` inside backend/).
 * - Override any time with ?api=https://your-backend in the page URL.
 */
(function () {
    var PROD_API = 'https://rapos-api.onrender.com';
    var qp = new URLSearchParams(window.location.search).get('api');
    var local = ['localhost', '127.0.0.1', ''].indexOf(window.location.hostname) !== -1;
    window.RAPOS_API_BASE = (qp || (local ? 'http://localhost:8000' : PROD_API)).replace(/\/$/, '');
})();
