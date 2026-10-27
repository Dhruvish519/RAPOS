/**
 * Backend URL for the RAPOS AI service.
 * - Running the frontend locally (localhost / file://) talks to a local backend
 *   on port 8000 (`uvicorn main:app --reload` inside backend/).
 * - Otherwise the deployed Render service is used.
 * - Override any time with ?api=https://your-backend in the page URL.
 */
(function () {
    const qp = new URLSearchParams(window.location.search).get('api');
    const local = ['localhost', '127.0.0.1', ''].includes(window.location.hostname);
    window.RAPOS_API_BASE = (qp || (local ? 'http://localhost:8000' : 'https://rapos-backend.onrender.com')).replace(/\/$/, '');
})();
