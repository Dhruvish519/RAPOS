// Supabase Configuration for RAPOS
const SUPABASE_URL = "https://pskjlckkhzfsxqpcpytk.supabase.co";
const SUPABASE_ANON_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6InBza2psY2traHpmc3hxcGNweXRrIiwicm9sZSI6ImFub24iLCJpYXQiOjE3OTMxMDIwNzcsImV4cCI6MjEwODY3ODA3N30.NkPDedKMDH0OCgbut3vZii6eiLl6BSgOYKwiodjJSh4";

// Backend API URL: set once in js/rapos-config.js (loaded before this file).
// `var`, not `const`: js/api.js declares it too, and a second `const` in the
// same page is a SyntaxError that stops the whole script from loading.
var API_BASE_URL = window.RAPOS_API_BASE || "https://rapos-api.onrender.com";

// Initialize the Supabase Client (avoiding variable shadowing)
const supabaseClient = supabase.createClient(SUPABASE_URL, SUPABASE_ANON_KEY);

// Make the client available globally for the rest of the application
window.supabase = supabaseClient;

console.log("RAPOS: Supabase initialized. API Base:", API_BASE_URL);
