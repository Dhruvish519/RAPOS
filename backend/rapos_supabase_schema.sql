-- =========================================
-- RAPOS: Supabase Database Setup
-- Run this in your Supabase SQL Editor
-- =========================================

-- 1. Create Portfolios Table
CREATE TABLE IF NOT EXISTS public.portfolios (
    id UUID DEFAULT gen_random_uuid() PRIMARY KEY,
    user_id UUID REFERENCES auth.users(id) ON DELETE CASCADE NOT NULL,
    asset_name TEXT NOT NULL,
    ticker TEXT NOT NULL,
    quantity DECIMAL(12,4) DEFAULT 1.0, -- Added for live weighted tracking
    value DECIMAL(12,2) NOT NULL DEFAULT 0,
    performance_pct DECIMAL(5,2) DEFAULT 0,
    asset_type TEXT DEFAULT 'Stocks',
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- 2. Enable Row Level Security (RLS)
ALTER TABLE public.portfolios ENABLE ROW LEVEL SECURITY;

-- 3. Create Security Policies
-- Users can only see their own portfolios
CREATE POLICY "Users can view their own portfolios" 
ON public.portfolios FOR SELECT 
USING (auth.uid() = user_id);

-- Users can only insert their own portfolios
CREATE POLICY "Users can insert their own portfolios" 
ON public.portfolios FOR INSERT 
WITH CHECK (auth.uid() = user_id);

-- Users can only update their own portfolios
CREATE POLICY "Users can update their own portfolios" 
ON public.portfolios FOR UPDATE 
USING (auth.uid() = user_id);

-- Users can only delete their own portfolios (the dashboard's remove-asset button)
CREATE POLICY "Users can delete their own portfolios"
ON public.portfolios FOR DELETE
USING (auth.uid() = user_id);

-- 4. Daily portfolio value snapshots (js/api.js saveDailySnapshot upserts here)
CREATE TABLE IF NOT EXISTS public.portfolio_history (
    id UUID DEFAULT gen_random_uuid() PRIMARY KEY,
    user_id UUID REFERENCES auth.users(id) ON DELETE CASCADE NOT NULL,
    total_val DECIMAL(14,2) NOT NULL DEFAULT 0,
    recorded_at DATE NOT NULL DEFAULT CURRENT_DATE,
    UNIQUE (user_id, recorded_at)
);

ALTER TABLE public.portfolio_history ENABLE ROW LEVEL SECURITY;

CREATE POLICY "Users can view their own history"
ON public.portfolio_history FOR SELECT
USING (auth.uid() = user_id);

CREATE POLICY "Users can insert their own history"
ON public.portfolio_history FOR INSERT
WITH CHECK (auth.uid() = user_id);

CREATE POLICY "Users can update their own history"
ON public.portfolio_history FOR UPDATE
USING (auth.uid() = user_id);

-- 5. Seed some sample data (Optional)
-- Note: You can add sample data through the Table Editor in Supabase
