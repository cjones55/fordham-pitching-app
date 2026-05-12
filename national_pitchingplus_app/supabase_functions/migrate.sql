-- Run this in Supabase SQL Editor to add subscription tracking columns
ALTER TABLE cbb_users
  ADD COLUMN IF NOT EXISTS subscription_end  TEXT DEFAULT '',
  ADD COLUMN IF NOT EXISTS stripe_customer   TEXT DEFAULT '',
  ADD COLUMN IF NOT EXISTS stripe_sub_id     TEXT DEFAULT '',
  ADD COLUMN IF NOT EXISTS trial_expires     TEXT DEFAULT '';
