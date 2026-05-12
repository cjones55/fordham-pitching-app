/**
 * Supabase Edge Function: stripe-webhook
 *
 * Receives Stripe webhook events and automatically upgrades / downgrades
 * CBB+ user tiers in the cbb_users table.
 *
 * Deploy with:
 *   supabase functions deploy stripe-webhook --no-verify-jwt
 *
 * Add to Stripe Dashboard → Webhooks → Endpoint URL:
 *   https://wsesdgxskuwgqaqvjivm.supabase.co/functions/v1/stripe-webhook
 *
 * Required Stripe events:
 *   - checkout.session.completed
 *   - customer.subscription.deleted
 *   - invoice.payment_failed
 *
 * Required Supabase secrets (set via: supabase secrets set KEY=value):
 *   STRIPE_WEBHOOK_SECRET   — from Stripe Dashboard → Webhooks → Signing secret
 *   SUPABASE_SERVICE_KEY    — from Supabase Dashboard → Settings → API → service_role key
 */

import { serve } from "https://deno.land/std@0.168.0/http/server.ts";
import Stripe from "https://esm.sh/stripe@13.3.0?target=deno";

const stripe = new Stripe(Deno.env.get("STRIPE_SECRET_KEY") ?? "", {
  apiVersion: "2023-10-16",
  httpClient: Stripe.createFetchHttpClient(),
});

const supabaseUrl    = Deno.env.get("SUPABASE_URL") ?? "";
const supabaseKey    = Deno.env.get("SUPABASE_SERVICE_KEY") ?? "";
const webhookSecret  = Deno.env.get("STRIPE_WEBHOOK_SECRET") ?? "";

// Plan duration in days (for one-time payments)
const PLAN_DAYS: Record<string, number> = {
  weekly:    7,
  monthly:   30,
  quarterly: 90,
  annual:    365,
};

function addDays(days: number): string {
  const d = new Date();
  d.setDate(d.getDate() + days);
  return d.toISOString().split("T")[0];
}

async function updateUser(username: string, patch: Record<string, string>) {
  const res = await fetch(
    `${supabaseUrl}/rest/v1/cbb_users?username=eq.${encodeURIComponent(username)}`,
    {
      method: "PATCH",
      headers: {
        "Content-Type": "application/json",
        "apikey": supabaseKey,
        "Authorization": `Bearer ${supabaseKey}`,
        "Prefer": "return=minimal",
      },
      body: JSON.stringify(patch),
    }
  );
  if (!res.ok) {
    const err = await res.text();
    console.error(`Failed to update user ${username}:`, err);
  }
}

serve(async (req) => {
  if (req.method !== "POST") {
    return new Response("Method not allowed", { status: 405 });
  }

  const body      = await req.text();
  const signature = req.headers.get("stripe-signature") ?? "";

  let event: Stripe.Event;
  try {
    event = await stripe.webhooks.constructEventAsync(body, signature, webhookSecret);
  } catch (err) {
    console.error("Webhook signature verification failed:", err);
    return new Response(`Webhook Error: ${err}`, { status: 400 });
  }

  console.log(`Received event: ${event.type}`);

  if (event.type === "checkout.session.completed") {
    const session   = event.data.object as Stripe.Checkout.Session;
    const username  = session.client_reference_id;

    if (!username) {
      console.warn("No client_reference_id — cannot identify user.");
      return new Response("OK", { status: 200 });
    }

    // Detect plan from amount / metadata if available
    const amount    = session.amount_total ?? 0;   // in cents
    let planDays    = 30;  // default monthly
    if (amount <= 399)       planDays = 7;    // weekly $3.99
    else if (amount <= 999)  planDays = 30;   // monthly $9.99
    else if (amount <= 2499) planDays = 90;   // quarterly $24.99
    else                     planDays = 365;  // annual $59.99

    const subEnd = addDays(planDays);
    const stripeSubId = typeof session.subscription === "string"
      ? session.subscription : "";

    await updateUser(username, {
      tier:              "pro",
      subscription_end:  subEnd,
      stripe_customer:   session.customer as string ?? "",
      stripe_sub_id:     stripeSubId,
    });
    console.log(`Upgraded ${username} to pro until ${subEnd}`);
  }

  else if (event.type === "customer.subscription.deleted") {
    // Find user by stripe_sub_id
    const sub      = event.data.object as Stripe.Subscription;
    const subId    = sub.id;
    const res      = await fetch(
      `${supabaseUrl}/rest/v1/cbb_users?stripe_sub_id=eq.${subId}&select=username`,
      {
        headers: {
          "apikey": supabaseKey,
          "Authorization": `Bearer ${supabaseKey}`,
        },
      }
    );
    const rows = await res.json();
    if (rows.length > 0) {
      const username = rows[0].username;
      await updateUser(username, { tier: "free", subscription_end: "" });
      console.log(`Downgraded ${username} — subscription cancelled`);
    }
  }

  else if (event.type === "invoice.payment_failed") {
    // Optional: log or notify — don't immediately downgrade (Stripe retries first)
    console.log("Payment failed — Stripe will retry. No action taken.");
  }

  return new Response("OK", { status: 200 });
});
