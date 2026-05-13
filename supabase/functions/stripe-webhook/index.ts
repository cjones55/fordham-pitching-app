/**
 * CBB+ Stripe Webhook — Supabase Edge Function
 * SUPABASE_URL and SUPABASE_SERVICE_ROLE_KEY are injected automatically.
 * Only set: STRIPE_SECRET_KEY and STRIPE_WEBHOOK_SECRET
 */
import { serve } from "https://deno.land/std@0.168.0/http/server.ts";
import Stripe from "https://esm.sh/stripe@13.3.0?target=deno";

const stripe = new Stripe(Deno.env.get("STRIPE_SECRET_KEY") ?? "", {
  apiVersion: "2023-10-16",
  httpClient: Stripe.createFetchHttpClient(),
});

const supabaseUrl = Deno.env.get("SUPABASE_URL") ?? "";
const supabaseKey = Deno.env.get("SUPABASE_SERVICE_ROLE_KEY") ?? "";
const webhookSecret = Deno.env.get("STRIPE_WEBHOOK_SECRET") ?? "";

function addDays(days: number): string {
  const d = new Date();
  d.setDate(d.getDate() + days);
  return d.toISOString().split("T")[0];
}

async function updateUser(username: string, patch: Record<string, string>) {
  await fetch(
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
}

serve(async (req) => {
  if (req.method !== "POST") return new Response("Method not allowed", { status: 405 });

  const body      = await req.text();
  const signature = req.headers.get("stripe-signature") ?? "";

  let event: Stripe.Event;
  try {
    event = await stripe.webhooks.constructEventAsync(body, signature, webhookSecret);
  } catch (err) {
    return new Response(`Webhook Error: ${err}`, { status: 400 });
  }

  if (event.type === "checkout.session.completed") {
    const session  = event.data.object as Stripe.Checkout.Session;
    const username = session.client_reference_id;
    if (!username) return new Response("OK", { status: 200 });

    const amount = session.amount_total ?? 0;
    let days = 30;
    if (amount <= 399)       days = 7;
    else if (amount <= 999)  days = 30;
    else if (amount <= 2499) days = 90;
    else                     days = 365;

    await updateUser(username, {
      tier:             "pro",
      subscription_end: addDays(days),
      stripe_customer:  session.customer as string ?? "",
      stripe_sub_id:    typeof session.subscription === "string" ? session.subscription : "",
    });
  }

  else if (event.type === "customer.subscription.deleted") {
    const sub = event.data.object as Stripe.Subscription;
    const res = await fetch(
      `${supabaseUrl}/rest/v1/cbb_users?stripe_sub_id=eq.${sub.id}&select=username`,
      { headers: { "apikey": supabaseKey, "Authorization": `Bearer ${supabaseKey}` } }
    );
    const rows = await res.json();
    if (rows.length > 0) {
      await updateUser(rows[0].username, { tier: "free", subscription_end: "" });
    }
  }

  return new Response("OK", { status: 200 });
});
