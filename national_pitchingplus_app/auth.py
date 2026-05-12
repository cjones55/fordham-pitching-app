"""
auth.py — CBB+ user authentication, profiles, and subscription management.
Backend: Supabase (cloud) when credentials are in st.secrets,
         file-based YAML/JSON fallback for local dev / first deploy.
"""
from __future__ import annotations

import json
from datetime import date
from pathlib import Path

import bcrypt
import streamlit as st
import yaml

# ── Supabase backend (persistent across deploys) ──────────────────────────────
def _supabase_client():
    """Return a Supabase client if credentials are configured, else None."""
    try:
        url   = st.secrets["supabase"]["url"]
        key   = st.secrets["supabase"]["key"]
        from supabase import create_client
        return create_client(url, key)
    except KeyError:
        return None  # secrets not configured — use file backend
    except Exception as e:
        st.warning(f"Supabase connection error: {e}")
        return None


def _sb_load_users() -> dict:
    sb = _supabase_client()
    if sb is None:
        return None
    try:
        rows = sb.table("cbb_users").select("*").execute().data or []
        return {"users": {r["username"]: r for r in rows}}
    except Exception:
        return None


def _sb_save_user(username: str, record: dict) -> bool:
    sb = _supabase_client()
    if sb is None:
        return False
    try:
        sb.table("cbb_users").upsert({"username": username, **record}).execute()
        return True
    except Exception as e:
        st.error(f"Supabase error: {e}")
        return False


def _sb_load_profile(username: str) -> dict | None:
    sb = _supabase_client()
    if sb is None:
        return None
    try:
        rows = sb.table("cbb_profiles").select("*").eq("username", username).execute().data
        if rows:
            r = rows[0]
            return {
                "favorite_teams":   json.loads(r.get("favorite_teams",  "[]")),
                "favorite_players": json.loads(r.get("favorite_players", "[]")),
            }
        return {"favorite_teams": [], "favorite_players": []}
    except Exception:
        return None


def _sb_save_profile(username: str, profile: dict) -> bool:
    sb = _supabase_client()
    if sb is None:
        return False
    try:
        sb.table("cbb_profiles").upsert({
            "username":        username,
            "favorite_teams":  json.dumps(profile.get("favorite_teams", [])),
            "favorite_players": json.dumps(profile.get("favorite_players", [])),
        }).execute()
        return True
    except Exception:
        return False

_DIR = Path(__file__).resolve().parent
USERS_FILE    = _DIR / "users_db.yaml"
PROFILES_FILE = _DIR / "user_profiles.json"

# ── Subscription config ───────────────────────────────────────────────────────
# Set to True when you're ready to start charging.
# While False every logged-in user gets full access regardless of tier.
SUBSCRIPTIONS_ENFORCED = False

# Usernames that always have admin rights (add yours here after registering)
ADMIN_USERNAMES: list[str] = ["cjones55"]

# Stripe Payment Link URLs — config ID: pmc_1TSa8C2MXoJPAlC5U34rHkSQ
# Currently all pointing to the same test link for QA.
# Create separate links per plan in Stripe dashboard when going live.
STRIPE_LINKS = {
    "weekly":    "https://buy.stripe.com/test_00wcMY0blc012jw6qt8k800",
    "monthly":   "https://buy.stripe.com/test_00wcMY0blc012jw6qt8k800",
    "quarterly": "https://buy.stripe.com/test_7sYaEQ5vFaVXe2ecOR8k802",
    "annual":    "https://buy.stripe.com/test_3cI5kwgaj7JL9LY0258k801",
}

PRICING = {
    "weekly":    {"label": "Weekly",    "price": "$3.99",  "period": "/ week",
                  "per_month": "≈ $17/mo",          "badge": ""},
    "monthly":   {"label": "Monthly",   "price": "$9.99",  "period": "/ month",
                  "per_month": "",                  "badge": "MOST POPULAR"},
    "quarterly": {"label": "Quarterly", "price": "$24.99", "period": "/ 3 months",
                  "per_month": "≈ $8.33/mo — save 17%",   "badge": ""},
    "annual":    {"label": "Annual",    "price": "$59.99", "period": "/ year",
                  "per_month": "≈ $5.00/mo — save 50%",   "badge": "BEST VALUE"},
}

PRO_FEATURES = [
    "Unlimited pitcher & hitter reports",
    "Full national scouting database",
    "Stuff+ and Loc+ models",
    "Postgame and season graphics",
    "D1 leaderboards and HR distance",
    "Player percentile cards",
    "Favorite teams & players",
    "Early access to new features",
]


# ── Low-level file I/O ────────────────────────────────────────────────────────

def _load_users() -> dict:
    if not USERS_FILE.exists():
        return {"users": {}}
    with open(USERS_FILE) as f:
        data = yaml.safe_load(f) or {}
    return data if "users" in data else {"users": {}}


def _save_users(data: dict) -> None:
    with open(USERS_FILE, "w") as f:
        yaml.dump(data, f, default_flow_style=False)


def _load_profiles() -> dict:
    if not PROFILES_FILE.exists():
        return {}
    with open(PROFILES_FILE) as f:
        return json.load(f)


def _save_profiles(data: dict) -> None:
    with open(PROFILES_FILE, "w") as f:
        json.dump(data, f, indent=2)


# ── Auth operations ───────────────────────────────────────────────────────────

def _hash(password: str) -> str:
    return bcrypt.hashpw(password.encode(), bcrypt.gensalt()).decode()


def _verify(password: str, hashed: str) -> bool:
    try:
        return bcrypt.checkpw(password.encode(), hashed.encode())
    except Exception:
        return False


def register(username: str, email: str, password: str) -> tuple[bool, str]:
    username = username.strip().lower()
    email    = email.strip().lower()
    if not username or len(username) < 3:
        return False, "Username must be at least 3 characters."
    if not email or "@" not in email:
        return False, "Enter a valid email address."
    if len(password) < 6:
        return False, "Password must be at least 6 characters."

    role   = "admin" if username in ADMIN_USERNAMES else "user"
    record = {
        "email":    email,
        "password": _hash(password),
        "joined":   str(date.today()),
        "role":     role,
        "tier":     "pro" if role == "admin" else "free",
    }

    # ── Try Supabase first ────────────────────────────────────────────────────
    sb_data = _sb_load_users()
    if sb_data is not None:
        if username in sb_data["users"]:
            return False, "Username already taken."
        if email in {u.get("email","") for u in sb_data["users"].values()}:
            return False, "An account with that email already exists."
        if not _sb_save_user(username, record):
            return False, "Could not save account. Please try again."
        _sb_save_profile(username, {"favorite_teams": [], "favorite_players": []})
        return True, "Account created successfully."

    # ── File fallback ─────────────────────────────────────────────────────────
    data = _load_users()
    if username in data["users"]:
        return False, "Username already taken."
    if email in {u["email"] for u in data["users"].values()}:
        return False, "An account with that email already exists."
    data["users"][username] = record
    _save_users(data)
    profiles = _load_profiles()
    profiles[username] = {"favorite_teams": [], "favorite_players": []}
    _save_profiles(profiles)
    return True, "Account created successfully."


def login(username: str, password: str) -> tuple[bool, str]:
    username = username.strip().lower()

    # ── Try Supabase first ────────────────────────────────────────────────────
    sb_data = _sb_load_users()
    if sb_data is not None:
        user = sb_data["users"].get(username)
        if user is None:
            return False, "Username not found."
        if not _verify(password, user["password"]):
            return False, "Incorrect password."
        if username in ADMIN_USERNAMES and user.get("role") != "admin":
            user["role"] = "admin"; user["tier"] = "pro"
            _sb_save_user(username, user)
        st.session_state["cbb_user"]      = username
        st.session_state["cbb_user_info"] = user
        return True, "Logged in."

    # ── File fallback ─────────────────────────────────────────────────────────
    data = _load_users()
    user = data["users"].get(username)
    if user is None:
        return False, "Username not found."
    if not _verify(password, user["password"]):
        return False, "Incorrect password."
    if username in ADMIN_USERNAMES and user.get("role") != "admin":
        data["users"][username]["role"] = "admin"
        data["users"][username]["tier"] = "pro"
        _save_users(data)
        user = data["users"][username]
    st.session_state["cbb_user"]      = username
    st.session_state["cbb_user_info"] = user
    return True, "Logged in."


def logout() -> None:
    for k in ["cbb_user", "cbb_user_info"]:
        st.session_state.pop(k, None)


def current_user() -> str | None:
    return st.session_state.get("cbb_user")


def current_user_info() -> dict:
    return st.session_state.get("cbb_user_info", {})


def is_logged_in() -> bool:
    return bool(current_user())


def is_admin() -> bool:
    info = current_user_info()
    return info.get("role") == "admin" or current_user() in ADMIN_USERNAMES


def has_pro_access() -> bool:
    """Returns True if the user can access pro features.
    When SUBSCRIPTIONS_ENFORCED is False everyone has access."""
    if not SUBSCRIPTIONS_ENFORCED:
        return True
    info = current_user_info()
    return info.get("tier") in ("pro",) or is_admin()


# ── Admin operations ──────────────────────────────────────────────────────────

def set_user_tier(username: str, tier: str) -> None:
    sb_data = _sb_load_users()
    if sb_data is not None:
        user = sb_data["users"].get(username, {})
        user["tier"] = tier
        _sb_save_user(username, user)
    else:
        data = _load_users()
        if username in data["users"]:
            data["users"][username]["tier"] = tier
            _save_users(data)
    if username == current_user():
        st.session_state["cbb_user_info"]["tier"] = tier


def all_users() -> list[dict]:
    sb_data = _sb_load_users()
    if sb_data is not None:
        return [
            {"username": u, **{k: v for k, v in info.items() if k != "password"}}
            for u, info in sb_data["users"].items()
        ]
    data = _load_users()
    return [
        {"username": u, **{k: v for k, v in info.items() if k != "password"}}
        for u, info in data["users"].items()
    ]


# ── Profile operations ────────────────────────────────────────────────────────

def get_profile(username: str) -> dict:
    sb = _sb_load_profile(username)
    if sb is not None:
        return sb
    return _load_profiles().get(username, {"favorite_teams": [], "favorite_players": []})


def save_profile(username: str, profile: dict) -> None:
    if not _sb_save_profile(username, profile):
        profiles = _load_profiles()
        profiles[username] = profile
        _save_profiles(profiles)


# ── UI helpers ────────────────────────────────────────────────────────────────

def _field(label, key, type_="default", placeholder=""):
    return st.text_input(label, key=key, type=type_,
                         placeholder=placeholder, label_visibility="collapsed")


def render_auth_page() -> None:
    st.markdown("""
    <style>
    .auth-title{font-size:2.2rem;font-weight:900;color:#F7F2E8;
                text-align:center;letter-spacing:-.01em;margin-bottom:.2rem}
    .auth-sub  {font-size:.95rem;color:#9BAABF;text-align:center;margin-bottom:1.8rem}
    .auth-label{font-size:.72rem;color:#9BAABF;font-weight:700;
                letter-spacing:.08em;text-transform:uppercase;margin-bottom:.2rem}
    </style>
    """, unsafe_allow_html=True)

    _, col, _ = st.columns([1, 2, 1])
    with col:
        st.markdown('<div class="auth-title">CBB+</div>', unsafe_allow_html=True)
        st.markdown(
            '<div class="auth-sub">College Baseball Analytics</div>',
            unsafe_allow_html=True)

        if not SUBSCRIPTIONS_ENFORCED:
            st.success("🎉 Free access for all members right now — sign up and explore everything.")

        tab_in, tab_up = st.tabs(["Sign In", "Create Account"])

        with tab_in:
            st.markdown('<div class="auth-label">Username</div>', unsafe_allow_html=True)
            li_user = _field("Username", "li_user", placeholder="your username")
            st.markdown('<div class="auth-label">Password</div>', unsafe_allow_html=True)
            li_pass = _field("Password", "li_pass", type_="password", placeholder="••••••••")
            if st.button("Sign In", use_container_width=True, key="li_btn", type="primary"):
                ok, msg = login(li_user, li_pass)
                if ok:
                    st.rerun()
                else:
                    st.error(msg)

        with tab_up:
            st.markdown('<div class="auth-label">Username</div>', unsafe_allow_html=True)
            su_user  = _field("Username",  "su_user",  placeholder="choose a username")
            st.markdown('<div class="auth-label">Email</div>', unsafe_allow_html=True)
            su_email = _field("Email",     "su_email", placeholder="you@example.com")
            st.markdown('<div class="auth-label">Password</div>', unsafe_allow_html=True)
            su_pass  = _field("Password",  "su_pass",  type_="password",
                              placeholder="at least 6 characters")
            st.markdown('<div class="auth-label">Confirm Password</div>',
                        unsafe_allow_html=True)
            su_pass2 = _field("Confirm",   "su_pass2", type_="password",
                              placeholder="repeat password")
            if st.button("Create Account", use_container_width=True,
                         key="su_btn", type="primary"):
                if su_pass != su_pass2:
                    st.error("Passwords do not match.")
                else:
                    ok, msg = register(su_user, su_email, su_pass)
                    if ok:
                        login(su_user, su_pass)
                        st.rerun()
                    else:
                        st.error(msg)


def render_sidebar_user() -> bool:
    user = current_user()
    info = current_user_info()
    if not user:
        return False

    initials = user[:2].upper()
    tier      = info.get("tier", "free")
    tier_color = "#C8A45D" if tier == "pro" else "#6B7A93"
    tier_label = "PRO" if tier == "pro" else "FREE"
    enforced_note = "" if SUBSCRIPTIONS_ENFORCED else " ✓"

    st.sidebar.markdown(
        f'<div style="display:flex;align-items:center;gap:10px;padding:10px 4px 6px">'
        f'<div style="width:36px;height:36px;border-radius:50%;background:#8C1515;'
        f'display:flex;align-items:center;justify-content:center;'
        f'font-weight:800;font-size:.9rem;color:#FFF7E8;flex-shrink:0">{initials}</div>'
        f'<div style="min-width:0">'
        f'<div style="color:#F7F2E8;font-weight:700;font-size:.9rem;'
        f'white-space:nowrap;overflow:hidden;text-overflow:ellipsis">{user}</div>'
        f'<span style="background:{tier_color}22;color:{tier_color};font-size:.65rem;'
        f'font-weight:700;border-radius:4px;padding:1px 6px">'
        f'{tier_label}{enforced_note}</span>'
        f'</div></div>',
        unsafe_allow_html=True,
    )

    go_profile  = st.sidebar.button("My Profile",  use_container_width=True, key="sb_profile")
    go_upgrade  = st.sidebar.button("⭐ Upgrade",   use_container_width=True, key="sb_upgrade")
    st.sidebar.button("Sign Out", use_container_width=True, key="sb_logout", on_click=logout)

    if go_upgrade:
        st.session_state["cbb_show_upgrade"] = True
        st.session_state.pop("cbb_show_profile", None)
    if go_profile:
        st.session_state["cbb_show_profile"] = True
        st.session_state.pop("cbb_show_upgrade", None)

    return go_profile


def render_pricing_page() -> None:
    st.markdown("## CBB+ Subscription Plans")

    if not SUBSCRIPTIONS_ENFORCED:
        st.success(
            "**Currently free for all members.** Full access is open while we're in early access. "
            "Subscribe now to lock in your rate before we go live.")

    st.markdown("<br>", unsafe_allow_html=True)

    plans = ["weekly", "monthly", "quarterly", "annual"]
    cols  = st.columns(4)

    for col, plan_key in zip(cols, plans):
        p = PRICING[plan_key]
        link = STRIPE_LINKS[plan_key]
        is_placeholder = "PLACEHOLDER" in link

        badge_html = (
            f'<div style="background:#C8A45D;color:#0E1117;font-size:.7rem;'
            f'font-weight:800;border-radius:4px;padding:2px 8px;'
            f'display:inline-block;margin-bottom:.6rem">{p["badge"]}</div>'
            if p["badge"] else '<div style="height:24px;margin-bottom:.6rem"></div>'
        )

        with col:
            st.markdown(
                f'<div style="background:#171D27;border:1px solid #2E3D55;'
                f'border-radius:12px;padding:1.6rem 1.4rem;text-align:center;height:100%">'
                f'{badge_html}'
                f'<div style="color:#9BAABF;font-size:.85rem;font-weight:700;'
                f'text-transform:uppercase;letter-spacing:.08em">{p["label"]}</div>'
                f'<div style="color:#F7F2E8;font-size:2.4rem;font-weight:900;'
                f'margin:.4rem 0 0">{p["price"]}</div>'
                f'<div style="color:#9BAABF;font-size:.85rem;margin-bottom:.2rem">'
                f'{p["period"]}</div>'
                f'{"<div style=color:#C8A45D;font-size:.8rem;font-weight:700;margin-bottom:1rem>" + p["per_month"] + "</div>" if p["per_month"] else "<div style=height:1.4rem></div>"}'
                f'</div>',
                unsafe_allow_html=True,
            )
            if is_placeholder:
                st.button(f"Subscribe — {p['price']}", key=f"sub_{plan_key}",
                          use_container_width=True, disabled=True,
                          help="Payment link coming soon")
            else:
                st.link_button(f"Subscribe — {p['price']}", link,
                               use_container_width=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("### What's included")
    for feat in PRO_FEATURES:
        st.markdown(f"✓ &nbsp; {feat}")

    st.markdown("---")
    st.caption(
        "After subscribing, email us at support@cbbreports.com with your username "
        "and we'll activate your Pro account within 24 hours. "
        "Cancel anytime — no questions asked.")


def render_admin_panel() -> None:
    st.markdown("---")
    st.markdown("### Admin Panel")

    users = all_users()
    if not users:
        st.info("No users yet.")
        return

    st.caption(f"{len(users)} total users")

    cols = st.columns([2, 2.5, 1.2, 1.5, 1.5])
    for h, c in zip(["Username", "Email", "Tier", "Joined", "Action"], cols):
        c.markdown(f"**{h}**")

    for u in sorted(users, key=lambda x: x["joined"], reverse=True):
        c_name, c_email, c_tier, c_joined, c_action = st.columns([2, 2.5, 1.2, 1.5, 1.5])
        tier_color = "#C8A45D" if u["tier"] == "pro" else "#6B7A93"
        c_name.markdown(
            f'<span style="color:#F7F2E8;font-weight:600">{u["username"]}</span>'
            + (' 🛡' if u.get("role") == "admin" else ""),
            unsafe_allow_html=True)
        c_email.markdown(f'<span style="color:#9BAABF;font-size:.85rem">{u["email"]}</span>',
                         unsafe_allow_html=True)
        c_tier.markdown(
            f'<span style="background:{tier_color}22;color:{tier_color};'
            f'font-size:.72rem;font-weight:700;border-radius:4px;padding:2px 7px">'
            f'{u["tier"].upper()}</span>',
            unsafe_allow_html=True)
        c_joined.markdown(f'<span style="color:#9BAABF;font-size:.82rem">{u["joined"]}</span>',
                          unsafe_allow_html=True)
        if u["tier"] == "pro":
            if c_action.button("Downgrade", key=f"dn_{u['username']}", use_container_width=True):
                set_user_tier(u["username"], "free")
                st.rerun()
        else:
            if c_action.button("Upgrade ⭐", key=f"up_{u['username']}", use_container_width=True,
                               type="primary"):
                set_user_tier(u["username"], "pro")
                st.rerun()


def render_profile_page(safe_team_name_fn, all_team_codes, all_player_names) -> None:
    user    = current_user()
    info    = current_user_info()
    profile = get_profile(user)

    st.markdown("## My Profile")

    initials   = user[:2].upper()
    tier       = info.get("tier", "free")
    tier_color = "#C8A45D" if tier == "pro" else "#6B7A93"

    av_col, info_col = st.columns([1, 5])
    with av_col:
        st.markdown(
            f'<div style="width:64px;height:64px;border-radius:50%;background:#8C1515;'
            f'display:flex;align-items:center;justify-content:center;'
            f'font-weight:800;font-size:1.5rem;color:#FFF7E8;margin-top:8px">'
            f'{initials}</div>',
            unsafe_allow_html=True)
    with info_col:
        st.markdown(f"**Username:** {user}")
        st.markdown(f"**Email:** {info.get('email','—')}")
        st.markdown(f"**Member since:** {info.get('joined','—')}")
        enforced_note = "" if SUBSCRIPTIONS_ENFORCED else "  *(free access for everyone right now)*"
        st.markdown(
            f'**Plan:** <span style="background:{tier_color}22;color:{tier_color};'
            f'font-size:.8rem;font-weight:700;border-radius:4px;padding:2px 8px">'
            f'{tier.upper()}</span>{enforced_note}',
            unsafe_allow_html=True)

    st.markdown("---")

    # ── Favorite Teams ────────────────────────────────────────────────────────
    st.markdown("### Favorite Teams")
    team_opts = sorted(all_team_codes, key=lambda c: safe_team_name_fn(c).lower())
    cur_teams = [t for t in profile.get("favorite_teams", []) if t in team_opts]
    new_teams = st.multiselect(
        "Select your favorite teams",
        options=team_opts,
        default=cur_teams,
        format_func=safe_team_name_fn,
        key="prof_fav_teams",
    )

    if new_teams:
        tc = st.columns(min(len(new_teams), 4))
        for i, code in enumerate(new_teams):
            tc[i % 4].markdown(
                f'<div style="background:#171D27;border:1px solid #2E3D55;'
                f'border-radius:8px;padding:8px 10px;text-align:center;margin:4px 0">'
                f'<div style="color:#F7F2E8;font-weight:700;font-size:.88rem">'
                f'{safe_team_name_fn(code)}</div>'
                f'<div style="color:#9BAABF;font-size:.72rem">{code}</div></div>',
                unsafe_allow_html=True)

    st.markdown("---")

    # ── Favorite Players ──────────────────────────────────────────────────────
    st.markdown("### Favorite Players")
    cur_players = [p for p in profile.get("favorite_players", []) if p in all_player_names]
    new_players = st.multiselect(
        "Search and add players",
        options=sorted(all_player_names),
        default=cur_players,
        key="prof_fav_players",
        help="Type to search by name",
    )

    if new_players:
        for p in new_players:
            st.markdown(
                f'<div style="background:#171D27;border:1px solid #2E3D55;'
                f'border-radius:6px;padding:6px 12px;margin:3px 0;'
                f'color:#F7F2E8;font-size:.92rem">⚾ &nbsp; {p}</div>',
                unsafe_allow_html=True)

    st.markdown("---")

    if st.button("Save Profile", type="primary"):
        save_profile(user, {"favorite_teams": new_teams, "favorite_players": new_players})
        st.success("Profile saved.")

    # ── Admin panel ───────────────────────────────────────────────────────────
    if is_admin():
        render_admin_panel()
