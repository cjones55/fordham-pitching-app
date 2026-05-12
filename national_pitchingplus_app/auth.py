"""
auth.py — CBB+ user authentication and profile management.
File-based backend: users_db.yaml (credentials) + user_profiles.json (favorites).
"""
from __future__ import annotations

import json
import os
from datetime import date
from pathlib import Path

import bcrypt
import streamlit as st
import yaml

_DIR = Path(__file__).resolve().parent
USERS_FILE    = _DIR / "users_db.yaml"
PROFILES_FILE = _DIR / "user_profiles.json"


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

    data = _load_users()
    if username in data["users"]:
        return False, "Username already taken."
    emails = {u["email"] for u in data["users"].values()}
    if email in emails:
        return False, "An account with that email already exists."

    data["users"][username] = {
        "email":    email,
        "password": _hash(password),
        "joined":   str(date.today()),
        "role":     "user",
        "tier":     "free",
    }
    _save_users(data)

    profiles = _load_profiles()
    profiles[username] = {"favorite_teams": [], "favorite_players": []}
    _save_profiles(profiles)
    return True, "Account created successfully."


def login(username: str, password: str) -> tuple[bool, str]:
    username = username.strip().lower()
    data = _load_users()
    user = data["users"].get(username)
    if user is None:
        return False, "Username not found."
    if not _verify(password, user["password"]):
        return False, "Incorrect password."
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


# ── Profile operations ────────────────────────────────────────────────────────

def get_profile(username: str) -> dict:
    return _load_profiles().get(username, {"favorite_teams": [], "favorite_players": []})


def save_profile(username: str, profile: dict) -> None:
    profiles = _load_profiles()
    profiles[username] = profile
    _save_profiles(profiles)


# ── UI helpers ────────────────────────────────────────────────────────────────

def _field(label, key, type_="default", placeholder=""):
    return st.text_input(label, key=key,
                         type=type_, placeholder=placeholder,
                         label_visibility="collapsed")


def render_auth_page() -> bool:
    """
    Render the full-screen login/signup gate.
    Returns True once the user is authenticated (triggers app rerun).
    """
    st.markdown("""
    <style>
    .auth-wrap{display:flex;flex-direction:column;align-items:center;
               justify-content:center;min-height:78vh;padding:2rem 1rem}
    .auth-card{background:#171D27;border:1px solid #2E3D55;border-radius:14px;
               padding:2.4rem 2.8rem;width:100%;max-width:420px;
               box-shadow:0 8px 32px rgba(0,0,0,.45)}
    .auth-title{font-size:2rem;font-weight:800;color:#F7F2E8;
                text-align:center;margin-bottom:.3rem}
    .auth-sub  {font-size:.95rem;color:#9BAABF;text-align:center;
                margin-bottom:1.6rem}
    .auth-label{font-size:.78rem;color:#9BAABF;font-weight:600;
                letter-spacing:.06em;text-transform:uppercase;
                margin-bottom:.25rem}
    .auth-err  {color:#F04444;font-size:.85rem;margin-top:.5rem}
    .auth-ok   {color:#35C46B;font-size:.85rem;margin-top:.5rem}
    </style>
    """, unsafe_allow_html=True)

    _, col, _ = st.columns([1, 2, 1])
    with col:
        st.markdown('<div class="auth-title">CBB+</div>', unsafe_allow_html=True)
        st.markdown('<div class="auth-sub">College Baseball Analytics</div>',
                    unsafe_allow_html=True)

        tab_in, tab_up = st.tabs(["Sign In", "Create Account"])

        with tab_in:
            st.markdown('<div class="auth-label">Username</div>', unsafe_allow_html=True)
            li_user = _field("Username", "li_user", placeholder="your username")
            st.markdown('<div class="auth-label">Password</div>', unsafe_allow_html=True)
            li_pass = _field("Password", "li_pass", type_="password", placeholder="••••••••")
            if st.button("Sign In", use_container_width=True, key="li_btn",
                         type="primary"):
                ok, msg = login(li_user, li_pass)
                if ok:
                    st.rerun()
                else:
                    st.markdown(f'<div class="auth-err">{msg}</div>',
                                unsafe_allow_html=True)

        with tab_up:
            st.markdown('<div class="auth-label">Username</div>', unsafe_allow_html=True)
            su_user = _field("Username", "su_user", placeholder="choose a username")
            st.markdown('<div class="auth-label">Email</div>', unsafe_allow_html=True)
            su_email = _field("Email", "su_email", placeholder="you@example.com")
            st.markdown('<div class="auth-label">Password</div>', unsafe_allow_html=True)
            su_pass = _field("Password", "su_pass", type_="password",
                             placeholder="at least 6 characters")
            st.markdown('<div class="auth-label">Confirm Password</div>',
                        unsafe_allow_html=True)
            su_pass2 = _field("Confirm", "su_pass2", type_="password",
                              placeholder="repeat password")
            if st.button("Create Account", use_container_width=True, key="su_btn",
                         type="primary"):
                if su_pass != su_pass2:
                    st.markdown('<div class="auth-err">Passwords do not match.</div>',
                                unsafe_allow_html=True)
                else:
                    ok, msg = register(su_user, su_email, su_pass)
                    if ok:
                        login(su_user, su_pass)
                        st.rerun()
                    else:
                        st.markdown(f'<div class="auth-err">{msg}</div>',
                                    unsafe_allow_html=True)

    return False


def render_sidebar_user(all_teams: list[str] | None = None) -> bool:
    """
    Render the user chip in the sidebar.
    Returns True if the user navigated to the Profile page.
    """
    user = current_user()
    info = current_user_info()
    if not user:
        return False

    initials = user[:2].upper()
    tier_badge = (
        '<span style="background:#C8A45D;color:#0E1117;font-size:.65rem;'
        'font-weight:700;border-radius:4px;padding:1px 5px;margin-left:6px">PRO</span>'
        if info.get("tier") == "pro" else
        '<span style="background:#344055;color:#9BAABF;font-size:.65rem;'
        'font-weight:700;border-radius:4px;padding:1px 5px;margin-left:6px">FREE</span>'
    )
    st.sidebar.markdown(
        f'<div style="display:flex;align-items:center;gap:10px;padding:10px 4px 4px">'
        f'<div style="width:36px;height:36px;border-radius:50%;background:#8C1515;'
        f'display:flex;align-items:center;justify-content:center;'
        f'font-weight:800;font-size:.95rem;color:#FFF7E8">{initials}</div>'
        f'<div><div style="color:#F7F2E8;font-weight:700;font-size:.95rem">'
        f'{user}{tier_badge}</div>'
        f'<div style="color:#9BAABF;font-size:.75rem">{info.get("email","")}</div>'
        f'</div></div>',
        unsafe_allow_html=True,
    )

    go_profile = st.sidebar.button("My Profile", use_container_width=True,
                                   key="sb_profile_btn")
    st.sidebar.button("Sign Out", use_container_width=True, key="sb_logout_btn",
                      on_click=logout)
    return go_profile


def render_profile_page(safe_team_name_fn, all_team_codes: list[str],
                        all_player_names: list[str]) -> None:
    """Full-page profile editor."""
    user    = current_user()
    info    = current_user_info()
    profile = get_profile(user)

    st.markdown(f"## My Profile")

    initials = user[:2].upper()
    col_av, col_info = st.columns([1, 5])
    with col_av:
        st.markdown(
            f'<div style="width:64px;height:64px;border-radius:50%;background:#8C1515;'
            f'display:flex;align-items:center;justify-content:center;'
            f'font-weight:800;font-size:1.5rem;color:#FFF7E8;margin-top:8px">'
            f'{initials}</div>',
            unsafe_allow_html=True,
        )
    with col_info:
        st.markdown(f"**Username:** {user}")
        st.markdown(f"**Email:** {info.get('email', '—')}")
        st.markdown(f"**Member since:** {info.get('joined', '—')}")
        tier = info.get("tier", "free").upper()
        tier_color = "#C8A45D" if tier == "PRO" else "#9BAABF"
        st.markdown(
            f'**Plan:** <span style="color:{tier_color};font-weight:700">{tier}</span>',
            unsafe_allow_html=True,
        )

    st.markdown("---")

    # ── Favorite Teams ────────────────────────────────────────────────────────
    st.markdown("### Favorite Teams")
    team_options = sorted(all_team_codes,
                          key=lambda c: safe_team_name_fn(c).lower())
    current_fav_teams = [t for t in profile.get("favorite_teams", [])
                         if t in team_options]
    new_fav_teams = st.multiselect(
        "Select your favorite teams",
        options=team_options,
        default=current_fav_teams,
        format_func=safe_team_name_fn,
        key="prof_fav_teams",
    )

    st.markdown("---")

    # ── Favorite Players ──────────────────────────────────────────────────────
    st.markdown("### Favorite Players")
    current_fav_players = [p for p in profile.get("favorite_players", [])
                           if p in all_player_names]
    new_fav_players = st.multiselect(
        "Search and add players",
        options=sorted(all_player_names),
        default=current_fav_players,
        key="prof_fav_players",
        help="Type to search by name",
    )

    st.markdown("---")

    # ── Display current favorites ─────────────────────────────────────────────
    if new_fav_teams:
        st.markdown("#### Your Teams")
        cols = st.columns(min(len(new_fav_teams), 4))
        for i, tc in enumerate(new_fav_teams):
            cols[i % 4].markdown(
                f'<div style="background:#171D27;border:1px solid #2E3D55;'
                f'border-radius:8px;padding:8px 12px;text-align:center;margin:4px 0">'
                f'<div style="color:#F7F2E8;font-weight:700;font-size:.9rem">'
                f'{safe_team_name_fn(tc)}</div>'
                f'<div style="color:#9BAABF;font-size:.75rem">{tc}</div></div>',
                unsafe_allow_html=True,
            )

    if new_fav_players:
        st.markdown("#### Your Players")
        for p in new_fav_players:
            st.markdown(
                f'<div style="background:#171D27;border:1px solid #2E3D55;'
                f'border-radius:6px;padding:6px 12px;margin:3px 0;'
                f'color:#F7F2E8;font-size:.92rem">⚾  {p}</div>',
                unsafe_allow_html=True,
            )

    if st.button("Save Profile", type="primary", use_container_width=False):
        save_profile(user, {
            "favorite_teams":   new_fav_teams,
            "favorite_players": new_fav_players,
        })
        st.success("Profile saved.")
