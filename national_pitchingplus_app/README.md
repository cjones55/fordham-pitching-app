# CBBReports

Separate sellable scouting/report app. It does not modify the Fordham app.

In-app product title: **College Baseball Pitching Plus**.

## Paywall

Access is controlled by codes in Streamlit secrets:

```toml
[auth]
checkout_url = "https://buy.stripe.com/your-checkout-link"
access_codes = ["CUSTOMER-CODE-1", "CUSTOMER-CODE-2"]
```

For local testing, you can also use:

```bash
export PITCHINGPLUS_ACCESS_CODES="DEMO-2026,CUSTOMER-BINGHAMTON-2026"
streamlit run national_pitchingplus_app/app.py
```

## Data

The app reads the national TrackMan CSV folder:

```text
../scouting_2026_trackman
```

Override in Streamlit secrets:

```toml
[data]
scouting_data_dir = "/absolute/path/to/scouting_2026_trackman"
```

## Logos

Drop licensed/team-approved PNG logos into:

```text
national_pitchingplus_app/team_logos/TEAM_CODE.png
```

Examples:

```text
TEN_VOL.png
BIN_BEA.png
FOR_RAM.png
```

The app falls back to a clean color badge when a logo is missing.
