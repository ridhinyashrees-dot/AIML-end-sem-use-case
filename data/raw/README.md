# Real dataset required (not included)

**Dataset:** Location of Manganese Ore Deposits in India and its Salient Features (Geological Survey of India, Ministry of Mines)
**Official source:** https://tn.data.gov.in/catalog/location-manganese-ore-deposits-india-and-its-salient-features
(also reachable via https://data.gov.in). Open the page, log in/register on OGD if asked, and use the **CSV** download.

**Save as:** `data/raw/manganese_deposits.csv`

**Expected columns** (names matched loosely, case-insensitive): Locality, State, Latitude, Longitude, Toposheet, Host rock, Geological formation, Metallogenesis, Morphogenesis.
Latitude/Longitude may be decimal degrees or degree-minute-second text; both are parsed.
Only use the genuine OGD/GSI file. Do not substitute other datasets. Record the download date for your report.
