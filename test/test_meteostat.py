import requests

headers = {
    "x-rapidapi-key": "ce3ea3a25emsh859d0849eb48cf7p1887a9jsne373285f352c",
    "x-rapidapi-host": "meteostat.p.rapidapi.com",
}
params = {
    "lat": "40.7128",
    "lon": "-74.0060",
    "start": "2025-08-01",  # Note：YYYY-MM-DD
    "end": "2025-08-07",  # Note：YYYY-MM-DD
}

r = requests.get("https://meteostat.p.rapidapi.com/point/hourly", params=params, headers=headers, timeout=15)

print(r.status_code)
print(r.json() if r.ok else r.text)
