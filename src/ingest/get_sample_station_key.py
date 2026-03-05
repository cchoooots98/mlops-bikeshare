import json
for fn in [
     'data/samples/velib/station_information.sample.json',
     'data/samples/velib/station_status.sample.json'
 ]:
    with open(fn) as f:
        d = json.load(f)
    print('\n===', fn, '===')
    print('Top-level keys:', list(d.keys()))
    stations = d.get('data', {}).get('stations', [])
    print('Rows:', len(stations))
    if stations:
        print('Sample station keys:', sorted(stations[0].keys())[:20])