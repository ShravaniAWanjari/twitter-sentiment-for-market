import urllib.request, json
from datetime import date, timedelta

key = 'pub_d627624cbaf1496995d1135b92e85d5b'

def test(from_d, to_d):
    url = 'https://newsdata.io/api/1/archive?apikey={}&q=bitcoin&language=en&from_date={}&to_date={}&size=3'.format(key, from_d, to_d)
    try:
        r = urllib.request.urlopen(url, timeout=15)
        d = json.loads(r.read().decode())
        n = len(d.get('results', []))
        print('{} -> {}: status={} results={}'.format(from_d, to_d, d.get('status'), n))
        if n: print('  title:', d['results'][0].get('title','')[:80])
        if d.get('message'): print('  msg:', d['message'])
    except Exception as e:
        print('{} -> {}: ERROR {}'.format(from_d, to_d, str(e)[:80]))

today = date.today()
print('Today is:', today)
# Test recent dates (what the user showed works)
for offset in [1, 2, 3, 7, 10, 14, 20, 30]:
    d1 = today - timedelta(days=offset+1)
    d2 = today - timedelta(days=offset)
    test(str(d1), str(d2))
