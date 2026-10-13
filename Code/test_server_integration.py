import urllib.request
import json

def test_url(url, method='GET', data=None):
    req = urllib.request.Request(url, method=method, data=data)
    if data:
        req.add_header('Content-Type', 'application/json')
    try:
        with urllib.request.urlopen(req) as resp:
            content_type = resp.headers.get('Content-Type', '')
            body = resp.read()
            return resp.status, content_type, len(body), body
    except Exception as e:
        return 500, str(e), 0, b''

endpoints = [
    ('/', 'GET', None),
    ('/style.css', 'GET', None),
    ('/app.js', 'GET', None),
    ('/api/predict', 'GET', None),
    ('/api/equity', 'GET', None),
    ('/api/metrics', 'GET', None),
    ('/api/summary', 'GET', None),
    ('/presentation/index.html', 'GET', None),
    ('/reports/UE24CS352A_MiniProject_Writeup.pdf', 'GET', None)
]

print('=' * 68)
print('INTEGRATION TEST: LIVE SERVER ON PORT 8000')
print('=' * 68)
for path, meth, d in endpoints:
    status, ct, length, body = test_url(f'http://127.0.0.1:8000{path}', meth, d)
    short_ct = ct.split(';')[0] if ct else 'unknown'
    print(f'{meth:4} {path:<45} -> HTTP {status} | {length:>6,} bytes | {short_ct}')
    assert status == 200, f'Failed {path}'

# Test POST /api/predict with chip shortage shock
post_body = json.dumps({'returns': {'CON_DE': -1.8, 'IFX_DE': -3.8}, 'threshold': 0.005}).encode('utf-8')
status, ct, length, body = test_url('http://127.0.0.1:8000/api/predict', 'POST', post_body)
res = json.loads(body.decode('utf-8'))
print('-' * 68)
print(f'POST /api/predict (Chip Shortage Shock)       -> HTTP {status}')
print(f'     Model:            {res["model"].encode("ascii", "replace").decode()} (h={res["horizon_trading_days"]}d)')
print(f'     Predicted Return: {res["predicted_return_pct"]:+.4f}%')
print(f'     Signal:           {res["signal"]}')
print(f'     Position:         {res["position"]}')
print(f'     Conviction Ratio: {res["conviction_ratio"]}x')
print(f'     Feature Count:    {res["feature_count"]}')
print(f'     Pipeline Valid:   {res["pipeline_validated"]}')
assert res['success'] is True
assert res['pipeline_validated'] is True
assert res['feature_count'] == 50

print('=' * 68)
print('ALL INTEGRATION CHECKS PASSED!')
print('=' * 68)
