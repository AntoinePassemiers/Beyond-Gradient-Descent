from datetime import datetime as dt

def log(txt, end='\n'):
    print('[{}]\t{}'.format(now(), txt), end=end, flush=True)

def now():
    return dt.now().strftime('%Y-%m-%d %T.%f')
