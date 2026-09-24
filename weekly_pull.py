"""Weekly intraday bar pull for backtesting (runs on GitHub Actions).

Reads every earlier manifest (downloaded from Google Drive into PREV_DIR) to find the
last saved bar for each ticker + interval, fetches only newer bars from Yahoo Finance,
and writes them to OUT_DIR/<run date>/ in the same layout as the Colab snapshot:

    <run date>/1m/QQQ_1m_<run date>.csv
    <run date>/2m/...
    <run date>/5m/...
    <run date>/manifest_<run date>.csv

A ticker with no earlier data (new to the list) gets the full history Yahoo allows.
"""
import glob, os, sys, time, datetime as dt
import pandas as pd
import yfinance as yf
from zoneinfo import ZoneInfo

HERE = os.path.dirname(os.path.abspath(__file__))
PREV_DIR = os.environ.get('PREV_DIR', 'prev')      # earlier manifests from Drive
OUT_DIR = os.environ.get('OUT_DIR', 'out')         # this run's files, uploaded to Drive after

# Market benchmarks, always pulled alongside the watchlist (QQQ = relative-strength filter proxy)
BENCHMARKS = ['QQQ']

# How far back Yahoo serves each interval (1 day inside the limit so requests aren't rejected)
MAX_DAYS = {'1m': 29, '2m': 59, '5m': 59}
CHUNK_DAYS = {'1m': 5, '2m': 59, '5m': 59}          # 1m pulled 5 days at a time
BAR = {'1m': dt.timedelta(minutes=1), '2m': dt.timedelta(minutes=2), '5m': dt.timedelta(minutes=5)}
REGULAR_HOURS_ONLY = True
PAUSE = 1.0                                         # seconds between Yahoo requests
FAIL_IF_ERRORS_OVER = 0.10                          # mark the run failed (GitHub emails you) if >10% error

NY = ZoneInfo('America/New_York')
NOW_UTC = dt.datetime.now(dt.timezone.utc)
NOW_NY = NOW_UTC.astimezone(NY)
RUN_DATE = NOW_NY.strftime('%Y-%m-%d')
DROP_PARTIAL_TODAY = NOW_NY.weekday() < 5 and NOW_NY.time() < dt.time(16, 5)


def load_tickers():
    raw = open(os.path.join(HERE, 'tickers.txt')).read().replace('\n', ',')
    watch = [t.split(':')[-1].strip().upper() for t in raw.split(',') if t.strip()]
    return list(dict.fromkeys(BENCHMARKS + watch))


def last_saved_bars():
    """{(ticker, interval): last bar time (UTC)} across every earlier manifest."""
    files = glob.glob(os.path.join(PREV_DIR, '**', 'manifest_*.csv'), recursive=True)
    if not files:
        return {}, 0
    m = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    m = m[m['status'].astype(str).str.startswith(('ok', 'saved')) & m['last_bar'].notna()]
    m['last_bar'] = pd.to_datetime(m['last_bar'], utc=True)
    return m.groupby(['ticker', 'interval'])['last_bar'].max().to_dict(), len(files)


def fetch(ticker, interval, start):
    tk = yf.Ticker(ticker.replace('.', '-'))
    frames, cur_start = [], start
    while cur_start < NOW_UTC:
        cur_end = min(NOW_UTC, cur_start + dt.timedelta(days=CHUNK_DAYS[interval]))
        df, err = None, None
        for attempt in range(4):
            try:
                df = tk.history(start=cur_start, end=cur_end, interval=interval,
                                prepost=not REGULAR_HOURS_ONLY, auto_adjust=False, actions=False)
                break
            except Exception as e:           # rate limit / network hiccup: back off and retry
                err = e
                time.sleep(5 * (attempt + 1))
        if df is None and err:
            raise err
        if df is not None and len(df):
            frames.append(df)
        cur_start = cur_end
        time.sleep(PAUSE)
    if not frames:
        return pd.DataFrame()
    df = pd.concat(frames)
    df = df[~df.index.duplicated(keep='last')].sort_index()
    df.index = df.index.tz_convert(NY)
    df.index.name = 'Datetime'
    df = df[['Open', 'High', 'Low', 'Close', 'Volume']]
    df = df[df.index > start.astimezone(NY)]                 # only bars after the last saved one
    if DROP_PARTIAL_TODAY:
        df = df[df.index.date < NOW_NY.date()]
    return df


def main():
    tickers = load_tickers()
    last, n_manifests = last_saved_bars()
    # Never overwrite an existing Drive folder (e.g. a Colab snapshot or a second run the same day)
    existing = set(os.listdir(PREV_DIR)) if os.path.isdir(PREV_DIR) else set()
    if os.path.exists(os.path.join(PREV_DIR, '_dirs.txt')):
        existing |= {d.strip().strip('/') for d in open(os.path.join(PREV_DIR, '_dirs.txt'))}
    global RUN_DATE
    base, n = RUN_DATE, 2
    while RUN_DATE in existing:
        RUN_DATE, n = f'{base}_{n}', n + 1
    out = os.path.join(OUT_DIR, RUN_DATE)
    for iv in MAX_DAYS:
        os.makedirs(os.path.join(out, iv), exist_ok=True)
    print(f'{len(tickers)} tickers | {n_manifests} earlier manifests found | run date {RUN_DATE}')

    rows = pull(tickers, last, out)

    # One retry pass for anything that errored (usually Yahoo rate limiting)
    failed = sorted({r['ticker'] for r in rows if r['status'].startswith('ERROR')})
    if failed:
        print(f'\nRetrying {len(failed)} ticker(s) after a 2-minute pause: {", ".join(failed)}')
        time.sleep(120)
        retry = {(r['ticker'], r['interval']): r for r in pull(failed, last, out)}
        rows = [retry.get((r['ticker'], r['interval']), r) if r['status'].startswith('ERROR') else r for r in rows]
    finish(rows, tickers, out)


def pull(tickers, last, out):
    rows = []
    for i, t in enumerate(tickers, 1):
        line = []
        for iv in MAX_DAYS:
            floor = NOW_UTC - dt.timedelta(days=MAX_DAYS[iv])
            prev = last.get((t, iv))
            note = ''
            if prev is None:
                start, note = floor, 'no earlier data: pulled full history'
            elif prev + BAR[iv] < floor:
                start, note = floor, f'GAP: bars between {prev:%Y-%m-%d} and {floor:%Y-%m-%d} no longer available'
            else:
                start = prev.to_pydatetime()
            try:
                df = fetch(t, iv, start)
                if df.empty:
                    status = 'ok: no new bars'
                    line.append(f'{iv}:0')
                else:
                    df.to_csv(os.path.join(out, iv, f'{t}_{iv}_{RUN_DATE}.csv'))
                    status = 'ok'
                    line.append(f'{iv}:{len(df):,}')
            except Exception as e:
                df, status = pd.DataFrame(), f'ERROR: {e}'
                line.append(f'{iv}:ERROR')
            rows.append(dict(ticker=t, interval=iv, bars=len(df),
                             first_bar=df.index.min() if len(df) else None,
                             last_bar=df.index.max() if len(df) else None,
                             trading_days=pd.Index(df.index.date).nunique() if len(df) else 0,
                             status=status, note=note))
        print(f'[{i:>3}/{len(tickers)}] {t:<6} ' + '  '.join(line))
    return rows


def finish(rows, tickers, out):
    manifest = pd.DataFrame(rows)
    manifest.to_csv(os.path.join(out, f'manifest_{RUN_DATE}.csv'), index=False)

    # Summary shown on the GitHub run page
    errors = manifest[manifest.status.str.startswith('ERROR')]
    flagged = manifest[manifest.note != '']
    summary = [f'## Backtesting data pull — {RUN_DATE}', '',
               f'Tickers: {len(tickers)} | Files saved: {int((manifest.bars > 0).sum())} | '
               f'Bars saved: {int(manifest.bars.sum()):,} | Errors: {len(errors)}', '']
    if len(flagged):
        summary += ['### Notes', ''] + [f'- {r.ticker} {r.interval}: {r.note}' for r in flagged.itertuples()] + ['']
    if len(errors):
        summary += ['### Errors', ''] + [f'- {r.ticker} {r.interval}: {r.status}' for r in errors.itertuples()]
    text = '\n'.join(summary)
    print('\n' + text)
    if os.environ.get('GITHUB_STEP_SUMMARY'):
        with open(os.environ['GITHUB_STEP_SUMMARY'], 'a') as f:
            f.write(text + '\n')

    if len(errors) > FAIL_IF_ERRORS_OVER * len(manifest):
        sys.exit(f'Too many errors ({len(errors)}/{len(manifest)}) — files that did download are still uploaded.')


if __name__ == '__main__':
    main()
