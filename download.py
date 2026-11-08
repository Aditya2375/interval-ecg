"""Download the original public MIT-BIH files, preserving attribution in README/model."""
import re,urllib.request
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
BASE='https://www.physionet.org/files/mitdb/1.0.0/'
def main():
 r=Path('data');r.mkdir(exist_ok=True);records=urllib.request.urlopen(BASE+'RECORDS').read().decode().split()
 def get(pair):
  rec,ext=pair;p=r/(rec+'.'+ext)
  if not p.exists():urllib.request.urlretrieve(BASE+p.name,p)
 with ThreadPoolExecutor(4) as pool:list(pool.map(get,[(rec,ext) for rec in records for ext in ['dat','hea','atr']]))
 print('Downloaded',len(records),'records. Source and license: https://www.physionet.org/content/mitdb/1.0.0/')
if __name__=='__main__':main()
