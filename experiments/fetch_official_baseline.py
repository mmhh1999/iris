"""Range-read public IRIS ZIP archives; extract only requested prefixes.

No credentials required. Uses public download URLs whose share pages permit
file download. Signed redirects are held in memory and never printed.
"""
import argparse
import io
import json
from pathlib import Path
import time
import zipfile
import zlib
import requests

SHARES={'data':('q26vjzweu2fj7rb9aswfxdqrgxwq0uom',6596187539),
        'checkpoints':('64aor3xxzovnnf879cley89op5dde9ir',2172111129)}

class RemoteZIP(io.RawIOBase):
    def __init__(self,share,size):
        self.size=size;self.pos=0;self.session=requests.Session();self.url='https://uofi.box.com/shared/static/'+share+'.zip';self.downloaded=0
        self.cache=b'';self.cache_start=0
    def seekable(self):return True
    def readable(self):return True
    def tell(self):return self.pos
    def seek(self,offset,whence=0):
        self.pos=offset if whence==0 else self.pos+offset if whence==1 else self.size+offset
        if self.pos<0:raise ValueError('negative position')
        return self.pos
    def read(self,n=-1):
        if n<0:n=self.size-self.pos
        n=min(n,self.size-self.pos)
        if n<=0:return b''
        if self.cache_start<=self.pos and self.pos+n<=self.cache_start+len(self.cache):
            offset=self.pos-self.cache_start;self.pos+=n
            return self.cache[offset:offset+n]
        if n>512*1024*1024:raise ValueError('range too large; use streaming zip extraction')
        begin=self.pos;end=min(self.size-1,begin+max(n,8*1024*1024)-1)
        for attempt in range(3):
            r=self.session.get(self.url,headers={'Range':f'bytes={begin}-{end}','Accept-Encoding':'identity'},timeout=90)
            if r.status_code==206:break
            time.sleep(1+attempt)
        if r.status_code!=206 or r.headers.get('Content-Range')!=f'bytes {begin}-{end}/{self.size}':
            raise RuntimeError(f'Range request failed: HTTP {r.status_code}')
        content=r.content
        if len(content)!=end-begin+1:raise RuntimeError('truncated range')
        self.cache_start=begin;self.cache=content
        self.url=r.url;self.pos+=n;self.downloaded+=len(content)
        return content[:n]

def main(args):
    remote=RemoteZIP(*SHARES[args.archive]);out=Path(args.out);out.mkdir(parents=True,exist_ok=True)
    with zipfile.ZipFile(remote) as z:
        info=[dict(name=i.filename,compressed=i.compress_size,size=i.file_size) for i in z.infolist()]
        (out/(args.archive+'_inventory.json')).write_text(json.dumps(info,indent=2))
        if not args.prefix:
            groups={}
            for i in info:
                parts=i['name'].split('/');group='/'.join(parts[:min(4,len(parts)-1)])
                g=groups.setdefault(group,dict(files=0,compressed_MB=0,size_MB=0));g['files']+=1;g['compressed_MB']+=i['compressed']/1e6;g['size_MB']+=i['size']/1e6
            print(json.dumps(groups,indent=2));return
        selected=[i for i in z.infolist() if any(i.filename.startswith(p) for p in args.prefix) and not i.is_dir()]
        print('Selected',len(selected),'files;',round(sum(i.compress_size for i in selected)/1e6,1),'MB compressed',flush=True)
        for count,i in enumerate(selected):
            dest=(out/i.filename).resolve()
            if not dest.is_relative_to(out.resolve()):raise ValueError('unsafe ZIP path')
            if dest.exists():
                crc=0
                with dest.open('rb') as existing:
                    while chunk:=existing.read(8*1024*1024):crc=zlib.crc32(chunk,crc)
                if dest.stat().st_size==i.file_size and crc==i.CRC:continue
                raise FileExistsError(f'Existing file differs from archive: {dest}')
            dest.parent.mkdir(parents=True,exist_ok=True)
            tmp=dest.with_name(dest.name+'.partial')
            with z.open(i) as src,open(tmp,'wb') as target:
                while chunk:=src.read(8*1024*1024):target.write(chunk)
            # zipfile checks CRC on full read.
            tmp.rename(dest)
            if count%20==0:print(count+1,'/',len(selected),i.filename,flush=True)
        print('Downloaded',round(remote.downloaded/1e6,2),'MB; ZIP CRCs verified.',flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--archive',choices=list(SHARES),required=True);p.add_argument('--prefix',nargs='*');p.add_argument('--out',default='data_download/iris_official');main(p.parse_args())
