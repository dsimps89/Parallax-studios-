
"""Atlas Parallax Studio: run in Pyto, open http://127.0.0.1:8777
Requires Pillow and NumPy. PDF export uses Pillow's PDF writer.
"""
import os, io, json, math, time, random, threading, traceback, cgi, uuid
from http.server import HTTPServer, BaseHTTPRequestHandler
from urllib.parse import urlparse, parse_qs
from PIL import Image, ImageDraw
import numpy as np

PORT=8778
BASE=os.path.join(os.path.expanduser('~'),'Documents','AtlasParallaxOutputs')
os.makedirs(BASE,exist_ok=True)
STATE={'busy':False,'message':'Ready','files':[],'progress':0,'upload':None,'upload_name':None}
LOCK=threading.Lock()

HTML=r'''<!doctype html><html lang="en"><head><meta name="viewport" content="width=device-width,initial-scale=1"><meta charset="utf-8"><title>Atlas Parallax Studio</title><style>
:root{color-scheme:dark}body{font:16px -apple-system,system-ui,sans-serif;background:#081522;color:#e7f1ff;margin:0;padding:18px;max-width:780px;margin:auto}h1{font-size:27px}h2{font-size:19px}section{background:#102239;border:1px solid #2b4964;border-radius:16px;padding:17px;margin:14px 0}label{display:block;margin:13px 0 5px;font-weight:650}select,input{width:100%;box-sizing:border-box;padding:12px;background:#081522;border:1px solid #58748e;border-radius:9px;color:white;font-size:16px}button{background:#24bdd0;color:#06202a;border:0;border-radius:11px;font-size:16px;font-weight:750;padding:14px;margin:6px 5px 6px 0}button.secondary{background:#466586;color:white}.small{color:#a8bed2;font-size:14px;line-height:1.5}#status{white-space:pre-wrap;color:#b6e7ed}a{color:#7ae1ef;display:block;padding:7px 0;word-break:break-word}img{max-width:100%;border-radius:9px}progress{width:100%}</style></head><body>
<h1>Atlas Parallax Studio</h1><p class="small">A local Pyto-powered, depth-map stereogram maker. The HTML controls run in Safari; Python generates the actual pixels on your device.</p>
<section><h2>1 · Upload your image</h2><p class="small">Choose a JPG/PNG/WEBP image. For genuine shape-based 3D, upload a <b>grayscale depth map</b> (white = closer, black = farther). A normal photo has no depth information; the "photo approximation" mode estimates depth from brightness and will not recover actual object geometry.</p><input type="file" id="upload" accept="image/*"><label>How should the uploaded image be interpreted?</label><select id="uploadMode"><option value="depth">Grayscale depth map (recommended)</option><option value="brightness">Ordinary photo: brightness-based approximation</option></select><button class="secondary" onclick="uploadImage()">Upload image</button><p id="uploadStatus" class="small">No image uploaded</p><img id="uploadPreview" style="display:none;max-height:260px;object-fit:contain" alt="Uploaded image preview"></section><section><h2>2 · Subject and print</h2><label>Hidden 3D subject</label><select id="subject"><option value="upload" selected>My uploaded image</option><option value="cube">Calibration cube</option><option value="atlas">Face + floating cubes + arch + staircase</option></select><label>Output size</label><select id="size"><option value="test">Small calibration / fast preview (1200 × 800)</option><option value="preview" selected>Atlas portrait preview (860 × 1600)</option><option value="75">43 × 80 inches at 75 DPI (3225 × 6000)</option><option value="100">43 × 80 inches at 100 DPI (4300 × 8000)</option><option value="150">43 × 80 inches at 150 DPI (6450 × 12000, high memory)</option></select><label>Pattern repeat (inches)</label><input id="repeat" type="number" min="1.0" max="2.4" step="0.05" value="1.85"><label>Maximum depth shift (inches)</label><input id="shift" type="number" min="0.02" max="0.25" step="0.01" value="0.12"><label>Pattern style</label><select id="style"><option value="tile">Crisp colored geometric tiles</option><option value="mono">High-contrast monochrome tiles</option></select><label>Pattern seed</label><input id="seed" type="number" value="42"></section>
<section><h2>3 · Generate</h2><button onclick="go('test')">Quick cube test</button><button onclick="go('custom')">Generate selected image</button><button class="secondary" onclick="refresh()">Refresh results</button><p id="status">Ready</p><progress id="bar" value="0" max="100"></progress><div id="files"></div></section>
<section><h2>4 · How to view</h2><p class="small">Look through the image with relaxed, parallel eyes until repeated tiles fuse. For a large wall print, try 7–10 feet. Begin with the cube test. Small-screen previews are for testing the encoding, not calibrating the final wall viewing distance. If the depth looks inverted, change the shift sign in the Python code (see README).</p><p class="small">The app writes files to Pyto's Documents/AtlasParallaxOutputs folder. Very large renders may exhaust iPhone memory; try preview or 75 DPI first. The stereogram is depth-dependent horizontal pattern displacement; this version does not implement complete hidden-surface occlusion correction.</p></section>
<script>
async function go(mode){let keys=['subject','size','repeat','shift','style','seed','uploadMode'];let p=new URLSearchParams();for(let k of keys)p.set(k,document.getElementById(k).value);if(mode==='test'){p.set('subject','cube');p.set('size','test')}let r=await fetch('/start?'+p);document.getElementById('status').textContent=await r.text();setTimeout(refresh,600)}
async function uploadImage(){let f=document.getElementById('upload').files[0];if(!f){alert('Choose an image first');return}let b=new FormData();b.append('image',f);document.getElementById('uploadStatus').textContent='Uploading...';try{let r=await fetch('/upload',{method:'POST',body:b});let t=await r.text();if(!r.ok)throw Error(t);document.getElementById('uploadStatus').textContent=t;document.getElementById('subject').value='upload';document.getElementById('uploadPreview').src='/uploaded?ts='+Date.now();document.getElementById('uploadPreview').style.display='block'}catch(e){document.getElementById('uploadStatus').textContent='Upload error: '+e}}
async function refresh(){try{let s=await(await fetch('/status')).json();document.getElementById('status').textContent=s.message;document.getElementById('bar').value=s.progress;document.getElementById('files').innerHTML=s.files.map(f=>'<a href="/file?name='+encodeURIComponent(f)+'" target="_blank">'+f+'</a>').join('');if(s.busy)setTimeout(refresh,1200)}catch(e){document.getElementById('status').textContent=String(e)}}refresh();
</script></body></html>'''

def update(msg,pct=None):
    with LOCK:
        STATE['message']=msg
        if pct is not None: STATE['progress']=pct

def shapes(w,h,subject):
    # Draw normalized depth (white = nearer). Supersample only at target size.
    im=Image.new('L',(w,h),15);d=ImageDraw.Draw(im)
    if subject=='cube':
        cx,cy=w*.51,h*.52; a=min(w,h)*.24
        d.polygon([(cx-a,cy-a*.58),(cx,cy-a),(cx+a,cy-a*.58),(cx,cy-a*.12)],fill=205)
        d.polygon([(cx-a,cy-a*.58),(cx,cy-a*.12),(cx,cy+a),(cx-a,cy+a*.51)],fill=170)
        d.polygon([(cx,cy-a*.12),(cx+a,cy-a*.58),(cx+a,cy+a*.51),(cx,cy+a)],fill=245)
    else:
        # Architectural arch, behind the face.
        x0,x1=int(w*.13),int(w*.87); y0,y1=int(h*.18),int(h*.87)
        d.ellipse((x0,y0,x1,int(h*.66)),fill=82)
        d.rectangle((x0,int(h*.44),x1,y1),fill=82)
        d.ellipse((int(w*.22),int(h*.26),int(w*.78),int(h*.65)),fill=15)
        d.rectangle((int(w*.22),int(h*.47),int(w*.78),int(h*.88)),fill=15)
        # Floating solid cube faces.
        def cube(cx,cy,s):
            a=s
            d.polygon([(cx-a,cy-a*.45),(cx,cy-a*.85),(cx+a,cy-a*.45),(cx,cy-a*.05)],fill=235)
            d.polygon([(cx-a,cy-a*.45),(cx,cy-a*.05),(cx,cy+a),(cx-a,cy+a*.56)],fill=160)
            d.polygon([(cx,cy-a*.05),(cx+a,cy-a*.45),(cx+a,cy+a*.56),(cx,cy+a)],fill=210)
        cube(int(w*.22),int(h*.17),int(w*.085));cube(int(w*.78),int(h*.31),int(w*.095))
        cube(int(w*.22),int(h*.72),int(w*.075))
        # Perspective staircase at lower right.
        for i in range(6):
            y=int(h*(.78+i*.026)); x=int(w*(.51+i*.028));ww=int(w*(.30-i*.025))
            d.polygon([(x,y),(x+ww,y),(x+ww-int(w*.015),y+int(h*.014)),(x,y+int(h*.014))],fill=85+i*19)
        # Face silhouette, ears, neck and sculpted surface bands.
        d.ellipse((int(w*.32),int(h*.28),int(w*.68),int(h*.68)),fill=150)
        d.ellipse((int(w*.305),int(h*.44),int(w*.355),int(h*.52)),fill=115)
        d.ellipse((int(w*.645),int(h*.44),int(w*.695),int(h*.52)),fill=115)
        d.rectangle((int(w*.43),int(h*.64),int(w*.57),int(h*.73)),fill=115)
        # Vertical face depth bulge and cheek highlights: vectorized only within face bounding box.
        xa,xb=int(w*.32),int(w*.68);ya,yb=int(h*.28),int(h*.68)
        xs=np.linspace(-1,1,xb-xa,dtype=np.float32)[None,:]
        ys=np.linspace(-1,1,yb-ya,dtype=np.float32)[:,None]
        ell=xs*xs+ys*ys
        z=148+68*np.sqrt(np.clip(1-ell,0,1))
        z+=25*np.exp(-((xs/.13)**2+((ys-.10)/.32)**2)) # nose
        z+=9*np.exp(-(((xs+.49)/.20)**2+((ys-.13)/.22)**2))
        z+=9*np.exp(-(((xs-.49)/.20)**2+((ys-.13)/.22)**2))
        z-=22*np.exp(-(((xs+.40)/.21)**2+((ys+.22)/.10)**2))
        z-=22*np.exp(-(((xs-.40)/.21)**2+((ys+.22)/.10)**2))
        z-=12*np.exp(-((xs/.31)**2+((ys-.53)/.055)**2)) # mouth
        arr=np.asarray(im).copy();patch=arr[ya:yb,xa:xb];mask=ell<=1;patch[mask]=np.clip(z[mask],0,255).astype(np.uint8);im=Image.fromarray(arr)
    return im

def make_tile(period,style,seed):
    rng=random.Random(seed);size=period
    tile=Image.new('RGB',(size,size),(226,242,237));d=ImageDraw.Draw(tile)
    pal=([(10,49,79),(0,112,147),(245,195,80),(239,108,84),(255,250,227),(55,166,153)] if style=='tile' else [(12,28,45),(242,247,241),(90,112,119),(196,209,206)])
    cell=max(6,period//14)
    for y in range(0,size,cell):
        for x in range(0,size,cell):
            c=rng.choice(pal);d.rectangle((x,y,x+cell-1,y+cell-1),fill=c)
            if rng.random()<.5:
                c2=rng.choice(pal);d.polygon([(x,y),(x+cell,y),(x+cell,y+cell)],fill=c2)
            if rng.random()<.25:
                d.ellipse((x+cell//3,y+cell//3,x+2*cell//3,y+2*cell//3),fill=rng.choice(pal))
    return np.asarray(tile)[size//2].copy(),np.asarray(tile)

def render(opts):
    subj=opts['subject'];sz=opts['size'];seed=int(opts['seed']);style=opts['style']
    if sz=='test':w,h,dpi=1200,800,100
    elif sz=='preview':w,h,dpi=860,1600,20
    else:dpi=int(sz);w,h=43*dpi,80*dpi
    if w*h>85_000_000:raise ValueError('Image too large for this device')
    repeat=float(opts['repeat']);shift=float(opts['shift'])
    if not (1<=repeat<=2.4 and .02<=shift<=.25):raise ValueError('Repeat or shift outside safe UI range')
    # For preview, scale print dimensions to match wall geometry rather than 20dpi exact.
    effective_dpi=(w/43 if subj=='atlas' else dpi)
    period=max(24,int(round(repeat*effective_dpi)))
    maxshift=max(1,int(round(shift*effective_dpi)))
    if maxshift>=period//3:raise ValueError('Depth shift is too large relative to pattern period')
    update('Building depth geometry...',4)
    if subj=='upload':
        with LOCK: path=STATE['upload']
        if not path or not os.path.isfile(path):raise ValueError('Please upload an image first')
        with Image.open(path) as source:
            source=source.convert('RGB')
            # Center-fit on black background; preserve aspect ratio.
            source.thumbnail((w,h),Image.BICUBIC)
            canvas=Image.new('RGB',(w,h),(0,0,0))
            canvas.paste(source,((w-source.width)//2,(h-source.height)//2))
            depth=canvas.convert('L')
            if opts.get('uploadMode')=='brightness':
                # Brightness is NOT a learned/physical depth estimator.
                depth=depth.point(lambda p: min(255,int(p*.85)+15))
    else:
        depth=shapes(w,h,subj)
    stem='atlas_%s_%s_%s'%(subj,sz,time.strftime('%Y%m%d_%H%M%S'))
    depfile=stem+'_depth.png';depth.save(os.path.join(BASE,depfile))
    update('Encoding stereogram rows...',9)
    # Create unique texture columns. Horizontal source coordinates are shifted by depth.
    # This is a row-local disparity warp, NOT a complete occlusion-aware renderer.
    _,tile=make_tile(period,style,seed)
    result=Image.new('RGB',(w,h))
    depth_np=np.asarray(depth)
    xx=np.arange(w,dtype=np.int32)
    for y in range(h):
        drow=depth_np[y].astype(np.int32)
        offsets=(drow*maxshift//255)
        idx=(xx-offsets)%period
        # Tile row varies vertically; avoids featureless horizontal stripes.
        result.paste(Image.fromarray(np.ascontiguousarray(tile[y%period,idx,:][None,:,:]),mode='RGB'),(0,y))
        if y%max(1,h//30)==0:update('Encoding stereogram rows... %d%%'%(100*y//h),10+int(77*y/h))
    png=stem+'.png';result.save(os.path.join(BASE,png),optimize=False)
    update('Writing PDF...',91)
    pdf=stem+'.pdf'
    try:result.save(os.path.join(BASE,pdf),'PDF',resolution=float(dpi))
    except Exception as exc:pdf=None;update('PDF export unavailable: '+str(exc),95)
    with LOCK:
        STATE['files']=[png,depfile]+([pdf] if pdf else [])+STATE['files']
        STATE['files']=STATE['files'][:30]
        STATE['progress']=100;STATE['busy']=False;STATE['message']='Complete: '+png+'\nSaved in '+BASE

def worker(opts):
    try:render(opts)
    except Exception as e:
        with LOCK:STATE['busy']=False;STATE['message']='ERROR: '+str(e)+'\n'+traceback.format_exc()[-1200:];STATE['progress']=0

class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        u=urlparse(self.path)
        if u.path=='/':return self.send_data(HTML.encode(),'text/html; charset=utf-8')
        if u.path=='/uploaded':
            with LOCK: path=STATE['upload']
            if not path or not os.path.isfile(path):return self.send_data(b'No upload','text/plain',404)
            with Image.open(path) as img:
                im=img.copy();im.thumbnail((650,650),Image.BICUBIC)
                buf=io.BytesIO();im.convert('RGB').save(buf,'PNG')
            return self.send_data(buf.getvalue(),'image/png')
        if u.path=='/status':return self.send_data(json.dumps(STATE).encode(),'application/json')
        if u.path=='/start':
            with LOCK:
                if STATE['busy']:return self.send_data(b'Already generating. Please wait.','text/plain')
                STATE['busy']=True;STATE['progress']=0;STATE['message']='Starting...'
            p=parse_qs(u.query)
            opts={k:p.get(k,[v])[0] for k,v in {'subject':'upload','size':'preview','repeat':'1.85','shift':'.12','style':'tile','seed':'42','uploadMode':'depth'}.items()}
            if opts['subject'] not in ('cube','atlas','upload') or opts['size'] not in ('test','preview','75','100','150') or opts['style'] not in ('tile','mono') or opts['uploadMode'] not in ('depth','brightness'):
                STATE['busy']=False;return self.send_data(b'Invalid settings','text/plain',400)
            threading.Thread(target=worker,args=(opts,),daemon=True).start()
            return self.send_data(b'Generating... check status below.','text/plain')
        if u.path=='/file':
            name=parse_qs(u.query).get('name',[''])[0]
            if name!=os.path.basename(name) or not name.endswith(('.png','.pdf')):return self.send_data(b'Invalid file','text/plain',400)
            path=os.path.join(BASE,name)
            if not os.path.isfile(path):return self.send_data(b'Not found','text/plain',404)
            with open(path,'rb') as f:data=f.read()
            return self.send_data(data,'application/pdf' if name.endswith('.pdf') else 'image/png')
        return self.send_data(b'Not found','text/plain',404)
    def do_POST(self):
        if urlparse(self.path).path!='/upload':return self.send_data(b'Not found','text/plain',404)
        try:
            n=int(self.headers.get('Content-Length','0'))
            if n<=0 or n>16*1024*1024:raise ValueError('Image must be under 16 MB')
            form=cgi.FieldStorage(fp=self.rfile,headers=self.headers,environ={'REQUEST_METHOD':'POST','CONTENT_TYPE':self.headers.get('Content-Type',''),'CONTENT_LENGTH':str(n)},keep_blank_values=True)
            if 'image' not in form:raise ValueError('No image received')
            raw=form['image'].file.read(16*1024*1024+1)
            if len(raw)>16*1024*1024:raise ValueError('Image too large')
            with Image.open(io.BytesIO(raw)) as img:
                img.verify()
            with Image.open(io.BytesIO(raw)) as img:
                if img.width*img.height>40_000_000:raise ValueError('Image dimensions too large')
                img=img.convert('RGB')
                path=os.path.join(BASE,'uploaded_source.png')
                img.save(path,'PNG')
            with LOCK:
                STATE['upload']=path
                STATE['upload_name']=str(form['image'].filename or 'image')[:120]
            return self.send_data(('Uploaded: '+STATE['upload_name']).encode(),'text/plain')
        except Exception as e:return self.send_data(('Upload failed: '+str(e)).encode(),'text/plain',400)
    def send_data(self,data,ctype,status=200):
        self.send_response(status);self.send_header('Content-Type',ctype);self.send_header('Content-Length',str(len(data)));self.end_headers();self.wfile.write(data)
    def log_message(self,*args):pass

if __name__=='__main__':
    print('Atlas Parallax Studio: http://127.0.0.1:%d'%PORT)
    print('Files: '+BASE)
    HTTPServer(('127.0.0.1',PORT),Handler).serve_forever()
