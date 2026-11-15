"""Read-only guest evidence. No package installs, baseline resets or policy edits."""
import datetime,hashlib,json,struct,subprocess
from pathlib import Path

def cmd(args):
 p=subprocess.run(args,capture_output=True,text=True);return {'returncode':p.returncode,'stdout':p.stdout[-30000:],'stderr':p.stderr[-1000:]}
def fingerprint(path):
 p=Path(path)
 if not p.exists():return None
 b=p.read_bytes();return {'path':str(p),'bytes':len(b),'sha256':hashlib.sha256(b).hexdigest(),'mtime_utc':datetime.datetime.fromtimestamp(p.stat().st_mtime,datetime.timezone.utc).isoformat()}
d={'collected_at':datetime.datetime.now(datetime.timezone.utc).isoformat(),'read_only':True,'boot_id':Path('/proc/sys/kernel/random/boot_id').read_text().strip(),'uname':cmd(['uname','-a']),'boots':cmd(['journalctl','--list-boots','--no-pager'])}
for index,name in [('0','current_boot'),('-1','previous_boot')]:d[name]=cmd(['journalctl','-k','-b',index,'--grep','Secure boot|secureboot|EFI v|Linux version|integrity:|Mok|sbat|shim|UEFI','-o','short-iso','--no-pager'])
d['packages']=cmd(['dpkg-query','-W','-f=${binary:Package} ${Version} ${db:Status-Status}\n','shim-signed','shim','grub-efi-amd64-signed','grub-efi-amd64-bin','fwupd','fwupd-signed','linux-image-gcp','nvidia-kernel-common-580-server']);d['fwupd']=cmd(['systemctl','status','fwupd.service','--no-pager'])
history=Path('/var/log/apt/history.log').read_text();d['apt_nov14']=[block for block in history.split('\n\n') if 'Start-Date: 2026-11-14' in block]
terms=('shim','grub','fwupd','mok','secureboot','linux-image','nvidia-kernel');d['boot_security_package_history']=[line for line in Path('/var/log/dpkg.log').read_text().splitlines() if any(term in line.lower() for term in terms)]
d['efi_binaries']=[x for x in (fingerprint(p) for p in Path('/boot/efi/EFI').glob('*/*.efi')) if x]
variables=[]
for p in sorted(Path('/sys/firmware/efi/efivars').iterdir()):
 name=p.name.rsplit('-',5)[0]
 if not(name in ('SecureBoot','SetupMode','DeployedMode','AuditMode','PK','KEK','db','dbx') or name.startswith(('Mok','Sbat'))):continue
 b=p.read_bytes();variables.append({'name':p.name,'bytes':len(b),'sha256':hashlib.sha256(b).hexdigest(),'value':list(b[4:]) if len(b)<=12 else None,'mtime_note':'Current-boot creation time; not proof of when firmware contents changed'})
d['efi_variables']=variables
p=Path('/sys/kernel/security/tpm0/binary_bios_measurements');d['tpm_event_log']=fingerprint(p);events=[]
if p.exists():
 try:
  b=p.read_bytes();first_size=struct.unpack_from('<I',b,28)[0];spec=b[32:32+first_size];count=struct.unpack_from('<I',spec,24)[0];sizes={struct.unpack_from('<H',spec,28+i*4)[0]:struct.unpack_from('<H',spec,30+i*4)[0] for i in range(count)};offset=32+first_size;pcr=bytes(32);pcr1=bytes(20)
  while offset+12<=len(b):
   index,kind,n=struct.unpack_from('<III',b,offset);offset+=12;digests={}
   for _ in range(n):
    alg=struct.unpack_from('<H',b,offset)[0];offset+=2;size=sizes[alg];digests[alg]=b[offset:offset+size];offset+=size
   size=struct.unpack_from('<I',b,offset)[0];offset+=4;payload=b[offset:offset+size];offset+=size
   if index!=7:continue
   item={'pcr':index,'event_type':hex(kind),'event_bytes':size,'sha256':digests.get(11,b'').hex(),'sha1':digests.get(4,b'').hex()}
   if kind in (0x80000001,0x800000e0) and len(payload)>=32:
    name_len,data_len=struct.unpack_from('<QQ',payload,16);item['variable_name']=payload[32:32+name_len*2].decode('utf-16-le',errors='replace');item['variable_data_sha256']=hashlib.sha256(payload[32+name_len*2:]).hexdigest()
   if 11 in digests and kind!=3:pcr=hashlib.sha256(pcr+digests[11]).digest()
   if 4 in digests and kind!=3:pcr1=hashlib.sha1(pcr1+digests[4]).digest()
   item['sha1_pcr_after_event']=pcr1.hex()
   events.append(item)
  d['pcr7_events']=events;d['pcr7_sha256_replayed']=pcr.hex();d['pcr7_sha1_replayed']=pcr1.hex()
 except Exception as ex:d['tpm_parse_error']=str(ex)
print(json.dumps(d))
