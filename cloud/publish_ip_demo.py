"""Expose the mission UI on HTTPS with invisible automatic guest credentials.

Open TCP 80/443 only for the demo VM's network tag before running this script.
No application or research worker is restarted. Requires root on Ubuntu 24.04.
"""
import argparse
import ipaddress
import json
import os
from pathlib import Path
import pwd
import shutil
import subprocess
import time
import urllib.request


def run(*args, **kwargs):
    return subprocess.run(args, check=True, **kwargs)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--ip',required=True)
    args=parser.parse_args()
    ip=str(ipaddress.IPv4Address(args.ip))
    if os.geteuid()!=0:
        raise SystemExit('Run on the VM as root')
    base=Path('/opt/aerorozum/deployment')
    guest_root=base/'guest-sessions';guest_root.mkdir(mode=0o700,exist_ok=True)
    user=pwd.getpwnam('aerorozum-app');os.chown(guest_root,user.pw_uid,user.pw_gid)
    guest_script=base/'releases/48h-v1/cloud/guest_sessions.py'
    source=Path(__file__).with_name('guest_sessions.py')
    if source.resolve()!=guest_script.resolve():shutil.copy2(source,guest_script)
    units=Path('/etc/systemd/system')
    (units/'aerorozum-demo-guests.service').write_text(
        '[Unit]\nDescription=AeroRozum automatic guest identities\nAfter=network.target\n'
        '[Service]\nUser=aerorozum-app\nUMask=0077\nRestart=always\nRestartSec=2\n'
        'Environment=AEROROZUM_GUEST_ROOT='+str(guest_root)+'\n'
        'ExecStart='+str(base/'venv/bin/python')+' '+str(guest_script)+'\n'
        'NoNewPrivileges=true\nProtectHome=true\nPrivateTmp=true\nProtectSystem=strict\n'
        'ReadWritePaths='+str(guest_root)+'\nMemoryMax=256M\nCPUQuota=100%\n'
        '[Install]\nWantedBy=multi-user.target\n')
    run('systemctl','daemon-reload');run('systemctl','enable','aerorozum-demo-guests.service')
    run('systemctl','restart','aerorozum-demo-guests.service')
    for attempt in range(30):
        try:
            with urllib.request.urlopen('http://127.0.0.1:18950/session',timeout=1):break
        except OSError:
            if attempt==29:raise
            time.sleep(.1)
    web=Path('/var/www/aerorozum-acme');web.mkdir(parents=True,exist_ok=True)
    config=Path('/etc/nginx/sites-available/aerorozum-public')
    http=f'''server {{
    listen 80;
    server_name {ip};
    location /.well-known/acme-challenge/ {{ root {web}; }}
    location / {{ return 301 https://{ip}$request_uri; }}
}}
'''
    config.write_text(http)
    link=Path('/etc/nginx/sites-enabled/aerorozum-public')
    if not link.exists():link.symlink_to(config)
    run('nginx','-t');run('systemctl','reload','nginx')
    certbot=base/'certbot-venv/bin/certbot'
    if not certbot.exists():
        run('/usr/bin/python3','-m','venv',str(certbot.parents[1]))
        run(str(certbot.with_name('pip')),'install','--disable-pip-version-check','certbot>=5.4,<6')
    cert=Path('/etc/letsencrypt/live')/ip
    if not (cert/'fullchain.pem').exists():
        run(str(certbot),'certonly','--non-interactive','--agree-tos',
            '--register-unsafely-without-email','--preferred-profile','shortlived',
            '--webroot','--webroot-path',str(web),'--ip-address',ip)
    config.write_text('limit_req_zone $binary_remote_addr zone=aerorozum_demo:10m rate=30r/s;\n'
        'map $uri $aerorozum_document_cache { default ""; / "no-store"; }\n'+http+f'''
server {{
    listen 443 ssl http2;
    server_name {ip};
    ssl_certificate {cert}/fullchain.pem;
    ssl_certificate_key {cert}/privkey.pem;
    ssl_protocols TLSv1.2 TLSv1.3;
    client_max_body_size 1m;
    add_header X-Content-Type-Options nosniff always;
    add_header Cache-Control $aerorozum_document_cache always;
    add_header Set-Cookie $aerorozum_guest_cookie always;
    location = /_aerorozum_guest {{
        internal;
        proxy_pass http://127.0.0.1:18950/session;
        proxy_pass_request_body off;
        proxy_set_header Content-Length "";
        proxy_set_header Cookie $http_cookie;
        proxy_set_header Authorization "";
        proxy_set_header X-Original-URI $request_uri;
    }}
    location / {{
        limit_req zone=aerorozum_demo burst=60 nodelay;
        auth_request /_aerorozum_guest;
        auth_request_set $aerorozum_guest_cookie $upstream_http_set_cookie;
        auth_request_set $aerorozum_guest_user $upstream_http_x_aerorozum_guest;
        proxy_pass http://127.0.0.1:18901;
        proxy_http_version 1.1;
        proxy_set_header Host $host;
        proxy_set_header X-Forwarded-Proto https;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header Authorization "";
        proxy_set_header X-AeroRozum-Guest $aerorozum_guest_user;
        proxy_buffering off;
        proxy_read_timeout 240s;
    }}
}}
''')
    run('nginx','-t');run('systemctl','reload','nginx')
    (units/'aerorozum-demo-cert-renew.service').write_text(
        '[Unit]\nDescription=Renew AeroRozum public IP certificate\n'
        '[Service]\nType=oneshot\nExecStart='+str(certbot)+
        ' renew --quiet --deploy-hook "/usr/sbin/nginx -s reload"\n')
    (units/'aerorozum-demo-cert-renew.timer').write_text(
        '[Unit]\nDescription=Check AeroRozum certificate twice daily\n'
        '[Timer]\nOnCalendar=*-*-* 00,12:00:00\nRandomizedDelaySec=300\nPersistent=true\n'
        '[Install]\nWantedBy=timers.target\n')
    run('systemctl','daemon-reload')
    run('systemctl','enable','--now','aerorozum-demo-cert-renew.timer')
    print(json.dumps(dict(url='https://'+ip,access='automatic_guests',login_prompt=False)))


if __name__=='__main__':
    main()
