"""Expose the existing mission UI on a public HTTPS IP with a separate demo login.

Open TCP 80/443 only for the demo VM's network tag before running this script.
No application or research worker is restarted. Requires root on Ubuntu 24.04.
"""
import argparse
import grp
import ipaddress
import json
import os
from pathlib import Path
import secrets
import subprocess


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
    private=Path('/etc/aerorozum');private.mkdir(mode=0o700,exist_ok=True)
    cred=private/'demo-access.json'
    if cred.exists():
        login=json.loads(cred.read_text())
    else:
        login=dict(username='demo',password=secrets.token_urlsafe(12),url='https://'+ip)
        cred.write_text(json.dumps(login));cred.chmod(0o600)
    auth=Path('/etc/nginx/aerorozum-demo.htpasswd')
    run('htpasswd','-i','-B','-c',str(auth),login['username'],
        input=login['password']+'\n',text=True,capture_output=True)
    auth.chmod(0o640);os.chown(auth,0,grp.getgrnam('www-data').gr_gid)
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
    config.write_text('limit_req_zone $binary_remote_addr zone=aerorozum_demo:10m rate=30r/s;\n'+http+f'''
server {{
    listen 443 ssl http2;
    server_name {ip};
    ssl_certificate {cert}/fullchain.pem;
    ssl_certificate_key {cert}/privkey.pem;
    ssl_protocols TLSv1.2 TLSv1.3;
    auth_basic "AeroRozum demo";
    auth_basic_user_file {auth};
    client_max_body_size 1m;
    add_header X-Content-Type-Options nosniff always;
    location / {{
        limit_req zone=aerorozum_demo burst=60 nodelay;
        proxy_pass http://127.0.0.1:18901;
        proxy_http_version 1.1;
        proxy_set_header Host $host;
        proxy_set_header X-Forwarded-Proto https;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_buffering off;
        proxy_read_timeout 240s;
    }}
}}
''')
    run('nginx','-t');run('systemctl','reload','nginx')
    units=Path('/etc/systemd/system')
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
    print(json.dumps(dict(url='https://'+ip,username=login['username'],credentials_file=str(cred))))


if __name__=='__main__':
    main()
