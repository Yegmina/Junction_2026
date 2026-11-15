"""Stream the retained benchmark images to private GCS without a local tarball."""
from datetime import datetime, timezone
import base64
import hashlib
import json
from pathlib import Path
import subprocess
import urllib.parse
import urllib.request
import uuid


def main():
    from aerorozum.common import save, utcnow
    from cloud.airborne_worker import metadata_record_store
    root = Path('/opt/aerorozum')
    night = root / 'artifacts/overnight-20261115T000942'
    provenance = json.loads((night / 'native-prebuilt-recovery-v1.json').read_text())
    images = provenance['images']
    for image in images.values():
        actual = json.loads(subprocess.check_output(['docker', 'image', 'inspect', image], text=True))[0]
        assert actual['Id'] == image and actual['Os'] == 'linux'
    bucket = 'aerorozum-thermal-rl-avid-relic-509408-e4'
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    name = ('research/overnight-20261115/native-runtime-images/' + stamp + '-' + uuid.uuid4().hex[:8]
            + '/original-images.tar.gz')
    uri = 'gs://' + bucket + '/' + name
    docker = subprocess.Popen(['docker', 'image', 'save', *images.values()], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    zipper = subprocess.Popen(['gzip', '-1', '-n'], stdin=docker.stdout, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    docker.stdout.close()
    upload = subprocess.Popen(['/snap/bin/gcloud', 'storage', 'cp', '-', uri, '--quiet'],
                              stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    sha = hashlib.sha256()
    md5 = hashlib.md5()
    size = 0
    try:
        while block := zipper.stdout.read(2 * 1024 * 1024):
            sha.update(block)
            md5.update(block)
            size += len(block)
            upload.stdin.write(block)
        upload.stdin.close()
        upload.stdin = None
        stdout, stderr = upload.communicate(timeout=600)
        assert upload.returncode == 0, stderr.decode(errors='replace')[-2000:]
        assert zipper.wait(timeout=60) == 0, zipper.stderr.read().decode(errors='replace')[-2000:]
        assert docker.wait(timeout=60) == 0, docker.stderr.read().decode(errors='replace')[-2000:]
        store = metadata_record_store(bucket, prefix='research/overnight-20261115/native-runtime-images/manifests')
        request = urllib.request.Request('https://storage.googleapis.com/storage/v1/b/' + bucket +
                                         '/o/' + urllib.parse.quote(name, safe=''), headers=store._headers())
        with urllib.request.urlopen(request, timeout=60) as response:
            remote = json.load(response)
        assert int(remote['size']) == size
        assert remote['md5Hash'] == base64.b64encode(md5.digest()).decode()
        result = dict(at=utcnow(), status='server_digest_verified', uri=uri, generation=remote['generation'],
                      size_bytes=size, sha256=sha.hexdigest(), md5=remote['md5Hash'], images=images,
                      archive_format='docker image save | gzip -1 -n', local_tarball_created=False,
                      original_images_deleted=False, active_containers_changed=False,
                      restore_test_performed=False, original_published_submission_unchanged=True)
        store.put(sha.hexdigest(), result, 'progress')
        save(night / 'native-runtime-image-archive.json', result)
        print(json.dumps(result), flush=True)
    finally:
        for process in [upload, zipper, docker]:
            if process.poll() is None:
                process.terminate()


if __name__ == '__main__':
    main()
