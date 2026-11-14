"""Read-only cloud result mirror; never launches GPU or model work."""
import subprocess,time
from aerorozum.common import data_root

def main():
    root=data_root()/'mission-eval/paired-decisions-v1/results';root.mkdir(parents=True,exist_ok=True)
    command=['/home/yehormsi/.local/google-cloud-cli-install/google-cloud-sdk/bin/gcloud','storage','rsync','-r','gs://aerorozum-thermal-rl-avid-relic-509408-e4/mission-eval/paired-decisions-v1',str(root),'--quiet']
    blank=root.parents[1]/'paired-decisions-blank-v1/results';blank.mkdir(parents=True,exist_ok=True)
    blank_command=command.copy();blank_command[-3]='gs://aerorozum-thermal-rl-avid-relic-509408-e4/mission-eval/paired-decisions-blank-v1';blank_command[-2]=str(blank)
    deadline=time.monotonic()+1450
    while time.monotonic()<deadline:
        for current in (command,blank_command):
            try:subprocess.run(current,capture_output=True,timeout=45,check=True)
            except (subprocess.SubprocessError,OSError) as error:print(type(error).__name__,flush=True)
        time.sleep(20)

if __name__=='__main__':main()
