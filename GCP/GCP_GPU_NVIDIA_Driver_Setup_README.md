# GCP GPU VM NVIDIA Driver Setup

This README documents the NVIDIA driver setup and troubleshooting steps used on a Google Cloud GPU VM.

## Target VM

Observed target:

```text
Cloud: Google Cloud Platform
Machine type: g2-standard-16
GPU: NVIDIA L4
OS: Ubuntu 26.04 LTS
Kernel: 7.0.0-1005-gcp
Secure Boot: disabled
Driver target used: nvidia-driver-595-open / NVIDIA 595.71.05
```

Final working state:

```text
nvidia-smi works
NVIDIA L4 visible
NVIDIA driver loaded: 595.71.05
CUDA visible through driver: 13.2
Kernel driver in use: nvidia
nouveau not loaded
```

---

## Initial symptoms

The VM had the GPU attached, but `nvidia-smi` failed:

```bash
nvidia-smi
```

Output:

```text
NVIDIA-SMI has failed because it couldn't communicate with the NVIDIA driver.
Make sure that the latest NVIDIA driver is installed and running.
```

`modinfo` also showed that the actual kernel module was missing:

```bash
modinfo nvidia | head
```

Output:

```text
modinfo: ERROR: Module nvidia not found.
```

`lspci` showed the NVIDIA L4 device, but no NVIDIA driver in use:

```bash
lspci -k | grep -EA4 'VGA|3D|Display|NVIDIA'
```

Observed output:

```text
00:03.0 3D controller: NVIDIA Corporation AD104GL [L4] (rev a1)
        Subsystem: NVIDIA Corporation Device 16ee
        Kernel modules: nvidiafb, nouveau
```

This meant:

```text
GPU attached: yes
nvidia-smi binary installed: yes
NVIDIA kernel module loaded: no
NVIDIA kernel module available: initially no
Secure Boot blocker: no, Secure Boot was disabled
```

---

## Key diagnosis

Installing only `nvidia-utils-595` is not enough.

That package provides NVIDIA userspace utilities such as `nvidia-smi`, but the actual kernel driver must also exist and load:

```text
nvidia.ko
nvidia_uvm.ko
nvidia_drm.ko
nvidia_modeset.ko
```

The missing part was the actual NVIDIA kernel module path and/or the module loading step.

---

## Check OS, kernel, and installed NVIDIA packages

Run:

```bash
cat /etc/os-release
uname -r
dpkg -l | grep -Ei 'nvidia|cuda|libnvidia' || true
```

Observed values:

```text
PRETTY_NAME="Ubuntu 26.04 LTS"
VERSION_ID="26.04"
VERSION_CODENAME=resolute
Kernel: 7.0.0-1005-gcp
```

Partial package state initially included only userspace/common pieces:

```text
libnvidia-compute-595
nvidia-firmware-595-595.71.05
nvidia-kernel-common-595
nvidia-utils-595
```

The full working package set later included:

```text
nvidia-dkms-595-open
nvidia-driver-595-open
nvidia-kernel-common-595
nvidia-kernel-source-595-open
```

Check with:

```bash
dpkg -l | grep -Ei 'nvidia-driver|nvidia-dkms|linux-modules-nvidia|linux-objects-nvidia|nvidia-kernel'
```

Expected relevant output:

```text
ii  nvidia-dkms-595-open          595.71.05-0ubuntu0.26.04.1  amd64  NVIDIA DKMS package (open kernel module)
ii  nvidia-driver-595-open        595.71.05-0ubuntu0.26.04.1  amd64  NVIDIA driver (open kernel) metapackage
ii  nvidia-kernel-common-595      595.71.05-0ubuntu0.26.04.1  amd64  Shared files used with the kernel module
ii  nvidia-kernel-source-595-open 595.71.05-0ubuntu0.26.04.1  amd64  NVIDIA kernel source package
```

---

## Install missing driver/kernel module packages

If the full package set is not present, install it:

```bash
sudo apt update
sudo apt install -y "linux-headers-$(uname -r)" build-essential dkms
sudo apt install -y nvidia-driver-595-open
```

Then re-check:

```bash
dpkg -l | grep -Ei 'nvidia-driver|nvidia-dkms|linux-modules-nvidia|linux-objects-nvidia|nvidia-kernel'
```

---

## Check DKMS build status

Run:

```bash
dkms status | grep -i nvidia || true
```

Working output observed:

```text
nvidia/595.71.05, 7.0.0-1003-gcp, x86_64: installed
nvidia/595.71.05, 7.0.0-1005-gcp, x86_64: installed
```

The important line is the one matching the running kernel:

```bash
uname -r
```

For this VM:

```text
7.0.0-1005-gcp
```

So this must exist:

```text
nvidia/595.71.05, 7.0.0-1005-gcp, x86_64: installed
```

If DKMS is missing for the current kernel, run:

```bash
sudo apt update
sudo apt install -y "linux-headers-$(uname -r)" build-essential dkms
sudo dkms autoinstall -k "$(uname -r)"
```

Then refresh module dependencies:

```bash
sudo depmod -a "$(uname -r)"
```

---

## Refresh module index and verify `modinfo`

This was the key fix on the VM.

Run:

```bash
sudo depmod -a "$(uname -r)"
modinfo -k "$(uname -r)" nvidia | head
```

Working output observed:

```text
filename:       /lib/modules/7.0.0-1005-gcp/updates/dkms/nvidia.ko.zst
import_ns:      DMA_BUF
alias:          char-major-195-*
description:    NVIDIA core GPU kernel module
version:        595.71.05
supported:      external
license:        Dual MIT/GPL
firmware:       nvidia/595.71.05/gsp_tu10x.bin
firmware:       nvidia/595.71.05/gsp_ga10x.bin
softdep:        pre: ecdh_generic,ecdsa_generic
```

If `modinfo` still says module not found, inspect the module tree:

```bash
find /lib/modules/"$(uname -r)" -type f -iname '*nvidia*.ko*' -print
find /lib/modules -type f -iname '*nvidia*.ko*' -print | head -50
```

---

## Load NVIDIA modules manually

Run:

```bash
sudo modprobe -v nvidia
sudo modprobe -v nvidia_uvm
sudo modprobe -v nvidia_drm
```

Observed successful output:

```text
insmod /lib/modules/7.0.0-1005-gcp/updates/dkms/nvidia.ko.zst NVreg_PreserveVideoMemoryAllocations=1 NVreg_TemporaryFilePath=/var
insmod /lib/modules/7.0.0-1005-gcp/updates/dkms/nvidia-uvm.ko.zst
insmod /lib/modules/7.0.0-1005-gcp/updates/dkms/nvidia-drm.ko.zst modeset=1
```

Then test:

```bash
nvidia-smi
```

Working output showed:

```text
NVIDIA-SMI 595.71.05
Driver Version: 595.71.05
CUDA Version: 13.2
GPU: NVIDIA L4
Memory: 23034MiB
```

---

## Confirm loaded modules

Run:

```bash
lsmod | grep -E 'nvidia|nouveau'
```

Working output observed:

```text
nvidia_drm            147456  0
nvidia_uvm           2088960  0
drm_ttm_helper         20480  1 nvidia_drm
nvidia_modeset       1736704  1 nvidia_drm
video                  77824  1 nvidia_modeset
nvidia              14729216  2 nvidia_uvm,nvidia_modeset
```

Important points:

```text
nvidia loaded: yes
nvidia_uvm loaded: yes
nvidia_drm loaded: yes
nouveau loaded: no
```

---

## Make NVIDIA modules load on boot

Create a modules-load file:

```bash
cat <<'EOF' | sudo tee /etc/modules-load.d/nvidia.conf
nvidia
nvidia_uvm
nvidia_drm
EOF
```

This asks systemd to load these modules at boot.

---

## Blacklist nouveau

Even though nouveau was not loaded after the fix, blacklist it to avoid future races:

```bash
cat <<'EOF' | sudo tee /etc/modprobe.d/blacklist-nouveau.conf
blacklist nouveau
options nouveau modeset=0
EOF

sudo update-initramfs -u
```

A reboot may be needed for the blacklist/initramfs change to fully apply, but do not reboot until you have validated SSH access and module loading.

---

## Optional: enable NVIDIA persistence mode

For a headless GCP GPU VM:

```bash
sudo nvidia-smi -pm 1
```

Validate:

```bash
nvidia-smi
```

---

## Reboot safety notes

If you do not have GCP console/dashboard access, avoid unnecessary reboot until the driver is known-good.

Before rebooting, confirm:

```bash
modinfo -k "$(uname -r)" nvidia | head
sudo modprobe -v nvidia
nvidia-smi
```

Use a delayed reboot so you can cancel if needed:

```bash
sudo shutdown -r +1 "Rebooting to activate NVIDIA driver"
```

Cancel if needed:

```bash
sudo shutdown -c
```

After reconnecting:

```bash
nvidia-smi
lsmod | grep -E 'nvidia|nouveau'
lspci -k | grep -EA4 'VGA|3D|Display|NVIDIA'
```

---

## Google CUDA installer notes

The Google installer was also tested.

The command that failed:

```bash
sudo python3 cuda_installer.pyz install_driver \
  --installation-mode=repo \
  --installation-branch=lts
```

Error:

```text
RuntimeError: The LTS driver branch is supported only in binary installation mode.
Please use --installation-mode=binary and --installation-branch=lts to install LTS driver branch.
```

Correct command if using the Google installer LTS path:

```bash
sudo python3 cuda_installer.pyz install_driver \
  --installation-mode=binary \
  --installation-branch=lts
```

However, for this VM the successful path was the Ubuntu 595-open DKMS package path plus:

```bash
sudo depmod -a "$(uname -r)"
sudo modprobe -v nvidia
sudo modprobe -v nvidia_uvm
sudo modprobe -v nvidia_drm
```

---

## Troubleshooting commands

If `nvidia-smi` fails again:

```bash
nvidia-smi
modinfo -k "$(uname -r)" nvidia | head
sudo modprobe -v nvidia 2>&1 || true
dkms status | grep -i nvidia || true
lsmod | grep -E 'nvidia|nouveau' || true
lspci -k | grep -EA4 'VGA|3D|Display|NVIDIA'
```

Kernel logs:

```bash
journalctl -k -b | grep -Ei 'nvidia|nouveau|dkms|module|firmware|taint|secure|mok|gcc|headers|gsp|rm' | tail -200
```

DKMS build log:

```bash
sudo tail -n 160 /var/lib/dkms/nvidia/595.71.05/build/make.log
```

Find installed NVIDIA kernel objects:

```bash
find /lib/modules/"$(uname -r)" -type f -iname '*nvidia*.ko*' -print
```

---

## Docker GPU validation

After `nvidia-smi` works on the host, validate Docker GPU access:

```bash
docker run --rm --gpus all nvidia/cuda:12.6.3-base-ubuntu24.04 nvidia-smi
```

Expected:

```text
NVIDIA-SMI output appears inside the container
GPU: NVIDIA L4
```

If Docker cannot use `--gpus all`, install/configure NVIDIA Container Toolkit before running Ollama.

---

## Ollama Docker example

Only run this after host `nvidia-smi` and Docker GPU passthrough work.

```bash
docker volume create ollama

docker rm -f ollama 2>/dev/null || true

docker run -d \
  --name ollama \
  --restart unless-stopped \
  --gpus all \
  -e OLLAMA_HOST=0.0.0.0:11434 \
  -v ollama:/root/.ollama \
  -p 0.0.0.0:11434:11434 \
  ollama/ollama:latest
```

Validate:

```bash
curl -s http://127.0.0.1:11434/api/tags | jq .
docker logs --tail 100 ollama
```

Small first test model:

```bash
docker exec -it ollama ollama pull llama3.2:3b
docker exec -it ollama ollama run llama3.2:3b "Reply with OK only"
```

Then try a mid-size model:

```bash
docker exec -it ollama ollama pull deepseek-r1:14b
docker exec -it ollama ollama run deepseek-r1:14b "Reply with OK only"
```

---

## Final validation checklist

Run:

```bash
mokutil --sb-state || true
nvidia-smi
modinfo -k "$(uname -r)" nvidia | head
dkms status | grep -i nvidia || true
lsmod | grep -E 'nvidia|nouveau'
lspci -k | grep -EA4 'VGA|3D|Display|NVIDIA'
```

Expected:

```text
SecureBoot disabled
nvidia-smi works
modinfo nvidia returns /lib/modules/.../nvidia.ko.zst
DKMS installed for current kernel
nvidia/nvidia_uvm/nvidia_drm loaded
nouveau not loaded
lspci shows NVIDIA L4
```
SSH
```text
ssh -i ~/.ssh/pkoumantos pkoumantos@34.12.39.3 - Front VM
ssh -i ~/.ssh/pkoumantos pkoumantos@34.6.127.12 - Antonis
ssh -i ~/.ssh/pkoumantos pkoumantos@34.79.223.15 - GPU VM
```
---
DOCS
---
```text
RAG_KEY="$(
  sudo docker exec recitals-rag-api \
    printenv RAG_API_KEY
)"

curl -fsS \
  http://127.0.0.1:8010/v1/chat/completions \
  -H "Authorization: Bearer ${RAG_KEY}" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "recitals-rag",
    "stream": false,
    "messages": [
      {
        "role": "user",
        "content": "Which component provides the RECITALS user-facing chat interface?"
      }
    ]
  }' |
  python3 -m json.tool

unset RAG_KEY
{
    "id": "chatcmpl-cf8b719dc9234ac58847a4a5f2f83480",
    "object": "chat.completion",
    "created": 1788260843,
    "model": "recitals-rag",
    "choices": [
        {
            "index": 0,
            "message": {
                "role": "assistant",
                "content": "The Onyx component provides the RECITALS user-facing chat interface [1].\n\nSources:\n[1] recitals-test.md"
            },
            "finish_reason": "stop"
        }
    ],
    "usage": {
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "total_tokens": 0
    }
}
```
---
```text
onyx_pat_r1A2nru0qKlPHyhtxCPISbdxgsthFYKrD25Ijy1VXWHr6GoQnUlUtAwiJK3WR0kGCKlhOQ1WzGqrv41wiuB5QDIEWxcY8HNFgKFgCdv99kp25wDZrUrdEKXJ1B5jCqF4Ed4RkBWHP88q1PSUKwKuNSSayM1mYEcfk3nq1AjgV9qE9HkD_b0jK3j04FThxMAw0c44SYQeu65fwyHM643vQwFbLU_SXzdboIM9G48AYBG4odAM6sXUNxM9uoHNcDcD

read -rsp 'Onyx PAT: ' ONYX_PAT
printf '\n'

curl -fsS \
  --max-time 300 \
  http://34.12.39.3/api/chat/send-chat-message \
  -H "Authorization: Bearer ${ONYX_PAT}" \
  -H 'Content-Type: application/json' \
  -d '{
    "message": "According to the component integration specification, which partner is responsible for the LLM component and who is the lead technical contact?",
    "chat_session_info": {
      "persona_id": 0
    },
    "llm_override": {
      "model_provider": "openai_compatible",
      "model_version": "recitals-rag",
      "temperature": 0
    },
    "allowed_tool_ids": [],
    "file_descriptors": [],
    "stream": false,
    "include_citations": true,
    "origin": "api"
  }' |
  jq '{
    chat_session_id,
    message_id,
    answer,
    error_msg
  }'

unset ONYX_PAT
Onyx PAT:
{
  "chat_session_id": "d4218e62-7de0-43c3-9cd4-1b0bc0a7c35b",
  "message_id": 24,
  "answer": "The partner responsible for the LLM component is [PPC], and the lead technical contact is Panagiotis Koumantos with the email address [p.koumantos@ppcgroup.com]. \n\nSources:\n RECITALS - LLM - Component Integration and Deployment Specification Template (3) (1).docx",
  "error_msg": null
}

```