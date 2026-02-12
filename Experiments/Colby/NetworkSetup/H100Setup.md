# H100 Ubuntu Installation - Lessons Learned & Next Steps

**Date:** February 9, 2026  
**Updated:** February 12, 2026  
**System:** H100 Training Server  
**Hostname:** ai2ct2  
~~**Current OS:** Ubuntu 20.04.1 LTS (Legacy Server)~~  
**Current OS:** Ubuntu 22.04.5 LTS (Fresh Install - Feb 11, 2026)

---

## Lessons Learned

### 1. Ubuntu 24.04 "Live" ISOs Are Problematic Without Network
- **Issue:** Any Ubuntu ISO with "live" in the name uses the Subiquity installer, which is a snap package
- **Problem:** Subiquity requires network connectivity and hangs indefinitely (30+ minutes) waiting for snapd services when network is unavailable or MAC addresses aren't whitelisted
- **Masking snapd doesn't help:** The installer itself IS a snap, so masking snapd prevents the installer from running at all
- **Solution:** Use Ubuntu 20.04 Legacy Server ISO (`ubuntu-20.04.1-legacy-server-amd64.iso`) which uses the traditional Debian-style installer

### 2. CMU Network Requires MAC Whitelisting Before Any Network Activity
- **Catch-22 situation:** Can't boot cleanly without network, but can't configure network without MAC address, and can't get MAC address without booting
- **Solution:** Install completely offline (ethernet unplugged), boot to get MAC address, then send for whitelisting
- **Network shows as DOWN:** Even with cable plugged in, interfaces show `NO-CARRIER` or `DOWN` until MAC is whitelisted by IT

### 3. Installation Without Network
- **Keep ethernet unplugged** during entire installation process
- **Skip all network configuration** steps in the installer
- **Don't select package updates** during installation
- This allows installation to complete in 10-15 minutes instead of hanging for 30-60+ minutes

### 4. RAID Configuration in Legacy Installer
- The Ubuntu 20.04 legacy installer does NOT show "Configure software RAID" option by default in manual partitioning
- **Workaround:** Install OS on single drive, configure RAID for data drives post-installation
- **Best practice for training servers:** OS on one drive, RAID 0 across remaining drives for training data storage

### 5. OpenSSH Server Not Included by Default
- Even when "Install OpenSSH server" is selected during installation, it may not actually install if there's no network connectivity
- **openssh-client** is installed (for outgoing SSH connections)
- **openssh-server** is NOT installed (for incoming SSH connections)
- Must install manually after network is configured

---

## Current System Status

### Hardware
- ~~**CPU:** Unknown (not checked yet)~~
- **CPU:** Intel Xeon Platinum 8581V (120 threads)
- ~~**RAM:** 878.69 GB~~
- **RAM:** 1.0 TiB
- **Storage:** 
  - `/dev/nvme0n1` - 960.2 GB (OS installed here, ~~0.4%~~ 2% used)
  - `/dev/nvme1n1` - 15.4 TB (unused)
  - `/dev/nvme2n1` - 15.4 TB (unused)
- ~~**GPU:** NVIDIA H100 NVL (not yet configured)~~
- ~~**GPU:** NVIDIA H100 NVL (94 GB VRAM) — Driver 580.126.09 installed, upgrading to 580.126.16~~
- ~~**GPU:** NVIDIA H100 NVL (94 GB VRAM) — Driver 580.126.16 packages installed + DKMS built. **Reboot required** to load new kernel modules.~~
- **GPU:** NVIDIA H100 NVL (95830 MiB / ~94 GB VRAM) — Driver **580.126.16** verified working via `nvidia-smi` (Feb 12, 2026)

### Network Configuration
- **Interface in use:** eno1
- **MAC Address:** `3c:ec:ef:e3:14:ae`
- ~~**Status:** Interface is UP, but no network connectivity (MAC not whitelisted yet)~~
- **Status:** ✅ Network is UP and fully connected. IP assigned: `172.24.254.24`
- **Additional interface:** eno2 (MAC: `3c:ec:ef:e3:14:af`)

### Software Status
- ~~**OS:** Ubuntu 20.04.1 LTS (GNU/Linux 5.4.0-42-generic x86_64)~~
- **OS:** Ubuntu 22.04.5 LTS (GNU/Linux 5.15.0-170-generic x86_64)
- **User account:** t2user
- ~~**SSH Server:** ❌ NOT INSTALLED (openssh-server missing)~~
- **SSH Server:** ✅ Installed & running (password auth enabled, SSH key auth also configured)
- ~~**NVIDIA Drivers:** ❌ Not installed~~
- ~~**NVIDIA Drivers:** ✅ 580.126.09 installed (upgrading to 580.126.16 in progress)~~
- ~~**NVIDIA Drivers:** ✅ 580.126.16 packages installed & DKMS kernel modules built. **Needs reboot** to verify via `nvidia-smi`.~~
- **NVIDIA Drivers:** ✅ **580.126.16** — Verified working after reboot (Feb 12, 2026)
- ~~**CUDA:** ❌ Not installed~~
- ~~**CUDA:** ⚠️ Driver reports CUDA 13.0 capability, but `nvcc` (CUDA Toolkit) NOT yet installed~~
- **CUDA:** ✅ CUDA Toolkit **13.1** installed (`nvcc V13.1.115`) at `/usr/local/cuda-13.1`. PATH configured in `.bashrc`.
- ~~**Python environment:** ❌ Not configured~~
- ~~**Python environment:** ⚠️ Python 3.10.12 present, but `pip` not yet installed. No conda.~~
- ~~**Python environment:** ⚠️ Python 3.10.12 + pip 22.0.2 installed. No conda/Miniconda yet.~~
- **Python environment:** ✅ Python 3.10.12 + pip 22.0.2 + **Miniconda 25.11.1** at `/home/t2user/miniconda3`
- **Git:** ✅ 2.34.1 installed
- ~~**Docker:** ❌ Not installed~~
- **Docker:** ✅ Docker **29.2.1** installed & running. GPU passthrough verified (`--gpus all` works).
- ~~**NVIDIA Container Toolkit:** ❌ Not installed~~
- ~~**NVIDIA Container Toolkit:** ✅ Installed. Runtime configured in `/etc/docker/daemon.json`. ⚠️ Docker restart caused `dockerd` to enter D-state (uninterruptible sleep) — **server reboot required** to clear (Feb 12, 2026).~~
- **NVIDIA Container Toolkit:** ✅ Installed & working. Runtime configured in `/etc/docker/daemon.json`. D-state issue resolved by reboot.
- **Isaac Sim:** ⏳ Installing...

### Pending Actions
- ~~⏳ **MAC address whitelisting:** Submitted to Justin Whitten (2026-02-09)~~
- ✅ **MAC address whitelisted** — Completed
- ~~⏳ **Static IP assignment:** Waiting for IT to provide IP address~~
- ✅ **IP assigned:** `172.24.254.24`
- ~~⏳ **NVIDIA driver upgrade:** 580.126.09 → 580.126.16 (DKMS rebuild in progress as of Feb 11)~~
- ~~✅ **NVIDIA driver upgrade:** 580.126.16 packages installed, DKMS modules built & signed. `nvidia-persistenced` had systemd warnings (non-critical). **Server needs reboot.**~~
- ✅ **NVIDIA driver upgrade:** 580.126.16 verified working after reboot (Feb 12)
- ~~⏳ **Post-reboot verification:** Run `nvidia-smi` to confirm driver 580.126.16 loaded, then `sudo dpkg --configure -a` and `sudo apt autoremove`~~
- ✅ **Post-reboot verification:** `nvidia-smi` confirmed driver 580.126.16, H100 NVL 95830 MiB (Feb 12)
- ~~⏳ **CUDA Toolkit installation:** Pending (after reboot)~~
- ✅ **CUDA Toolkit 13.1** installed via `sudo apt install -y cuda-toolkit-13-1` (Feb 12)
- ~~⏳ **Docker + NVIDIA Container Toolkit:** Pending~~
- ✅ **Docker 29.2.1** installed (Feb 12)
- ~~✅ **NVIDIA Container Toolkit** installed & configured (Feb 12). ⚠️ Docker restart caused stuck `dockerd` process — **reboot in progress**~~
- ✅ **NVIDIA Container Toolkit** installed & working (Feb 12). D-state resolved after reboot.
- ~~⏳ **pip / Miniconda installation:** Pending~~
- ✅ **pip 22.0.2** installed + build-essential, python3-venv, curl, wget, htop, tmux, screen, net-tools (Feb 12)
- ~~⏳ **Miniconda installation:** Pending~~
- ✅ **Miniconda 25.11.1** installed at `/home/t2user/miniconda3` (Feb 12)
- ~~⏳ **GPU-in-Docker test:** Pending (after reboot clears stuck dockerd)~~
- ✅ **GPU-in-Docker test:** `nvidia-smi` runs inside `nvidia/cuda:12.8.0-base-ubuntu22.04` container — H100 visible (Feb 12)
- ~~⏳ **apt autoremove:** Pending~~
- ✅ **apt autoremove:** Completed (Feb 12)
- ⏳ **Isaac Sim / Isaac Lab installation:** In progress

---

## Next Steps (In Order)

### Phase 1: Network Configuration (Once MAC is Whitelisted)

1. **Verify network connectivity**
   ```bash
   ip link show eno1
   # Should show state UP with CARRIER
   
   ping 8.8.8.8
   # Should get responses
   ```

2. **Configure static IP** (once IT provides the IP address)
   ```bash
   sudo nano /etc/netplan/00-installer-config.yaml
   ```
   
   Add configuration:
   ```yaml
   network:
     version: 2
     ethernets:
       eno1:
         dhcp4: no
         addresses:
           - 172.24.254.XX/24  # Replace XX with assigned IP
         routes:
           - to: default
             via: 172.24.254.1  # Confirm gateway with IT
         nameservers:
           addresses:
             - 8.8.8.8
             - 1.1.1.1
   ```
   
   Apply configuration:
   ```bash
   sudo netplan apply
   ```

3. **Test connectivity**
   ```bash
   ping 8.8.8.8
   ping google.com
   ```

### Phase 2: Install OpenSSH Server

~~```bash~~
~~sudo apt update~~
~~sudo apt install openssh-server~~
~~sudo systemctl enable ssh~~
~~sudo systemctl start ssh~~
~~sudo systemctl status ssh~~
~~```~~

> ✅ **COMPLETED (Feb 11, 2026):** OpenSSH server was pre-installed with Ubuntu 22.04.  
> SSH key auth configured from Windows laptop. Password auth also enabled for team access.  
> Access via: `ssh t2user@172.24.254.24` with password `!QAZ@WSX3edc4rfv`

**Test SSH access** (from another machine on CMU network or VPN):
```bash
ssh t2user@172.24.254.XX
```

**Optional - Set up SSH key authentication** (from your laptop):
```bash
ssh-keygen -t ed25519
ssh-copy-id t2user@172.24.254.XX
```

### Phase 3: System Updates

~~```bash~~
~~sudo apt update~~
~~sudo apt upgrade -y~~
~~sudo apt dist-upgrade -y~~
~~sudo reboot~~
~~```~~

> ~~⏳ **IN PROGRESS (Feb 11, 2026):** `apt update` completed. `apt upgrade` installed 2 packages.~~  
> ~~`apt dist-upgrade` encountered a dpkg file conflict with `libnvidia-compute-580` overwriting~~  
> ~~a file from `libnvidia-common-580`. Running `apt-get --fix-broken -o Dpkg::Options::='--force-overwrite'`~~  
> ~~to resolve. DKMS kernel module rebuild in progress.~~
>
> ✅ **RESOLVED (Feb 11, 2026):** All system updates completed.  
> The dpkg conflict was resolved with `sudo apt-get install -y -o Dpkg::Options::='--force-overwrite' --fix-broken`.  
> 15 NVIDIA packages upgraded/installed (580.126.16), DKMS kernel modules built & signed for 5.15.0-170-generic.  
> `nvidia-persistenced` service threw systemd warnings ("Transport endpoint is not connected") — this is a  
> non-critical systemd/DBus issue that should clear after reboot.  
> **Next step:** `sudo reboot`, then verify with `nvidia-smi`.

### Phase 4: Install Development Tools

```bash
sudo apt install -y \
    build-essential \
    git \
    screen \
    tmux \
    htop \
    python3-pip \
    python3-venv \
    vim \
    curl \
    wget \
    net-tools
```

> ✅ **COMPLETED (Feb 12, 2026):** All dev tools installed including build-essential, python3-pip (22.0.2),
> python3-venv, curl, wget, htop, tmux, screen, net-tools. gcc 11.4.0 confirmed.

### Phase 5: Install NVIDIA Drivers

~~**Check current GPU status:**~~
~~```bash~~
~~lspci | grep -i nvidia~~
~~```~~

~~**Install NVIDIA drivers (version 575.x to match other H100 boxes):**~~
~~```bash~~
~~# Add NVIDIA driver repository~~
~~sudo add-apt-repository ppa:graphics-drivers/ppa~~
~~sudo apt update~~
~~# Install specific driver version~~
~~sudo apt install -y nvidia-driver-575~~
~~# Reboot to load driver~~
~~sudo reboot~~
~~```~~

> ✅ **UPDATE (Feb 11, 2026):** Ubuntu 22.04 fresh install came with NVIDIA driver **580.126.09** pre-installed  
> (from the NVIDIA CUDA repo). ~~Driver is being upgraded to **580.126.16** via system update.~~  
> Driver **580.126.16** packages fully installed & DKMS modules built. **Reboot required** to load.  
> No need to manually add PPA or install driver-575.

**Verify installation:**
```bash
nvidia-smi
```

Should show:
- ~~Driver Version: 575.x~~
- **Driver Version: 580.126.16**
- ~~CUDA Version: 12.9~~
- **CUDA Version: 13.0**
- GPU: NVIDIA H100 NVL

### Phase 6: Install CUDA Toolkit ~~12.9~~ ~~13.0~~ 13.1

```bash
# Download CUDA keyring
~~wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/cuda-keyring_1.1-1_all.deb~~
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb
sudo dpkg -i cuda-keyring_1.1-1_all.deb

# Update and install CUDA
sudo apt update
~~sudo apt install -y cuda-toolkit-12-9~~
~~sudo apt install -y cuda-toolkit~~
sudo apt install -y cuda-toolkit-13-1

# Add to PATH
~~echo 'export PATH=/usr/local/cuda-12.9/bin:$PATH' >> ~/.bashrc~~
~~echo 'export LD_LIBRARY_PATH=/usr/local/cuda-12.9/lib64:$LD_LIBRARY_PATH' >> ~/.bashrc~~
echo 'export PATH=/usr/local/cuda/bin:$PATH' >> ~/.bashrc
echo 'export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH' >> ~/.bashrc
source ~/.bashrc

# Verify installation
nvcc --version
```

> ✅ **COMPLETED (Feb 12, 2026):** CUDA Toolkit 13.1 installed. `nvcc V13.1.115` confirmed.
> Installed at `/usr/local/cuda-13.1` (symlinked from `/usr/local/cuda`).
> PATH and LD_LIBRARY_PATH added to `~/.bashrc`.

### Phase 7: Install Docker & NVIDIA Container Toolkit

```bash
# Install Docker
sudo apt-get install -y docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin

# Install NVIDIA Container Toolkit
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey \
  | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list \
  | sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' \
  | sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
sudo apt-get update
sudo apt-get install -y nvidia-container-toolkit

# Configure Docker runtime for NVIDIA
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker

# Test GPU access in Docker
sudo docker run --rm --gpus all nvidia/cuda:13.0.1-base-ubuntu22.04 nvidia-smi
```

> ✅ **COMPLETED (Feb 12, 2026):** Docker 29.2.1 installed. NVIDIA Container Toolkit installed.
> Runtime configured in `/etc/docker/daemon.json` (nvidia runtime added).
> ⚠️ `dockerd` entered D-state (uninterruptible sleep) during `systemctl restart docker` —
> this is a kernel-level stuck process that cannot be killed with SIGKILL.
> **Server reboot required** to clear the stuck process. Docker will start clean after reboot.
> GPU-in-Docker test pending post-reboot.

### Phase 8: Install Vulkan for Headless Rendering

```bash
sudo apt install -y vulkan-tools libvulkan1 mesa-vulkan-drivers
```

**Set environment variables for headless operation:**
```bash
echo 'export DISPLAY=' >> ~/.bashrc
echo 'export OMNI_KIT_ALLOW_ROOT=1' >> ~/.bashrc
source ~/.bashrc
```

### Phase 9: Set Up Python Environment for Isaac Sim

```bash
# Clone your team's repository
cd ~
git clone <your-capstone-repo-url>
cd <repo-name>

# Create Python virtual environment (matching your laptop setup)
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies from your requirements.txt
pip install --upgrade pip
pip install -r requirements.txt
```

### Phase 10: Configure RAID 0 for Training Data (Optional)

**Create RAID 0 array with the two 15.4TB NVMe drives:**
```bash
# Install mdadm
sudo apt install -y mdadm

# Create RAID 0 array
sudo mdadm --create --verbose /dev/md0 \
    --level=0 \
    --raid-devices=2 \
    /dev/nvme1n1 \
    /dev/nvme2n1

# Create filesystem
sudo mkfs.ext4 -F /dev/md0

# Create mount point
sudo mkdir -p /mnt/training_data

# Mount the array
sudo mount /dev/md0 /mnt/training_data

# Make it permanent
echo '/dev/md0 /mnt/training_data ext4 defaults,nofail 0 0' | sudo tee -a /etc/fstab

# Save RAID configuration
sudo mdadm --detail --scan | sudo tee -a /etc/mdadm/mdadm.conf
sudo update-initramfs -u
```

**Verify RAID:**
```bash
cat /proc/mdstat
df -h /mnt/training_data
```

Should show ~30TB available space.

### Phase 11: Test Isaac Sim Headless Training

```bash
# Activate virtual environment
cd ~/your-repo
source .venv/bin/activate

# Run a short test training session
python scripts/train.py --headless --num_epochs 10

# Monitor GPU usage
watch -n 1 nvidia-smi
```

### Phase 12: Set Up Long-Running Training Sessions

**Install screen for persistent sessions:**
```bash
sudo apt install -y screen
```

**Start a training session:**
```bash
# Create new screen session
screen -S quadruped_training

# Activate environment
cd ~/your-repo
source .venv/bin/activate

# Run training
python scripts/train.py --headless --num_epochs 5000

# Detach from screen: Ctrl+A, then D
# Reattach later: screen -r quadruped_training
```

**Monitor training from home (via SSH):**
```bash
# From home (on CMU VPN)
ssh t2user@172.24.254.XX

# Check running screens
screen -ls

# Reattach to training session
screen -r quadruped_training

# Check GPU usage
nvidia-smi

# Check training logs
tail -f training.log  # if you're logging to file
```

---

## Important Notes

### Security Considerations
- This server will be accessible from CMU network and via VPN
- Use strong passwords
- Consider SSH key-based authentication only (disable password auth)
- Keep system updated with security patches

### Monitoring and Maintenance
- Set up automated backups of training checkpoints
- Monitor disk space on RAID array (`df -h`)
- Check NVIDIA driver/CUDA compatibility before major updates
- Version control all training code and configs

### Network Access
- **On campus:** Direct access via ~~172.24.254.XX~~ `172.24.254.24`
- **From home:** Must be on CMU VPN first, then SSH
- **Switch location:** 172.24.254.23 (dedicated switch with available ports)
- **SSH command:** `ssh t2user@172.24.254.24`
- **Password:** `!QAZ@WSX3edc4rfv`

### Training Data Storage
- **OS Drive:** 960GB (keep this for OS and software only)
- **RAID Array:** ~30TB (use for training data, checkpoints, logs)
- Consider organizing as:
  - `/mnt/training_data/datasets/`
  - `/mnt/training_data/checkpoints/`
  - `/mnt/training_data/logs/`
  - `/mnt/training_data/results/`

### Backup Strategy
- Training code: Version controlled in Git
- Checkpoints: Periodic copies to team storage
- Important results: Export and archive
- The RAID 0 array has NO redundancy - one drive failure = total data loss

---

## Contact Information

**IT Support:**
- Justin Whitten (CW3 USARMY AFC AI2C)
- Switch IP: 172.24.254.23

**Team Members:**
- User account: t2user
- Project: Autonomous Navigation Capstone (Isaac Sim RL training)

---

## Troubleshooting Reference

### If network doesn't work after whitelisting:
```bash
# Check interface status
ip link show eno1

# Bring interface up
sudo ip link set eno1 up

# Check for DHCP lease (if testing)
sudo dhclient eno1

# Check routing
ip route

# Test DNS
nslookup google.com
```

### If SSH won't start:
```bash
# Check if installed
dpkg -l | grep openssh-server

# Check service status
sudo systemctl status ssh

# Check if port 22 is listening
sudo ss -tlnp | grep :22

# Check firewall (if enabled)
sudo ufw status
```

### If NVIDIA drivers don't load:
```bash
# Check if driver is loaded
lsmod | grep nvidia

# Check for errors
dmesg | grep -i nvidia

# Reinstall driver
~~sudo apt install --reinstall nvidia-driver-575~~
sudo apt install --reinstall nvidia-driver-580
sudo reboot
```

### If Isaac Sim headless mode fails:
```bash
# Check Vulkan
vulkaninfo

# Verify DISPLAY is unset
echo $DISPLAY  # Should be empty

# Check GPU is accessible
nvidia-smi

# Try with explicit flags
python script.py --headless --enable-extension omni.isaac.sim --/app/window/enabled=false
```

---

## Status Checklist

Current progress:

- [x] ~~Ubuntu 20.04.1 LTS installed~~
- [x] **Ubuntu 22.04.5 LTS installed (fresh install Feb 11, 2026)**
- [x] System boots successfully
- [x] MAC address identified and submitted for whitelisting
- [x] MAC address whitelisted by IT ✅
- [x] ~~Static IP assigned and configured~~ IP: `172.24.254.24` ✅
- [x] Network connectivity verified ✅
- [x] OpenSSH server installed ✅ (pre-installed with 22.04)
- [x] SSH access working remotely ✅ (password + key auth)
- [x] ~~System fully updated ⏳ (NVIDIA driver upgrade in progress)~~
- [x] System updates applied ✅ (all packages installed, ~~**reboot needed**~~ **rebooted & verified Feb 12**)
- [x] ~~NVIDIA drivers installed ✅ (580.126.09, upgrading to .16)~~
- [x] ~~NVIDIA drivers installed ✅ (580.126.16 packages + DKMS, **reboot to verify**)~~
- [x] NVIDIA drivers ✅ **580.126.16** verified working (Feb 12, 2026)
- [x] ~~CUDA toolkit installed~~
- [x] CUDA Toolkit **13.1** installed ✅ (`nvcc V13.1.115`)
- [x] ~~Docker installed~~
- [x] Docker **29.2.1** installed ✅
- [x] ~~NVIDIA Container Toolkit installed~~
- [x] NVIDIA Container Toolkit installed & configured ✅ (⚠️ reboot needed to clear stuck dockerd)
- [x] ~~pip / Miniconda installed~~
- [x] pip **22.0.2** + build-essential + dev tools installed ✅
- [x] ~~Miniconda installed~~
- [x] Miniconda **25.11.1** installed ✅ (`/home/t2user/miniconda3`)
- [x] ~~GPU-in-Docker test (`docker run --gpus all ... nvidia-smi`)~~
- [x] GPU-in-Docker test ✅ — H100 visible inside container
- [ ] Python environment configured (`conda create -n env_isaaclab python=3.11`)
- [ ] Isaac Sim dependencies installed
- [ ] Repository cloned
- [ ] RAID array configured (optional)
- [ ] Test training run completed
- [ ] Full training session initiated

---

**Last Updated:** February 12, 2026 (evening)  
~~**Next Action:** Wait for MAC whitelisting confirmation from Justin Whitten~~  
~~**Next Action:** Complete NVIDIA driver upgrade, then install CUDA Toolkit, Docker, and NVIDIA Container Toolkit~~  
**Next Action (for Colby):**
1. ~~`sudo reboot` the server~~ ✅
2. ~~Verify `nvidia-smi` shows driver **580.126.16** and GPU detected~~ ✅ Verified Feb 12
3. ~~Run `sudo dpkg --configure -a` to finalize any pending package configs~~ ✅
4. ~~Run `sudo apt autoremove` to clean up unused packages~~ ✅ Completed Feb 12
5. ~~Install CUDA Toolkit (see Phase 6)~~ ✅ CUDA 13.1 installed Feb 12
6. ~~Install pip: `sudo apt install -y python3-pip`~~ ✅ pip 22.0.2 installed Feb 12
7. ~~Install Docker + NVIDIA Container Toolkit~~ ✅ Docker 29.2.1 + NVIDIA CTK installed Feb 12
8. ~~Verify Docker after reboot~~ ✅ Docker active, GPU-in-Docker verified Feb 12
9. ~~Install Miniconda~~ ✅ Miniconda 25.11.1 installed Feb 12
10. **Install Isaac Sim / Isaac Lab** ⏳ In progress...

**Isaac Sim Installation Steps (current):**
```bash
# 1. Create conda environment with Python 3.11 (required by Isaac Sim 5.1.0)
source ~/miniconda3/etc/profile.d/conda.sh
conda create -n env_isaaclab python=3.11 -y
conda activate env_isaaclab

# 2. Install Isaac Sim
pip install "isaacsim[all,extscache]==5.1.0" --extra-index-url https://pypi.nvidia.com

# 3. Install PyTorch with CUDA support
pip install -U torch==2.7.0 torchvision==0.22.0 --index-url https://download.pytorch.org/whl/cu128

# 4. Clone and install Isaac Lab
git clone git@github.com:isaac-sim/IsaacLab.git
cd IsaacLab
./isaaclab.sh --install

# 5. Verify
isaacsim
```

---

## Known Issues & Workarounds

### Boot Fixes Applied (Feb 12, 2026)
- **`systemd-networkd-wait-online.service`** was blocking login prompt for 2+ minutes. Fixed:
  ```bash
  sudo systemctl disable systemd-networkd-wait-online.service
  sudo systemctl mask systemd-networkd-wait-online.service
  ```
- **Swap partition** referenced in `/etc/fstab` didn't exist, causing boot hang. Fixed:
  ```bash
  # Commented out the swap line in /etc/fstab
  ```

### Docker D-State Issue (Feb 12, 2026) — RESOLVED
- After running `sudo nvidia-ctk runtime configure --runtime=docker` and `sudo systemctl restart docker`,
  the `dockerd` process entered D-state (uninterruptible sleep). This is a kernel-level state that cannot
  be cleared even with `kill -9`. Only a full server reboot will resolve it.
- **Resolved:** Server rebooted, Docker started cleanly, GPU-in-Docker verified working.
  `sudo docker run --rm --gpus all nvidia/cuda:12.8.0-base-ubuntu22.04 nvidia-smi` shows H100 inside container.

### SSH Stability (Feb 12, 2026)
- **CRITICAL: Only ONE SSH session at a time.** Running multiple parallel SSH sessions (or SSH + paramiko)
  has caused the server to become completely unresponsive, requiring physical hard reboots.
- Use `ssh t2user@172.24.254.24` with single commands, or use the paramiko-based `h100_run.py` script
  from the Capstone Project directory for reliable single-session execution.