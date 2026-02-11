# H100 Ubuntu Installation - Lessons Learned & Next Steps

**Date:** February 9, 2026  
**Updated:** February 11, 2026  
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
- **GPU:** NVIDIA H100 NVL (94 GB VRAM) — Driver 580.126.09 installed, upgrading to 580.126.16

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
- **NVIDIA Drivers:** ✅ 580.126.09 installed (upgrading to 580.126.16 in progress)
- ~~**CUDA:** ❌ Not installed~~
- **CUDA:** ⚠️ Driver reports CUDA 13.0 capability, but `nvcc` (CUDA Toolkit) NOT yet installed
- ~~**Python environment:** ❌ Not configured~~
- **Python environment:** ⚠️ Python 3.10.12 present, but `pip` not yet installed. No conda.
- **Git:** ✅ 2.34.1 installed
- **Docker:** ❌ Not installed
- **NVIDIA Container Toolkit:** ❌ Not installed
- **Isaac Sim:** ❌ Not installed

### Pending Actions
- ~~⏳ **MAC address whitelisting:** Submitted to Justin Whitten (2026-02-09)~~
- ✅ **MAC address whitelisted** — Completed
- ~~⏳ **Static IP assignment:** Waiting for IT to provide IP address~~
- ✅ **IP assigned:** `172.24.254.24`
- ⏳ **NVIDIA driver upgrade:** 580.126.09 → 580.126.16 (DKMS rebuild in progress as of Feb 11)
- ⏳ **CUDA Toolkit installation:** Pending
- ⏳ **Docker + NVIDIA Container Toolkit:** Pending
- ⏳ **pip / Miniconda installation:** Pending

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

> ⏳ **IN PROGRESS (Feb 11, 2026):** `apt update` completed. `apt upgrade` installed 2 packages.  
> `apt dist-upgrade` encountered a dpkg file conflict with `libnvidia-compute-580` overwriting  
> a file from `libnvidia-common-580`. Running `apt-get --fix-broken -o Dpkg::Options::='--force-overwrite'`  
> to resolve. DKMS kernel module rebuild in progress.

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
> (from the NVIDIA CUDA repo). Driver is being upgraded to **580.126.16** via system update.  
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

### Phase 6: Install CUDA Toolkit ~~12.9~~ 13.0

```bash
# Download CUDA keyring
~~wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/cuda-keyring_1.1-1_all.deb~~
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb
sudo dpkg -i cuda-keyring_1.1-1_all.deb

# Update and install CUDA
sudo apt update
~~sudo apt install -y cuda-toolkit-12-9~~
sudo apt install -y cuda-toolkit

# Add to PATH
~~echo 'export PATH=/usr/local/cuda-12.9/bin:$PATH' >> ~/.bashrc~~
~~echo 'export LD_LIBRARY_PATH=/usr/local/cuda-12.9/lib64:$LD_LIBRARY_PATH' >> ~/.bashrc~~
echo 'export PATH=/usr/local/cuda/bin:$PATH' >> ~/.bashrc
echo 'export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH' >> ~/.bashrc
source ~/.bashrc

# Verify installation
nvcc --version
```

### Phase 7: Install Vulkan for Headless Rendering

```bash
sudo apt install -y vulkan-tools libvulkan1 mesa-vulkan-drivers
```

**Set environment variables for headless operation:**
```bash
echo 'export DISPLAY=' >> ~/.bashrc
echo 'export OMNI_KIT_ALLOW_ROOT=1' >> ~/.bashrc
source ~/.bashrc
```

### Phase 8: Set Up Python Environment for Isaac Sim

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

### Phase 9: Configure RAID 0 for Training Data (Optional)

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

### Phase 10: Test Isaac Sim Headless Training

```bash
# Activate virtual environment
cd ~/your-repo
source .venv/bin/activate

# Run a short test training session
python scripts/train.py --headless --num_epochs 10

# Monitor GPU usage
watch -n 1 nvidia-smi
```

### Phase 11: Set Up Long-Running Training Sessions

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
- [ ] System fully updated ⏳ (NVIDIA driver upgrade in progress)
- [x] NVIDIA drivers installed ✅ (580.126.09, upgrading to .16)
- [ ] CUDA toolkit installed
- [ ] Docker installed
- [ ] NVIDIA Container Toolkit installed
- [ ] pip / Miniconda installed
- [ ] Python environment configured
- [ ] Isaac Sim dependencies installed
- [ ] Repository cloned
- [ ] RAID array configured (optional)
- [ ] Test training run completed
- [ ] Full training session initiated

---

**Last Updated:** February 11, 2026  
~~**Next Action:** Wait for MAC whitelisting confirmation from Justin Whitten~~  
**Next Action:** Complete NVIDIA driver upgrade, then install CUDA Toolkit, Docker, and NVIDIA Container Toolkit