# Installation Guide

- [Installation Guide](#installation-guide)
  - [How to install Nvidia drivers in Debian 12 Bookworm](#how-to-install-nvidia-drivers-in-debian-12-bookworm)
    - [Step 1: Identify Your Nvidia Graphics Card](#step-1-identify-your-nvidia-graphics-card)
    - [Step 2: Update Your System](#step-2-update-your-system)
    - [Step 3: Install Pre-requisites](#step-3-install-pre-requisites)
    - [Step 4: Add the Non-Free Repository](#step-4-add-the-non-free-repository)
    - [Step 5: Install Nvidia Drivers](#step-5-install-nvidia-drivers)
    - [Step 6: Reboot Your System and Turn off Secure Boot](#step-6-reboot-your-system-and-turn-off-secure-boot)
    - [Step 7: Verify the Installation](#step-7-verify-the-installation)

## How to install Nvidia drivers in Debian 12 Bookworm

Installing Nvidia drivers in Debian 12 Bookworm is easier than imagined. The process is straightforward and can be done in a few simple steps. This guide will walk you through the installation process, ensuring that you have the latest drivers for your Nvidia graphics card.

### Step 1: Identify Your Nvidia Graphics Card

Before you begin, it's important to know which Nvidia graphics card you have. You can find this information by running the following command in your terminal:

```bash
lspci -nn | egrep -i "3d|display|vga"
```

This command will list all the graphics devices on your system. Look for a line that mentions "Nvidia" and note the model number. For example, you might see something like this:

```txt
00:02.0 VGA compatible controller [0300]: Intel Corporation Raptor Lake-P [UHD Graphics] [8086:a7a8] (rev 04)
01:00.0 VGA compatible controller [0300]: NVIDIA Corporation AD107M [GeForce RTX 4050 Max-Q / Mobile] [10de:28a1] (rev a1)
```

### Step 2: Update Your System

Before installing any new software, it's always a good idea to update your system. Open a terminal and run the following commands:

```bash
sudo apt update
sudo apt upgrade
```

This will ensure that your system is up to date with the latest packages and security updates.

### Step 3: Install Pre-requisites

Before installing the Nvidia drivers, you need to install some prerequisites. Open a terminal and run the following command:

```bash
sudo apt install linux-headers-$(uname -r) build-essential
```

### Step 4: Add the Non-Free Repository

Debian 12 Bookworm has moved the Nvidia drivers to the non-free repository. To enable this repository, you need to edit your `/etc/apt/sources.list` file. Open it with your favorite text editor:

```bash
sudo nano /etc/apt/sources.list
```

Add `contrib non-free` to the end of the lines that start with `deb` and `deb-src`. For example, change:

```txt
deb http://deb.debian.org/debian/ bookworm main non-free-firmware
deb-src http://deb.debian.org/debian/ bookworm main non-free-firmware
```

to:

```txt
deb http://deb.debian.org/debian/ bookworm main non-free-firmware contrib non-free 
deb-src http://deb.debian.org/debian/ bookworm main non-free-firmware contrib non-free
```

After making the changes, save the file and exit the text editor. If you're using `nano`, you can do this by pressing `CTRL + X`, then `Y`, and finally `Enter`.

### Step 5: Install Nvidia Drivers

After adding the non-free repository, you need to update your package list again. Run the following command:

```bash
sudo apt update
```

Now, you can install the Nvidia drivers. Run the following command:

```bash
sudo apt install nvidia-driver firmware-misc-nonfree
```

It is preferred to install proprietary drivers for your graphics card. There are open-source drivers available, but they may not provide the same level of performance or compatibility as the proprietary ones. If you want to install the open-source drivers, you can run:

```bash
sudo apt install nvidia-open-kernel-dkms nvidia-driver firmware-misc-nonfree
```

### Step 6: Reboot Your System and Turn off Secure Boot

After the installation is complete, you need to reboot your system for the changes to take effect. Run the following command:

```bash
sudo reboot
```

While rebooting, make sure to turn off Secure Boot in your BIOS settings. This is important because Secure Boot can prevent the Nvidia drivers from loading properly.

### Step 7: Verify the Installation

After your system has rebooted, you can verify that the Nvidia drivers are installed and working correctly. Open a terminal and run the following command:

```bash
nvidia-smi
```

This command should display information about your Nvidia graphics card, including the driver version and GPU utilization. If you see this information, congratulations! You have successfully installed the Nvidia drivers on your Debian 12 Bookworm system.

```txt
Mon Mar 31 16:01:41 2025       
+-----------------------------------------------------------------------------------------+
| NVIDIA-SMI 570.124.06             Driver Version: 570.124.06     CUDA Version: 12.8     |
|-----------------------------------------+------------------------+----------------------+
| GPU  Name                 Persistence-M | Bus-Id          Disp.A | Volatile Uncorr. ECC |
| Fan  Temp   Perf          Pwr:Usage/Cap |           Memory-Usage | GPU-Util  Compute M. |
|                                         |                        |               MIG M. |
|=========================================+========================+======================|
|   0  NVIDIA GeForce RTX 4050 ...    On  |   00000000:01:00.0 Off |                  N/A |
| N/A   39C    P4            312W /   30W |      15MiB /   6141MiB |     21%      Default |
|                                         |                        |                  N/A |
+-----------------------------------------+------------------------+----------------------+
                                                                                         
+-----------------------------------------------------------------------------------------+
| Processes:                                                                              |
|  GPU   GI   CI              PID   Type   Process name                        GPU Memory |
|        ID   ID                                                               Usage      |
|=========================================================================================|
|    0   N/A  N/A            1667      G   /usr/lib/xorg/Xorg                        4MiB |
+-----------------------------------------------------------------------------------------+
```
