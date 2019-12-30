#!/bin/bash
set -euxo pipefail

# Configure the VM
mkdir -p ~/"VirtualBox VMs"/ub18automachine
VBoxManage createvm --name ub18automachine --ostype Ubuntu_64 --register
VBoxManage createmedium --filename ~/"VirtualBox VMs"/ub18automachine/ub18automachine.vdi --size 10240
VBoxManage storagectl ub18automachine --name SATA --add SATA --controller IntelAhci
VBoxManage storageattach ub18automachine --storagectl SATA --port 0 --device 0 --type hdd --medium ~/"VirtualBox VMs"/ub18automachine/ub18automachine.vdi
VBoxManage modifyvm ub18automachine --memory 2048 --vram 16
VBoxManage modifyvm ub18automachine --cpus 4
VBoxManage modifyvm ub18automachine --nic1 bridged --bridgeadapter1 $(ip link | grep wl | cut -f2 -d ':') --nic2 nat
VBoxManage modifyvm ub18automachine --natpf1 "guestssh,tcp,,2222,,22"

# get the ubuntu install iso
PID=$(echo $RANDOM)
transmission-cli -f dl_cleanup.sh -w $(pwd) -D ubuntu-19.10-desktop-amd64.iso.torrent &
echo 'kill '$(jobs -p | tail -1)'; rm -f "ubuntu_dl.pid"' > dl_cleanup.sh
chmod +x dl_cleanup.sh

wait $(jobs -p)
rm -f ubuntu_dl.pid dl_cleanup.sh

# this takes a bit, I think around two hours :-(
VBoxManage unattended install ub18automachine --user=mlart --password=gangen --country=CA --time-zone=EST --hostname=mlart.localhost --script-template ./ubuntu_preseed.cfg --post-install-template ./debian_postinstall.sh --iso=ubuntu-19.10-desktop-amd64.iso --start-vm=gui
