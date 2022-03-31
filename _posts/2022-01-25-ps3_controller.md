---
header:
    teaser: "/assets/images/notebooks/ps3_controller/ps3_controller.jpg"
title: "PS3 Controller RPI Connection"
classes: wide
author_profile: true
categories:
    - notebooks
tags: 
    - banrc
    - linux
    - bluetooth
    - ps3
    
---

## Linux Bluetooth Stack

Read File: **banrc01/ext-doc/bluetooth/Bluetooth_Stack_Linux_OS.pdf**

https://www.opensourceforu.com/2015/06/linux-without-wires-the-basics-of-bluetooth/ \
https://naehrdine.blogspot.com/2021/03/bluez-linux-bluetooth-stack-overview.html

### Particularities of PS3 Bluetooth

https://pythonhosted.org/triangula/sixaxis.html

Configuring Playstation 3 Controllers

If you’re building a robot you will at some point probably want a way to manually drive it around. The Playstation3 controller, also known as the SixAxis, makes for a great option - it connects over bluetooth, has a bundle of different buttons, sticks and motion sensors, and is readily available. You’ll probably google for how to make it work with the Raspberry Pi and then, if your experience is anything like mine, you’ll find that every single online guide is a) a copy of one original document and b) doesn’t work. Having solved all these problems, I thought I’d be nice and write the method down here in the hope that no-one else has to waste the time I’ve just spent on it...
A note on pairing

One of the reasons the SixAxis isn’t as easy as it could be to use is how pairing works. Normal bluetooth devices will establish a link between the device and the host once, then the host can initiate connection using this previously stored information. In the case of the SixAxis, it’s actually the controller that initiates the process, so we have to do some setup beforehand. We need to tell the controller to which bluetooth host it should attempt to connect, and we need to tell the host (the Pi) that it should allow the controller’s connection.
Hardware

This guide assumes you’re using a Raspberry Pi (I’m using a Pi 2, but there’s no reason this wouldn’t work with older ones). You’ll also need a USB bluetooth dongle and, obviously, a SixAxis controller. I’ve only tried this with genuine Sony ones, many of the cheaper ones you’ll find online are clones, they should work but YMMV.
Bluetooth dongles

Some people are finding this guide does not work. I suspect this is down to the bluetooth dongle, having eliminated everything else in the process. The one I’m using is an Asus USB-BT400, it’s tiny and supports all the current Bluetooth standards. If you get this to work with a different dongle can you let me know on twitter at @approx_eng_ and I’ll add it to this list:

> Asus USB-BT400

Software

Note 1 - this assumes you’ve set up git and installed a public key with github, you don’t have to do this but you’ll need to modify some of the git commands below if you haven’t. You can set up public keys using the instructions at https://help.github.com/articles/generating-ssh-keys/#platform-all

Note 2 - this is also assuming you’re starting from a clean installation of the latest Jessie based Raspbian. Other distributions may need varying combinations of dev libraries etc. For testing I was using the minimal installation with filename 2015-11-21-raspbian-jessie-lite.zip but these instructions should apply to any recent version. As always, it’s not a bad idea to run sudo apt-get update and sudo apt-get upgrade to get any changes to packages since your distribution was built.

You’ll need to install some packages on your Pi first, and enable the bluetooth services:
```
pi@raspberrypi ~ $ sudo apt-get install bluetooth libbluetooth3 libusb-dev
pi@raspberrypi ~ $ sudo systemctl enable bluetooth.service
```
You also need to add the default user to the bluetooth group:

```
pi@raspberrypi ~ $ sudo usermod -G bluetooth -a pi
```

You must now power cycle your Pi. Do not just reboot, actually shut down, pull the power, wait a few seconds and reconnect. This may be overkill, but it’s been the best way I’ve found to consistently have the next steps succeed.
Pairing

Get and build the command line pairing tool:

```
pi@raspberrypi ~ $ wget http://www.pabr.org/sixlinux/sixpair.c
pi@raspberrypi ~ $ gcc -o sixpair sixpair.c -lusb
```

Firstly we need to tell the controller the address of the bluetooth dongle. To do this you need to connect the controller to your Pi with a mini-USB cable. Also make sure your Pi is powered from an external supply - the extra power needed when you connect the controllers can be too much for a laptop USB socket and you’ll get random errors or the process won’t work at all. The ‘sixpair’ command, run as root, updates the controller’s bluetooth master address:

```
pi@raspberrypi ~ $ sudo ./sixpair
```

Current Bluetooth master: 5c:f3:70:66:5c:e2
Setting master bd_addr to 5c:f3:70:66:5c:e2

You should see a message indicating that the bluetooth master address on the controller has been changed (you can specify the address to which it should change, the default with no arguments is to use the first installed bluetooth adapter, which is what you want unless for some reason you’ve got more than one plugged in). The controller will now attempt to connect to your bluetooth dongle when you press the PS button (don’t do this just yet, it won’t work). The example above shows that no change has been made, as this particular controller had been paired with the dongle before, but you should see two different addresses - the first is the address the controller was trusting, the second is the one it now trusts.

Next we need to configure the bluetooth software on the Pi to accept connections from the controller.

Disconnect your controller from the USB port, and run the ‘bluetoothctl’ command as a regular user (you don’t need to be root for this):

```
pi@raspberrypi ~ $ bluetoothctl
[NEW] Controller 5C:F3:70:66:5C:E2 raspberrypi [default]
... (other messages may appear here if you have other bluetooth hardware)
```

Now re-connect your controller with the mini-USB cable. You should see messages in the terminal indicating that something has connected (but don’t worry if you don’t, as long as something useful appears in the next step!)

Type ‘devices’ in the terminal. You will see a list of possible devices, including at least your SixAxis controller. You need to take note of the MAC address of the controller for the next step:

```
[bluetooth]# devices
Device 60:38:0E:CC:OC:E3 PLAYSTATION(R)3 Controller
... (other devices may appear here)
```

Type ‘agent on’ and then ‘trust MAC’, replacing MAC with the MAC address you noted in the previous step (they won’t be the same as mine!). Quit the tool once you’re done.

```
[bluetooth]# agent on
Agent registered
[bluetooth]# trust 60:38:0E:CC:0C:E3
[CHG] Device 60:38:0E:CC:0C:E3 Trusted: yes
Changing 60:38:0E:CC:0C:E3 trust succeeded
[bluetooth]# quit
Agent unregistered
[DEL] Controller 5C:F3:70:66:5C:E2

Disconnect your controller, you should now be able to connect wirelessly. To check this, first list everything in /dev/input:

pi@raspberrypi ~ $ ls /dev/input
by-id  by-path  event0  event1  event2  event3  event5  mice  mouse0
```

Now press the PS button, the lights on the front of the controller should flash for a couple of seconds then stop, leaving a single light on. If you now look again at the contents of /dev/input you should see a new device, probably called something like ‘js0’:

```
pi@raspberrypi ~ $ ls /dev/input
by-id    event0  event2  event4  js0   mouse0
by-path  event1  event3  event5  mice
```

If a new device has appeared here then congratulations, you have successfully paired your dongle and SixAxis controller. This will persist across reboots, so from now on you can just connect by pressing the PS button on the controller. Pressing and holding this button will shut the controller down - at the moment there’s no timeout so be sure to turn the controller off when you’re not going to be using it for a while.
Accessing the SixAxis from Python

You now have a joystick device in /dev/input, but how do you use it in your Python code?

There are two different approaches I’ve tried. You can use PyGame - this has the advantage that you might be using it already (in which case it’s the simplest solution) and it’s already installed in the system Python on your Pi. It has the drawback though that it requires a display - while I’m aware there are workarounds for this they’re not really very satisfactory. The second option is to use the Python bindings for evdev - this is lightweight, but has drawback of being more complex to use and only working on linux, even if you’re on a unix-like system such as OSX you can’t use it whereas PyGame is generally suitable for cross-platform use. Because I only want to run this on the Pi and because I really need it to work cleanly in a headless environment I’ve gone with evdev, but there are arguments for both.

Actually using evdev isn’t trivial, the best documentation I have is the code I wrote to handle it. I’ve created a Python class triangula.input.SixAxis and corresponding resource triangula.input.SixAxisResource to make this simpler to work with. The class uses asyncore to poll the evdev device, updating internal state within the object. It also allows you to register button handlers which will be called, handles centering, hot zones (regions in the axis range which clamp to 1.0 or -1.0) and dead zones (regions near the centre point which clamp to 0.0).

By way of an example, the following code will connect to the controller (you’ll get an exception if you don’t have one connected) and print out the values of the two analogue sticks:

```
from triangula.input import SixAxis, SixAxisResource

# Button handler, will be bound to the square button later
def handler(button):
  print 'Button {} pressed'.format(button)

# Get a joystick, this will fail unless the SixAxis controller is paired and active
# The bind_defaults argument specifies that we should bind actions to the SELECT and START buttons to
# centre the controller and reset the calibration respectively.
with SixAxisResource(bind_defaults=True) as joystick:
    # Register a button handler for the square button
    joystick.register_button_handler(handler, SixAxis.BUTTON_SQUARE)
    while 1:
        # Read the x and y axes of the left hand stick, the right hand stick has axes 2 and 3
        x = joystick.axes[0].corrected_value()
        y = joystick.axes[1].corrected_value()
        print(x,y)
```

You’re welcome to pick up Triangula’s libraries, they’re uploaded to PyPi semi-regularly (get with ‘pip install triangula’) or from github. In either case you’ll need to install one extra package first, without which the evdev module won’t build:

```
pi@raspberrypi ~ $ sudo apt-get install libpython2.7-dev
```

Now you can get Triangula’s code from github and build it to acquire the triangula.input module, you can then use this in your own code (there’s nothing particularly specific to Triangula in it)

```
pi@raspberrypi ~ $ git clone git@github.com:basebot/triangula.git
pi@raspberrypi ~ $ cd triangula/src/python
pi@raspberrypi ~/triangula/src/python python setup.py develop
```

This will set up the libraries in develop mode, creating symbolic links into your python installation (I’m assuming here that you’re using a virtual environment, because you should be - if you’re not you’ll need to run some of these commands as root)


## Configuring PS3 on RaspberryPi


```python
# Kernel modules
!lsmod  | grep blue
```

    bluetooth             393216  24 hci_uart,bnep,btbcm
    ecdh_generic           16384  2 bluetooth
    rfkill                 32768  6 bluetooth,cfg80211


### Bluetooth FrontEnds

Exists 2 main console frontends applications to manage bluetooth stack.
- **hcitool**: More scriptable. Deprecated? Commication via HW?
- **bluetoothclt**: New BlueZ frontend. Interactive. Commication via D-BUS


```
Because bluetoothctl talks to the Bluetooth daemon via DBus, instead of directly with the hardware like hcitool does. And hcitool usually fails now because the BT daemon has exclusive access.
```

https://www.linuxquestions.org/questions/programming-9/control-bluetoothctl-with-scripting-4175615328/
```
echo Restarting bluetooth service.
sudo service bluetooth restart

coproc bluetoothctl
echo -e 'agent on\nconnect AE:2D:22:00:35:A2\nexit' >&${COPROC[1]}
output=$(cat <&${COPROC[0]})
echo $output
```

### Needed add user to bluetooth group
```
sudo usermod -a -G bluetooth banshee
reboot
```

### Device Power UP


```python
!rfkill list all
```

    0: phy0: Wireless LAN
    	Soft blocked: no
    	Hard blocked: no
    1: hci0: Bluetooth
    	Soft blocked: no
    	Hard blocked: no


**TODO**: UP/DOWN device

### Bluetooth Device Info 

##### Several methods


```python
!hcitool dev
```

    Devices:
    	hci0	B8:27:EB:C9:BF:B7



```python
!hciconfig
```

    hci0:	Type: Primary  Bus: UART
    	BD Address: B8:27:EB:C9:BF:B7  ACL MTU: 1021:8  SCO MTU: 64:1
    	UP RUNNING 
    	RX bytes:1562 acl:0 sco:0 events:96 errors:0
    	TX bytes:2574 acl:0 sco:0 commands:96 errors:0
    



```python
!echo 'list' | bluetoothctl
```

    Agent registered
    [0;94m[bluetooth][0m# list
    Controller B8:27:EB:C9:BF:B7 banrc01 [default]
    [K;94m[bluetooth][0m# 

## SixPair

https://github.com/lakkatv/sixpair


```python
!mkdir -p sixpair
!cp ../../ext-libs/sixpair/sixpair.c sixpair/
!file sixpair/sixpair.c

!gcc -o sixpair/sixpair sixpair/sixpair.c -lusb
!file sixpair/sixpair
```

    sixpair/sixpair.c: C source, ASCII text
    sixpair/sixpair: ELF 32-bit LSB executable, ARM, EABI5 version 1 (SYSV), dynamically linked, interpreter /lib/ld-linux-armhf.so.3, for GNU/Linux 3.2.0, BuildID[sha1]=cefbafaf1819182b960bb784f2fc2e34568ef99f, not stripped


```
sudo ./sixpair 
Current Bluetooth master: eb:37:ef:55:e2:48
Setting master bd_addr to b8:27:eb:c9:bf:b7
```

## **Problems** Thrusting Device

```banshee@banrc01:~ $ sudo bluetoothctl 
[sudo] password for banshee: 
Agent registered
[NEW] Device 0C:FC:83:F5:92:73 Sony PLAYSTATION(R)3 Controller
[DEL] Device 0C:FC:83:F5:92:73 Sony PLAYSTATION(R)3 Controller
[bluetooth]# trust 0C:FC:83:F5:92:73
Device 0C:FC:83:F5:92:73 not available
```

Turn bluetoothd to verbose mode:


Edit **/etc/systemd/system/bluetooth.target.wants/bluetooth.service**

```
# Add -d flag
ExecStart=/usr/lib/bluetooth/bluetoothd -d
```

Restart bluetooth daemon
```
sudo systemctl restart bluetooth.service
```

Now inspecting error on **/var/log/syslog** why device is deleted after creating

```
Dec 13 19:28:05 banrc01 bluetoothd[1150]: sixaxis: compatible device connected: Sony PLAYSTATION(R)3 Controller (054C:0268 /sys/devices/platform/soc/3f980000.usb/usb1/1-1/1-1.1/1-1.1.2/1-1.1.2:1.0/0003:054C:0268.0007/hidraw/hidraw0)
Dec 13 19:28:05 banrc01 bluetoothd[1150]: src/device.c:device_create() dst 0C:FC:83:F5:92:73
Dec 13 19:28:05 banrc01 bluetoothd[1150]: src/device.c:device_new() address 0C:FC:83:F5:92:73
Dec 13 19:28:05 banrc01 bluetoothd[1150]: src/device.c:device_new() Creating device /org/bluez/hci0/dev_0C_FC_83_F5_92_73
Dec 13 19:28:05 banrc01 bluetoothd[1150]: sixaxis: setting up new device
Dec 13 19:28:05 banrc01 bluetoothd[1150]: src/device.c:btd_device_device_set_name() /org/bluez/hci0/dev_0C_FC_83_F5_92_73 Sony PLAYSTATION(R)3 Controller
Dec 13 19:28:05 banrc01 bluetoothd[1150]: Authentication attempt without agent
Dec 13 19:28:05 banrc01 bluetoothd[1150]: plugins/sixaxis.c:agent_auth_cb() Agent replied negatively, removing temporary device
Dec 13 19:28:05 banrc01 bluetoothd[1150]: src/device.c:device_remove() Removing device /org/bluez/hci0/dev_0C_FC_83_F5_92_73
Dec 13 19:28:05 banrc01 bluetoothd[1150]: src/device.c:btd_device_unref() Freeing device /org/bluez/hci0/dev_0C_FC_83_F5_92_73
Dec 13 19:28:05 banrc01 bluetoothd[1150]: src/device.c:device_free() 0x486ea8
```

On detail:
```
plugins/sixaxis.c:agent_auth_cb() Agent replied negatively, removing temporary device
```


https://github.com/RetroPie/RetroPie-Setup/pull/2263/commits/017f00f6e15f04b3272ff1abae8742dc4c47b608#diff-293af5b19ba1d1f50c86e8b9b0b1feaf

https://github.com/RetroPie/RetroPie-Setup/blob/master/scriptmodules/supplementary/customhidsony/0001-hidsony-gasiafix.diff

Alternative to manually patch module hid-sony.
Using RetroPie SW only install customhidsont module

```
Install RetroPie

Install the needed packages for the RetroPie setup script:

sudo apt install git lsb-release

Download the latest RetroPie setup script with

cd
git clone --depth=1 https://github.com/RetroPie/RetroPie-Setup.git

The script is executed with

cd RetroPie-Setup
chmod +x retropie_setup.sh
sudo ./retropie_setup.sh
```


After installing several driver packages from retropie script (sixaxis, pse) the problem **persist**.

## **Problem SOLVED** Thrusting Device using RetroPie utils

After executing RetroPie Script options:

```
configuration-tools -> sixaxis -> enable
configuration-tools -> bluetooth -> pair and connect

>>Follow the instructions
```

Now bluethoohctl shows the device **successfully**.

Unplug USB and pressing button automatically connects to gamepad


```
bluetoothctl 
Agent registered
[CHG] Device 0C:FC:83:F5:92:73 Connected: yes
[Sony PLAYSTATION(R)3 Controller]# exit
```

In this situation the gamepad not sync succesfully. \
But **not opening** bluetoothctl, the gamepad **sync** and one led is showed.


**/var/log/syslog**:
```
Dec 13 20:55:09 banrc01 bluetoothd[600]: src/adapter.c:connected_callback() hci0 device 0C:FC:83:F5:92:73 connected eir_len 5
Dec 13 20:55:14 banrc01 bluetoothd[600]: profiles/input/server.c:connect_event_cb() Incoming connection from 0C:FC:83:F5:92:73 on PSM 17
Dec 13 20:55:14 banrc01 bluetoothd[600]: profiles/input/device.c:input_device_set_channel() idev 0x1998718 psm 17
Dec 13 20:55:14 banrc01 bluetoothd[600]: profiles/input/server.c:confirm_event_cb() 
Dec 13 20:55:14 banrc01 bluetoothd[600]: profiles/input/server.c:connect_event_cb() Incoming connection from 0C:FC:83:F5:92:73 on PSM 19
Dec 13 20:55:14 banrc01 bluetoothd[600]: profiles/input/device.c:input_device_set_channel() idev 0x1998718 psm 19
Dec 13 20:55:14 banrc01 kernel: [  599.485742] Bluetooth: HIDP (Human Interface Emulation) ver 1.2
Dec 13 20:55:14 banrc01 kernel: [  599.485783] Bluetooth: HIDP socket layer initialized
Dec 13 20:55:14 banrc01 bluetoothd[600]: src/service.c:change_state() 0x1995808: device 0C:FC:83:F5:92:73 profile input-hid state changed: disconnected -> connected (0)
Dec 13 20:55:14 banrc01 bluetoothd[600]: src/service.c:btd_service_ref() 0x1995808: ref=3
Dec 13 20:55:14 banrc01 bluetoothd[600]: plugins/policy.c:service_cb() Added input-hid reconnect 0
Dec 13 20:55:14 banrc01 kernel: [  599.492306] sony 0005:054C:0268.0002: unknown main item tag 0x0
Dec 13 20:55:14 banrc01 kernel: [  599.515744] input: Sony PLAYSTATION(R)3 Controller Motion Sensors as /devices/platform/soc/3f201000.serial/tty/ttyAMA0/hci0/hci0:11/0005:054C:0268.0002/input/input3
Dec 13 20:55:14 banrc01 kernel: [  599.517109] input: Sony PLAYSTATION(R)3 Controller as /devices/platform/soc/3f201000.serial/tty/ttyAMA0/hci0/hci0:11/0005:054C:0268.0002/input/input2
Dec 13 20:55:14 banrc01 kernel: [  599.517896] sony 0005:054C:0268.0002: input,hidraw0: BLUETOOTH HID v80.00 Joystick [Sony PLAYSTATION(R)3 Controller] on b8:27:eb:c9:bf:b7
Dec 13 20:55:14 banrc01 systemd-udevd[1151]: Process '/usr/bin/jscal-restore /dev/input/js0' failed with exit code 1.
Dec 13 20:55:14 banrc01 systemd[1]: Started sixaxis helper (sys/devices/platform/soc/3f201000.serial/tty/ttyAMA0/hci0/hci0:11/0005:054C:0268.0002/input/input2).
Dec 13 20:55:14 banrc01 systemd[1]: Invalid unit name "sixaxis@/dev/input/js0.service" was escaped as "sixaxis@-dev-input-js0.service" (maybe you should use systemd-escape?)
Dec 13 20:55:14 banrc01 systemd[1]: sixaxis@sys-devices-platform-soc-3f201000.serial-tty-ttyAMA0-hci0-hci0:11-0005:054C:0268.0002-input-input2.service: Succeeded.
Dec 13 20:55:14 banrc01 systemd[1]: Started sixaxis helper (/dev/input/js0).
Dec 13 20:55:14 banrc01 systemd[1]: sixaxis@-dev-input-js0.service: Succeeded.
Dec 13 20:55:14 banrc01 systemd[1]: Invalid unit name "sixaxis@/dev/input/event1.service" was escaped as "sixaxis@-dev-input-event1.service" (maybe you should use systemd-escape?)
Dec 13 20:55:14 banrc01 systemd[1]: Started sixaxis helper (/dev/input/event1).
Dec 13 20:55:14 banrc01 sixaxis-helper.sh[1158]: Calibrating: Sony PLAYSTATION(R)3 Controller (0C:FC:83:F5:92:73)
Dec 13 20:55:14 banrc01 sixaxis-helper.sh[1158]: Setting 600 second timeout on: Sony PLAYSTATION(R)3 Controller (0C:FC:83:F5:92:73)
```

Connecting gamepad with USB shows on syslog the original problem with bluetooth. It appears it's complex to trust with it. Revise internal code of RetroPie to understand.
In any case, USB plugged works well, as USB device and recognices as joystick.

## Test PS3 gamepad Bluetooth connected


```python
!ls /dev/input/
```

    event1	js0  mice


```
jstest /dev/input/js0 
Driver version is 2.1.0.
Joystick (Sony PLAYSTATION(R)3 Controller) has 6 axes (X, Y, Z, Rx, Ry, Rz)
and 17 buttons (BtnA, BtnB, BtnX, BtnY, BtnTL, BtnTR, BtnTL2, BtnTR2, BtnSelect, BtnStart, BtnMode, BtnThumbL, BtnThumbR, (null), (null), (null), (null)).
Testing ... (interrupt to exit)
Axes:  0:     0  1:     0  2:-32767  3:     0  4:     0  5:-32767 Buttons:  0:off  1:off  2:off  3:off  4:off  5:off  6:off  7:off  8:off  9:off 10:off 11:off 12:off 13:off 14:off 15:off 16:off
```

### Calibration

```
banshee@banrc01:~ $ jscal -c /dev/input/js0 
Joystick has 6 axes and 17 buttons.
Correction for axis 0 is none (raw), precision is 0.
Correction for axis 1 is none (raw), precision is 0.
Correction for axis 2 is none (raw), precision is 0.
Correction for axis 3 is none (raw), precision is 0.
Correction for axis 4 is none (raw), precision is 0.
Correction for axis 5 is none (raw), precision is 0.

Calibrating precision: wait and don't touch the joystick.
Done. Precision is:                                               128,  128 Axis 4:  128,  128 Axis 5:    8,    8 
Axis: 0:     0
Axis: 1:     0
Axis: 2:     0
Axis: 3:     0
Axis: 4:     0
Axis: 5:     0

Move axis 0 to minimum position and push any button.
Hold ... OK.                                                                  
Move axis 0 to center position and push any button.
Hold ... OK.                                                                  
Move axis 0 to maximum position and push any button.
Hold ... OK.                                                                  
Move axis 1 to minimum position and push any button.
Hold ... OK.                                                                  
Move axis 1 to center position and push any button.
Hold ... OK.                                                                  
Move axis 1 to maximum position and push any button.
Hold ... OK.                                                                  
Move axis 2 to minimum position and push any button.
Hold ... OK.                                                                  
Move axis 2 to center position and push any button.
Hold ... OK.                                                                  
Move axis 2 to maximum position and push any button.
Hold ... OK.                                                                  
Move axis 3 to minimum position and push any button.
Hold ... OK.                                                                  
Move axis 3 to center position and push any button.
Hold ... OK.                                                                  
Move axis 3 to maximum position and push any button.
Hold ... OK.                                                                  
Move axis 4 to minimum position and push any button.
Hold ... OK.                                                                  
Move axis 4 to center position and push any button.
Hold ... OK.                                                                  
Move axis 4 to maximum position and push any button.
Hold ... OK.                                                                  
Move axis 5 to minimum position and push any button.
Hold ... OK.                                                                  
Move axis 5 to center position and push any button.
Hold ... OK.                                                                  
Move axis 5 to maximum position and push any button.
Hold ... OK.                                                                  

Setting correction to:
Correction for axis 0: broken line, precision: 0.
Coeficients: 128, 128, 4400447, 4364671
Correction for axis 1: broken line, precision: 0.
Coeficients: 129, 129, -4473788, -4436814
Correction for axis 2: broken line, precision: 0.
Coeficients: 0, 0, 2147483647, 2147483647
Correction for axis 3: broken line, precision: 0.
Coeficients: 128, 128, 4400447, 4227201
Correction for axis 4: broken line, precision: 0.
Coeficients: 128, 128, -4549615, -4400447
Correction for axis 5: broken line, precision: 0.
Coeficients: 0, 0, -2105312, 2147483647
```

### Store Calibration

```
banshee@banrc01:~ $ sudo jscal-store /dev/input/js0 

banshee@banrc01:~ $ sudo cat /var/lib/joystick/joystick.state
NAME="Sony PLAYSTATION(R)3 Controller"
jscal -u 6,0,1,2,3,4,5,17,304,305,307,308,310,311,312,313,314,315,316,317,318,544,545,546,547
jscal -s 6,1,0,128,128,4400447,4364671,1,0,129,129,-4473788,-4436814,1,0,0,0,2147483647,2147483647,1,0,128,128,4400447,4227201,1,0,128,128,-4549615,-4400447,1,0,0,0,-2105312,2147483647
```



## Pygame Joystick Demo

https://www.pygame.org/wiki/GettingStarted

```
!sudo apt-get install demo
!sudo apt-get install libsdl2-2.0-0
```


```python
import sys
import pygame
from pygame.locals import *

pygame.init()
pygame.joystick.init()

joysticks = [pygame.joystick.Joystick(i) for i in range(pygame.joystick.get_count())]
for joystick in joysticks:
    print(joystick.get_name())
    
try:
    while True:
        for event in pygame.event.get():
            if event.type == JOYBUTTONDOWN:
                print(event)
            if event.type == JOYBUTTONUP:
                print(event)
            if event.type == JOYAXISMOTION:
                print(event)
            if event.type == JOYHATMOTION:
                print(event)
            if event.type == JOYDEVICEADDED:
                joysticks = [pygame.joystick.Joystick(i) for i in range(pygame.joystick.get_count())]
                for joystick in joysticks:
                    print(joystick.get_name())
            if event.type == JOYDEVICEREMOVED:
                joysticks = [pygame.joystick.Joystick(i) for i in range(pygame.joystick.get_count())]
            if event.type == QUIT:
                pygame.quit()
                sys.exit()
            if event.type == KEYDOWN:
                if event.key == K_ESCAPE:
                    pygame.quit()
                    sys.exit()
except KeyboardInterrupt:
    pygame.quit()
    print('Free pygame')
```

    Sony PLAYSTATION(R)3 Controller
    <Event(1540-JoyButtonUp {'joy': 0, 'instance_id': 1, 'button': 0})>
    <Event(1539-JoyButtonDown {'joy': 0, 'instance_id': 1, 'button': 2})>
    <Event(1540-JoyButtonUp {'joy': 0, 'instance_id': 1, 'button': 2})>
    <Event(1539-JoyButtonDown {'joy': 0, 'instance_id': 1, 'button': 1})>
    <Event(1540-JoyButtonUp {'joy': 0, 'instance_id': 1, 'button': 1})>
    <Event(1536-JoyAxisMotion {'joy': 0, 'instance_id': 1, 'axis': 0, 'value': 0.0})>
    <Event(1536-JoyAxisMotion {'joy': 0, 'instance_id': 1, 'axis': 0, 'value': -0.33335367900631735})>
    <Event(1536-JoyAxisMotion {'joy': 0, 'instance_id': 1, 'axis': 0, 'value': -1.000030518509476})>
    <Event(1536-JoyAxisMotion {'joy': 0, 'instance_id': 1, 'axis': 0, 'value': -0.8564409314249092})>
    <Event(1536-JoyAxisMotion {'joy': 0, 'instance_id': 1, 'axis': 0, 'value': 0.0})>
    <Event(1536-JoyAxisMotion {'joy': 0, 'instance_id': 1, 'axis': 1, 'value': 0.0})>
    <Event(1536-JoyAxisMotion {'joy': 0, 'instance_id': 1, 'axis': 1, 'value': 0.6615497299111911})>
    <Event(1536-JoyAxisMotion {'joy': 0, 'instance_id': 1, 'axis': 1, 'value': 1.0})>
    <Event(1536-JoyAxisMotion {'joy': 0, 'instance_id': 1, 'axis': 1, 'value': 0.3640858180486465})>
    <Event(1536-JoyAxisMotion {'joy': 0, 'instance_id': 1, 'axis': 1, 'value': 0.0})>
    <Event(1536-JoyAxisMotion {'joy': 0, 'instance_id': 1, 'axis': 1, 'value': -0.815424054689169})>
    <Event(1536-JoyAxisMotion {'joy': 0, 'instance_id': 1, 'axis': 1, 'value': -1.000030518509476})>
    <Event(1536-JoyAxisMotion {'joy': 0, 'instance_id': 1, 'axis': 1, 'value': 0.0})>
    <Event(1536-JoyAxisMotion {'joy': 0, 'instance_id': 1, 'axis': 4, 'value': 0.0})>
    <Event(1536-JoyAxisMotion {'joy': 0, 'instance_id': 1, 'axis': 4, 'value': 0.23075045014801476})>
    <Event(1536-JoyAxisMotion {'joy': 0, 'instance_id': 1, 'axis': 4, 'value': 1.0})>
    <Event(1536-JoyAxisMotion {'joy': 0, 'instance_id': 1, 'axis': 4, 'value': 0.8872035889767144})>
    <Event(1536-JoyAxisMotion {'joy': 0, 'instance_id': 1, 'axis': 4, 'value': 0.33332316049684135})>
    <Event(1536-JoyAxisMotion {'joy': 0, 'instance_id': 1, 'axis': 4, 'value': 0.0})>
    <Event(1536-JoyAxisMotion {'joy': 0, 'instance_id': 1, 'axis': 3, 'value': 0.0})>
    <Event(1536-JoyAxisMotion {'joy': 0, 'instance_id': 1, 'axis': 3, 'value': 0.35383159886471144})>
    <Event(1536-JoyAxisMotion {'joy': 0, 'instance_id': 1, 'axis': 3, 'value': 1.0})>
    <Event(1536-JoyAxisMotion {'joy': 0, 'instance_id': 1, 'axis': 3, 'value': 0.9077120273445844})>
    <Event(1536-JoyAxisMotion {'joy': 0, 'instance_id': 1, 'axis': 3, 'value': 0.0})>
    <Event(1536-JoyAxisMotion {'joy': 0, 'instance_id': 1, 'axis': 4, 'value': 0.1076693014313181})>
    <Event(1536-JoyAxisMotion {'joy': 0, 'instance_id': 1, 'axis': 3, 'value': -0.05642872402111881})>
    <Event(1536-JoyAxisMotion {'joy': 0, 'instance_id': 1, 'axis': 4, 'value': 0.32306894131290625})>
    <Event(1536-JoyAxisMotion {'joy': 0, 'instance_id': 1, 'axis': 3, 'value': -0.6615802484206671})>
    <Event(1536-JoyAxisMotion {'joy': 0, 'instance_id': 1, 'axis': 4, 'value': 0.641041291543321})>
    <Event(1536-JoyAxisMotion {'joy': 0, 'instance_id': 1, 'axis': 3, 'value': -0.9487594225898007})>
    <Event(1536-JoyAxisMotion {'joy': 0, 'instance_id': 1, 'axis': 4, 'value': 1.0})>
    <Event(1536-JoyAxisMotion {'joy': 0, 'instance_id': 1, 'axis': 3, 'value': -1.000030518509476})>
    <Event(1536-JoyAxisMotion {'joy': 0, 'instance_id': 1, 'axis': 3, 'value': -0.7744071779534287})>
    <Event(1536-JoyAxisMotion {'joy': 0, 'instance_id': 1, 'axis': 4, 'value': 0.641041291543321})>
    <Event(1536-JoyAxisMotion {'joy': 0, 'instance_id': 1, 'axis': 3, 'value': 0.0})>
    <Event(1536-JoyAxisMotion {'joy': 0, 'instance_id': 1, 'axis': 4, 'value': 0.0})>
    <Event(1539-JoyButtonDown {'joy': 0, 'instance_id': 1, 'button': 3})>
    <Event(1540-JoyButtonUp {'joy': 0, 'instance_id': 1, 'button': 3})>
    <Event(1539-JoyButtonDown {'joy': 0, 'instance_id': 1, 'button': 1})>
    <Event(1539-JoyButtonDown {'joy': 0, 'instance_id': 1, 'button': 2})>
    <Event(1540-JoyButtonUp {'joy': 0, 'instance_id': 1, 'button': 1})>
    <Event(1540-JoyButtonUp {'joy': 0, 'instance_id': 1, 'button': 2})>
    <Event(1539-JoyButtonDown {'joy': 0, 'instance_id': 1, 'button': 4})>
    <Event(1540-JoyButtonUp {'joy': 0, 'instance_id': 1, 'button': 4})>
    <Event(1539-JoyButtonDown {'joy': 0, 'instance_id': 1, 'button': 5})>
    <Event(1540-JoyButtonUp {'joy': 0, 'instance_id': 1, 'button': 5})>
    <Event(1539-JoyButtonDown {'joy': 0, 'instance_id': 1, 'button': 6})>
    <Event(1536-JoyAxisMotion {'joy': 0, 'instance_id': 1, 'axis': 2, 'value': -1.000030518509476})>
    <Event(1536-JoyAxisMotion {'joy': 0, 'instance_id': 1, 'axis': 2, 'value': 1.0})>
    <Event(1536-JoyAxisMotion {'joy': 0, 'instance_id': 1, 'axis': 2, 'value': -0.39490951261940366})>
    <Event(1540-JoyButtonUp {'joy': 0, 'instance_id': 1, 'button': 6})>
    <Event(1536-JoyAxisMotion {'joy': 0, 'instance_id': 1, 'axis': 2, 'value': -1.000030518509476})>
    <Event(1539-JoyButtonDown {'joy': 0, 'instance_id': 1, 'button': 7})>
    <Event(1536-JoyAxisMotion {'joy': 0, 'instance_id': 1, 'axis': 5, 'value': -1.000030518509476})>
    <Event(1536-JoyAxisMotion {'joy': 0, 'instance_id': 1, 'axis': 5, 'value': 1.0})>
    <Event(1536-JoyAxisMotion {'joy': 0, 'instance_id': 1, 'axis': 3, 'value': 0.11792352061525316})>
    <Event(1536-JoyAxisMotion {'joy': 0, 'instance_id': 1, 'axis': 3, 'value': 0.4769127475814081})>
    <Event(1536-JoyAxisMotion {'joy': 0, 'instance_id': 1, 'axis': 3, 'value': 0.8872035889767144})>
    <Event(1536-JoyAxisMotion {'joy': 0, 'instance_id': 1, 'axis': 3, 'value': 1.0})>
    <Event(1536-JoyAxisMotion {'joy': 0, 'instance_id': 1, 'axis': 4, 'value': -0.6718344676046022})>
    <Event(1536-JoyAxisMotion {'joy': 0, 'instance_id': 1, 'axis': 4, 'value': -1.000030518509476})>
    <Event(1540-JoyButtonUp {'joy': 0, 'instance_id': 1, 'button': 7})>
    <Event(1536-JoyAxisMotion {'joy': 0, 'instance_id': 1, 'axis': 5, 'value': -1.000030518509476})>
    <Event(1536-JoyAxisMotion {'joy': 0, 'instance_id': 1, 'axis': 4, 'value': -0.8051698355052339})>
    <Event(1536-JoyAxisMotion {'joy': 0, 'instance_id': 1, 'axis': 4, 'value': -0.16925565355388042})>
    <Event(1536-JoyAxisMotion {'joy': 0, 'instance_id': 1, 'axis': 4, 'value': -0.1487472151860103})>
    <Event(1536-JoyAxisMotion {'joy': 0, 'instance_id': 1, 'axis': 4, 'value': -0.09744560075685904})>
    <Event(1536-JoyAxisMotion {'joy': 0, 'instance_id': 1, 'axis': 4, 'value': -0.025666066469313638})>
    <Event(1536-JoyAxisMotion {'joy': 0, 'instance_id': 1, 'axis': 3, 'value': 0.9487289040803247})>
    <Event(1536-JoyAxisMotion {'joy': 0, 'instance_id': 1, 'axis': 4, 'value': 0.0})>
    <Event(1536-JoyAxisMotion {'joy': 0, 'instance_id': 1, 'axis': 3, 'value': 0.8564104129154332})>
    <Event(1536-JoyAxisMotion {'joy': 0, 'instance_id': 1, 'axis': 3, 'value': 0.7846308786278878})>
    <Event(1536-JoyAxisMotion {'joy': 0, 'instance_id': 1, 'axis': 3, 'value': 0.7641224402600177})>
    <Event(1536-JoyAxisMotion {'joy': 0, 'instance_id': 1, 'axis': 3, 'value': 0.7435834833826716})>
    <Event(1536-JoyAxisMotion {'joy': 0, 'instance_id': 1, 'axis': 3, 'value': 0.7230750450148015})>
    <Event(1536-JoyAxisMotion {'joy': 0, 'instance_id': 1, 'axis': 3, 'value': 0.7025666066469314})>
    <Event(1536-JoyAxisMotion {'joy': 0, 'instance_id': 1, 'axis': 3, 'value': 0.6820581682790613})>
    <Event(1536-JoyAxisMotion {'joy': 0, 'instance_id': 1, 'axis': 3, 'value': 0.6615497299111911})>
    <Event(1536-JoyAxisMotion {'joy': 0, 'instance_id': 1, 'axis': 3, 'value': 0.641041291543321})>
    <Event(1536-JoyAxisMotion {'joy': 0, 'instance_id': 1, 'axis': 3, 'value': 0.6205023346659749})>
    <Event(1536-JoyAxisMotion {'joy': 0, 'instance_id': 1, 'axis': 3, 'value': 0.5692312387462997})>
    <Event(1536-JoyAxisMotion {'joy': 0, 'instance_id': 1, 'axis': 3, 'value': 0.5487228003784295})>
    <Event(1536-JoyAxisMotion {'joy': 0, 'instance_id': 1, 'axis': 3, 'value': 0.5282143620105594})>
    <Event(1536-JoyAxisMotion {'joy': 0, 'instance_id': 1, 'axis': 3, 'value': 0.5077059236426893})>
    <Event(1536-JoyAxisMotion {'joy': 0, 'instance_id': 1, 'axis': 3, 'value': 0.48716696676534316})>
    <Event(1536-JoyAxisMotion {'joy': 0, 'instance_id': 1, 'axis': 3, 'value': 0.46665852839747307})>
    <Event(1536-JoyAxisMotion {'joy': 0, 'instance_id': 1, 'axis': 3, 'value': 0.44615009002960293})>
    <Event(1536-JoyAxisMotion {'joy': 0, 'instance_id': 1, 'axis': 3, 'value': 0.4153874324777978})>
    <Event(1536-JoyAxisMotion {'joy': 0, 'instance_id': 1, 'axis': 3, 'value': 0.39487899410992766})>
    <Event(1536-JoyAxisMotion {'joy': 0, 'instance_id': 1, 'axis': 3, 'value': 0.3743400372325816})>
    <Event(1536-JoyAxisMotion {'joy': 0, 'instance_id': 1, 'axis': 3, 'value': 0.35383159886471144})>
    <Event(1536-JoyAxisMotion {'joy': 0, 'instance_id': 1, 'axis': 3, 'value': 0.32306894131290625})>
    <Event(1536-JoyAxisMotion {'joy': 0, 'instance_id': 1, 'axis': 3, 'value': 0.2512894070253609})>
    <Event(1536-JoyAxisMotion {'joy': 0, 'instance_id': 1, 'axis': 3, 'value': 0.18973357341227454})>
    <Event(1536-JoyAxisMotion {'joy': 0, 'instance_id': 1, 'axis': 3, 'value': 0.15897091586046938})>
    <Event(1536-JoyAxisMotion {'joy': 0, 'instance_id': 1, 'axis': 3, 'value': 0.1282082583086642})>
    <Event(1536-JoyAxisMotion {'joy': 0, 'instance_id': 1, 'axis': 3, 'value': 0.07690664387951293})>
    <Event(1536-JoyAxisMotion {'joy': 0, 'instance_id': 1, 'axis': 3, 'value': 0.015381328775902585})>
    <Event(1536-JoyAxisMotion {'joy': 0, 'instance_id': 1, 'axis': 3, 'value': 0.0})>
    <Event(1539-JoyButtonDown {'joy': 0, 'instance_id': 1, 'button': 7})>
    <Event(1536-JoyAxisMotion {'joy': 0, 'instance_id': 1, 'axis': 5, 'value': 1.0})>
    <Event(1536-JoyAxisMotion {'joy': 0, 'instance_id': 1, 'axis': 3, 'value': 0.08716086306344799})>
    <Event(1536-JoyAxisMotion {'joy': 0, 'instance_id': 1, 'axis': 3, 'value': 0.30256050294503617})>
    <Event(1536-JoyAxisMotion {'joy': 0, 'instance_id': 1, 'axis': 3, 'value': 0.3743400372325816})>
    <Event(1536-JoyAxisMotion {'joy': 0, 'instance_id': 1, 'axis': 3, 'value': 0.42564165166173284})>
    <Event(1536-JoyAxisMotion {'joy': 0, 'instance_id': 1, 'axis': 3, 'value': 0.5179601428266244})>
    <Event(1536-JoyAxisMotion {'joy': 0, 'instance_id': 1, 'axis': 3, 'value': 0.5999938962981048})>
    <Event(1536-JoyAxisMotion {'joy': 0, 'instance_id': 1, 'axis': 3, 'value': 0.6615497299111911})>
    <Event(1536-JoyAxisMotion {'joy': 0, 'instance_id': 1, 'axis': 3, 'value': 0.6820581682790613})>
    <Event(1536-JoyAxisMotion {'joy': 0, 'instance_id': 1, 'axis': 3, 'value': 0.7025666066469314})>
    <Event(1536-JoyAxisMotion {'joy': 0, 'instance_id': 1, 'axis': 3, 'value': 0.7128208258308664})>
    <Event(1536-JoyAxisMotion {'joy': 0, 'instance_id': 1, 'axis': 3, 'value': 0.7230750450148015})>
    <Event(1536-JoyAxisMotion {'joy': 0, 'instance_id': 1, 'axis': 3, 'value': 0.7025666066469314})>
    <Event(1540-JoyButtonUp {'joy': 0, 'instance_id': 1, 'button': 7})>
    <Event(1536-JoyAxisMotion {'joy': 0, 'instance_id': 1, 'axis': 3, 'value': 0.15897091586046938})>
    <Event(1536-JoyAxisMotion {'joy': 0, 'instance_id': 1, 'axis': 5, 'value': -1.000030518509476})>
    <Event(1536-JoyAxisMotion {'joy': 0, 'instance_id': 1, 'axis': 3, 'value': 0.0})>
    Free pygame



```python

```
