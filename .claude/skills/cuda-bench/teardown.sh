#!/bin/bash
powershell.exe -NoProfile -Command "powercfg /setactive $(cat /tmp/bench-setup.prev)"
rm /tmp/bench-setup.prev
echo "Teardown success"
