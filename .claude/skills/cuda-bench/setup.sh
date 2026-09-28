#!/bin/bash
# Switch to High Performance power plan and disable turbo/throttling for stable benchmarks
powershell.exe -NoProfile -Command "powercfg /getactivescheme" | grep -oE '[a-f0-9-]{36}' > /tmp/bench-setup.prev
powershell.exe -NoProfile -Command "powercfg /setactive 8c5e7fda-e8bf-4a96-9a85-a6e23a8c635c"
powershell.exe -NoProfile -Command "powercfg -setacvalueindex SCHEME_CURRENT SUB_PROCESSOR PROCTHROTTLEMIN 100"
powershell.exe -NoProfile -Command "powercfg -setacvalueindex SCHEME_CURRENT SUB_PROCESSOR PROCTHROTTLEMAX 100"
powershell.exe -NoProfile -Command "powercfg -setacvalueindex SCHEME_CURRENT SUB_PROCESSOR PERFBOOSTMODE 0"
powershell.exe -NoProfile -Command "powercfg /setactive SCHEME_CURRENT"
echo "Power plan switched, turbo disabled"
