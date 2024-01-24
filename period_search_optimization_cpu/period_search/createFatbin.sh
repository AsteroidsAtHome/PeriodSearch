#!/bin/bash

ARMAPP=Period_search_BOINC_macos_10215_arm64_Release
X64APP=Period_search_BOINC_macos_10215_x64_cpu_Release
FATBIN=Period_search_BOINC_macos_10215_fatbin_cpu_Release

lipo -create -arch arm64 mac_builds/$ARMAPP -arch x86_64 mac_builds/$X64APP -output mac_builds/$FATBIN

lipo mac_builds/$FATBIN -info