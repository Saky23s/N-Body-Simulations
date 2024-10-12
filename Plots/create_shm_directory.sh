#!/bin/bash 

if [ -d /dev/shm/data ]; then
    rm -f /dev/shm/data/*.csv >/dev/null
    rm -f /dev/shm/data/*.bin >/dev/null
else
    mkdir /dev/shm/data >/dev/null
fi