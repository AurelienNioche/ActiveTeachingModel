#!/bin.sh
#
# Search all files at _project root_ for "HEAD"

for file in $(ls); do grep "HEAD" $file && echo $file; done
