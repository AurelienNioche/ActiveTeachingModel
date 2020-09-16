#!/bin/sh
#
# Make gunzipped tarball of ./fig with date

tar -cf $(date -I)-fig.tar.gz fig/
