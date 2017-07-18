#!/bin/sh
#
# run this from each language directory

MAX_SUBPROCESSES=16   # assumes beefy machine, e.g. same one for training

dir=`pwd`
if [ ! -f "$dir/LANG_DIR.md" ]; then
    echo "ERROR! missing $dir/LANG_DIR.md"
    echo "  If this directory contains valid repos in a given language,"
    echo "  then create $dir/LANG_DIR.md (content doesn't matter)"
    exit 1
fi
typ=`basename $dir`  # e.g. c_and_cpp

# subdirs that look like repos...
ls -1 -d */.git|perl -ne 's@/.git@@;print;' | xargs --max-procs=$MAX_SUBPROCESSES -n 1 ../get-commits-one-repo.sh $typ 
