#!/bin/sh

git show -U5 --oneline $* | tail -n +2 | perl -ne 'next if m/^(diff |index |--|[+@])/;s@^-@ @;s@\t@\\t@;print;'
