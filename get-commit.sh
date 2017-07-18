#!/bin/bash

HASH=$1

FILETYPE=$2
if [ "x$FILETYPE" = "xpython" ]; then      ext='"*.py" "*.py"';
elif [ "x$FILETYPE" = "xjava" ]; then      ext='"*.java"';
elif [ "x$FILETYPE" = "xc_and_cpp" ]; then ext='"*.c" "*.h" "*.cc" "*.cpp" "*.cxx" "*.C "*.H""';
elif [ "x$FILETYPE" = "xjs_backend" ]; then   ext='"*.js" "*.ejs"';
elif [ "x$FILETYPE" = "xjs_frontend" ]; then  ext='"*.js" "*.ejs"';
elif [ "x$FILETYPE" = "xjs" ]; then           ext='"*.js" "*.ejs"';
else echo "unknown extension type, try <c_and_cpp|java|python|...>"; exit 1;
fi

eval git show -U5 --oneline $HASH $ext | perl -ne 'BEGIN{<>;<>;}next if m/^(diff |index |--|[+@])/;s@^-@ @;s@\t@\\t@;print;'

