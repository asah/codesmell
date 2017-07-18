#!/bin/bash
#
# supports restartability & incremental development, hence all the 'if' statements
#
# rm all-hashes.txt fix-commits/hashes.txt clean-commits/hashes.txt
# ==> re-check if you're missing any hashes, e.g. changing algorithm which decides
#     which commits to keep
#
# rm fix-commits clean-commits
# ==> re-create the content files... expensive
#
# touch fix-commits/hashes.txt clean-commits/hashes.txt
# ==> re-create githubLabeledTrainData.tsv
# 

if [ $# == 0 ]; then
    echo "Usage: $0 <c|java|python|...> [repo directory]";
    exit 1
fi

DBG=0   # set to 1 for debugging
CLEANDIR=./clean-commits
FIXDIR=./fix-commits
GET_ONE_COMMIT=../../get-commit.sh   # relative to each repo directory
ALL_HASHES=all-hashes.txt
FILETYPE=$1
DIR=$2
if [ "x$DIR" = "x" ]; then DIR=.; fi  # default to current directory

if [ ! -d $DIR/.git ]; then echo "$2 must be a git repo directory"; exit 1; fi


cd $DIR

if [ ! -f $ALL_HASHES ]; then
  echo "getting all commit hashes and commit messages... ($ALL_HASHES)"
  git log --format="%h %s" > $ALL_HASHES
fi

# explicit /bin/rm for macOS
mkdir -p $CLEANDIR
mkdir -p $FIXDIR
if [ ! -f $CLEANDIR/hashes.txt ]; then
  echo "scanning for clean commits ==> $CLEANDIR/hashes.txt"
  perl -ane 'next if / Merge/; print if m@^[0-9a-z/-]+ ([#]?[a-z0-9]+: *)*(adds?|adding|added|use|update|implement|removed?|switch|split|refactor|make|improve|simplify|optimize|create|disable|handle|change|allow|clean|cleanup|drop|document|move|more|adjust|convert|perf|rename|replace|introduce|support):? @i;' < $ALL_HASHES | sort > $CLEANDIR/hashes.txt
fi
if [ ! -f $FIXDIR/hashes.txt ]; then
  echo "scanning for bugfix commits ==> $FIXDIR/hashes.txt"
  # fix includes "bugfix"
  # MDEV is for mariadb/mysql
  perl -ane 'next if / Merge/; print if /[^a-z](fix|fixed|fixes|revert|MDEV-)[^a-z]/i; ' < $ALL_HASHES | sort > $FIXDIR/hashes.txt
fi

num_fix_chgs=0
for h in `perl -ane 'print "$F[0]\n"' < $FIXDIR/hashes.txt`; do
  if [ ! -f "$FIXDIR/commit-$h.txt" ]; then
    if [ "x$num_fix_chgs" == "x0" ]; then echo -n "fetching fix commits: "; fi
    if [ "x$DBG" == "x1" ]; then echo "$GET_ONE_COMMIT $h $ext > $FIXDIR/commit-$h.txt";
    elif (( $num_fix_chgs % 100 == 0 )); then echo -n "."; fi
    $GET_ONE_COMMIT $h $FILETYPE > $FIXDIR/commit-$h.txt
    num_fix_chgs=$((num_fix_chgs+1))
  fi
done
# set file mod time for githubLabeledTrainData.tsv below
if [[ $num_fix_chgs -gt 0 ]]; then echo ""; touch $FIXDIR/hashes.txt; fi

# experimental removal of fix-hashes that are later implicated in bugfixes...
# requires https://github.com/dmnd/git-diff-blame (with modification for git show vs diff...)
if [ 0 = 1 ]; then
  GIT_SHOW_BLAME=~/codesmell/git-show-blame
  if [ ! -f $FIXDIR/hashes-before-sorted.txt ]; then
    if [ ! -f $FIXDIR/hashes-before.txt ]; then
      echo "generating $FIXDIR/hashes-before.txt using $GIT_SHOW_BLAME"
      for h in `perl -ane 'print "$F[0]\n"' < $FIXDIR/hashes.txt`; do
        eval $GIT_SHOW_BLAME $h $ext | perl -ne 'print "$1\n" if (/^^..31m - +^?([0-9a-f]+)/);'| sort | uniq >> $FIXDIR/hashes-before.txt
      done
    fi
    echo "sorting befor-commits..."
    sort $FIXDIR/hashes-before.txt > $FIXDIR/hashes-before-sorted.txt
  fi
  join -a 1 -v 2 -j 1 $CLEANDIR/hashes.txt $FIXDIR/hashes-before-sorted.txt > $CLEANDIR/hashes-clean.txt
fi

hashes=$CLEANDIR/hashes.txt
if [ -f $CLEANDIR/hashes-clean.txt ]; then
    hashes=$CLEANDIR/hashes-clean.txt
fi
num_clean_chgs=0
for h in `perl -ane 'print "$F[0]\n"' < $hashes`; do
  if [ ! -f "$CLEANDIR/commit-$h.txt" ]; then
    if [ "x$num_clean_chgs" == "x0" ]; then echo -n "fetching clean commits: "; fi
    if [ "x$DBG" == "x1" ]; then echo "$GET_ONE_COMMIT $h $ext > $CLEANDIR/commit-$h.txt";
    elif (( $num_clean_chgs % 100 == 0 )); then echo -n "."; fi
    $GET_ONE_COMMIT $h $FILETYPE > $CLEANDIR/commit-$h.txt
    num_clean_chgs=$((num_clean_chgs+1))
  fi
done
# set file mod time for githubLabeledTrainData.tsv below
if [[ $num_clean_chgs -gt 0 ]]; then echo ""; touch $CLEANDIR/hashes.txt; fi

# if githubLabeledTrainData.tsv needs refreshing
if [ "x$DBG" == "x1" ]; then ls -l githubLabeledTrainData.tsv $CLEANDIR/hashes.txt $FIXDIR/hashes.txt; fi
if [ "$CLEANDIR/hashes.txt" -nt "githubLabeledTrainData.tsv" ] && [ "$CLEANDIR/hashes.txt" -nt "githubLabeledTrainData.tsv" ]; then
  # gather commits into a single tab-separated file
  # 1. drop overly short/long commits.  TODO: maybe try keeping the long ones?
  # 2. add a TSV header
  # 3. set sentiment column based on directory name (clean-commits => 0, fix-commits => 1)
  # 4. replace newlines with \n and tabs with \t
  /bin/rm -f githubLabeledTrainData.tsv
  find clean-commits fix-commits -name "commit-*.txt" | perl -ne 'BEGIN{print "id\tsentiment\treview\n";}m@(clean|fix)-commits/commit-(.+)[.]@; $h=$2;$cls=($1 eq "clean" ? 0 : 1);$content=`cat $_`;next if length($content)>10000; next if length($content)<20; $content=~s/\n/\\n/g;$content=~s/\t/\\t/g;print "\"$h\"\t$cls\t$content\n"' > githubLabeledTrainData.tsv
fi

wc -l githubLabeledTrainData.tsv|perl -ane 'printf("%8d commits found for %s", int($F[0]), `pwd|xargs basename`);'
