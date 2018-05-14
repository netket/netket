#!/bin/sh
for f in $2; do
  echo $f
  cat $1 $f > $f.new
  mv $f.new $f
done
