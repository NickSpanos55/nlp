#!/usr/bin/env bash

n=${1}
mode=${2}

#Compile wrong word acceptor
fstcompile -isymbols=vocab/chars.syms --acceptor=true vocab/fsts/wrong_word.txt vocab/fsts/wrong_word.bin.fst

fstprint -isymbols=vocab/chars.syms -osymbols=vocab/chars.syms vocab/fsts/wrong_word.bin.fst > vocab/fsts/wrong_word_print.txt

fstdraw -isymbols=vocab/chars.syms -osymbols=vocab/chars.syms vocab/fsts/wrong_word.bin.fst | dot -Tpng  > vocab/fsts/wrong_word.png

#Compose wrong word acceptor with composed orthograph

if [ "${mode}" = 'L' ]
then
	fstcompose vocab/fsts/wrong_word.bin.fst vocab/fsts/LV.bin.fst vocab/fsts/check.bin.fst
elif [ "${mode}" = 'E' ]
then
	fstcompose vocab/fsts/wrong_word.bin.fst vocab/fsts/EV.bin.fst vocab/fsts/check.bin.fst
elif [ "${mode}" = 'LW' ]
then
	fstcompose vocab/fsts/wrong_word.bin.fst vocab/fsts/LVW.bin.fst vocab/fsts/check.bin.fst
elif [ "${mode}" =  'EW' ]
then
	fstcompose vocab/fsts/wrong_word.bin.fst vocab/fsts/EVW.bin.fst vocab/fsts/check.bin.fst
fi

#Get first n shortest paths through graph

fstshortestpath --nshortest=${n} vocab/fsts/check.bin.fst > vocab/fsts/check_first.bin.fst

fstdraw -isymbols=vocab/chars.syms -osymbols=vocab/words.syms vocab/fsts/check_first.bin.fst | dot -Tpng  > vocab/fsts/check_first.png

#Remove epsilon transitions

fstrmepsilon vocab/fsts/check_first.bin.fst > vocab/fsts/check_second.bin.fst

#Sort the top transitions

fsttopsort vocab/fsts/check_second.bin.fst vocab/fsts/checked.bin.fst

#Write print output and draw the orthograph

fstprint -isymbols=vocab/chars.syms -osymbols=vocab/words.syms vocab/fsts/checked.bin.fst > vocab/fsts/checked_print.txt

fstdraw -isymbols=vocab/chars.syms -osymbols=vocab/words.syms vocab/fsts/checked.bin.fst | dot -Tpng  > vocab/fsts/checked.png

rm vocab/fsts/check_first.bin.fst
rm vocab/fsts/check_second.bin.fst
