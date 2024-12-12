#!/usr/bin/env bash 

fstcompile -isymbols=vocab/chars.syms -osymbols=vocab/words.syms vocab/fsts/V.txt >vocab/fsts/V.bin.fst

fstprint -isymbols=vocab/chars.syms -osymbols=vocab/words.syms vocab/fsts/V.bin.fst > vocab/fsts/V_print.txt

fstrmepsilon vocab/fsts/V.bin.fst > vocab/fsts/V_opt_1.bin.fst

fstdeterminize vocab/fsts/V_opt_1.bin.fst > vocab/fsts/V_opt_2.bin.fst

fstminimize vocab/fsts/V_opt_2.bin.fst > vocab/fsts/V_opt.bin.fst

fstprint -isymbols=vocab/chars.syms -osymbols=vocab/words.syms vocab/fsts/V_opt.bin.fst > vocab/fsts/V_opt_print.txt

#Compile Low-Word Acceptors and save images after each optimization
fstcompile -isymbols=vocab/chars.syms -osymbols=vocab/words.syms vocab/fsts/V_low.txt vocab/fsts/V_low.bin.fst

fstprint -isymbols=vocab/chars.syms -osymbols=vocab/words.syms vocab/fsts/V_low.bin.fst > vocab/fsts/V_low_print.txt

fstdraw -isymbols=vocab/chars.syms -osymbols=vocab/words.syms vocab/fsts/V_low.bin.fst | dot -Tpng  > vocab/fsts/V_low.png


fstrmepsilon vocab/fsts/V_low.bin.fst > vocab/fsts/V_low_eps.bin.fst

fstdraw -isymbols=vocab/chars.syms -osymbols=vocab/words.syms vocab/fsts/V_low_eps.bin.fst | dot -Tpng  > vocab/fsts/V_low_opt_eps.png


fstdeterminize vocab/fsts/V_low_eps.bin.fst >vocab/fsts/V_low_det.bin.fst

fstdraw -isymbols=vocab/chars.syms -osymbols=vocab/words.syms vocab/fsts/V_low_det.bin.fst | dot -Tpng  > vocab/fsts/V_low_opt_det.png


fstminimize vocab/fsts/V_low_det.bin.fst >vocab/fsts/V_low_opt.bin.fst

fstprint -isymbols=vocab/chars.syms -osymbols=vocab/words.syms vocab/fsts/V_low_opt.bin.fst > vocab/fsts/V_low_opt_print.txt

fstdraw -isymbols=vocab/chars.syms -osymbols=vocab/words.syms vocab/fsts/V_low_opt.bin.fst | dot -Tpng  > vocab/fsts/V_low_opt_min.png