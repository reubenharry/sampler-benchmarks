
for IMODEL in {0..4}
do
  echo $IMODEL
  shifter --image=jrobnik/mcmc:1.0 python3 -m papers.LAPS.main $IMODEL 0
done

