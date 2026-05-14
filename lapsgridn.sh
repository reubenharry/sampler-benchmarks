for ITASK in {0..5}
do
  echo $IMODEL
  shifter --image=reubenharry/cosmo:1.0 python3 -m papers.LAPS.main 2 $ITASK
done



