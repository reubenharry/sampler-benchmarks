IMODEL=4

for IMODEL in {0..4}
do
  echo $IMODEL
  shifter --image=reubenharry/cosmo:1.0 python3 -m papers.LAPS.main $IMODEL 4
done


  shifter --image=reubenharry/cosmo:1.0 python3 -m papers.LAPS.main $IMODEL 4
