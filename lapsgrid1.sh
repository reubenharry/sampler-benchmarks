IMODEL=1

for TASK in {0..4}
do
  echo $TASK
  shifter --image=reubenharry/cosmo:1.0 python3 -m papers.LAPS.main $IMODEL $TASK
done

