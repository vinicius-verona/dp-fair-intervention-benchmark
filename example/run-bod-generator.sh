echo Executing BoD

# python3 DPF_Benchmark-BoD.py 5 42 253 4112 32645 --combo $COMBO  > log/BoD-$COMBO-batch-1.out 2> log/BoD-$COMBO-batch-1.err &
# python3 DPF_Benchmark-BoD.py 602627 153073 53453 178753 243421 --combo $COMBO  > log/BoD-$COMBO-batch-2.out 2> log/BoD-$COMBO-batch-2.err &
# python3 DPF_Benchmark-BoD.py 767707 113647 796969 553067 96797 --combo $COMBO  > log/BoD-$COMBO-batch-3.out 2> log/BoD-$COMBO-batch-3.err &
# python3 DPF_Benchmark-BoD.py 133843 6977 460403 126613 583879 --combo $COMBO  > log/BoD-$COMBO-batch-4.out 2> log/BoD-$COMBO-batch-4.err &

# python3 DPF_DataGeneration-BoD.py --combo $COMBO -s 5 42 253 4112 > log/DataGeneration/BoD-${COMBO}-PETS-batch-1.out 2> log/DataGeneration/BoD-${COMBO}-PETS-batch-1.err &
# python3 DPF_DataGeneration-BoD.py --combo $COMBO -s 32645 602627 153073 53453 > log/DataGeneration/BoD-${COMBO}-PETS-batch-2.out 2> log/DataGeneration/BoD-${COMBO}-PETS-batch-2.err &
# python3 DPF_DataGeneration-BoD.py --combo $COMBO -s 178753 243421 767707 113647 > log/DataGeneration/BoD-${COMBO}-PETS-batch-3.out 2> log/DataGeneration/BoD-${COMBO}-PETS-batch-3.err &
# python3 DPF_DataGeneration-BoD.py --combo $COMBO -s 796969 553067 96797 133843 > log/DataGeneration/BoD-${COMBO}-PETS-batch-4.out 2> log/DataGeneration/BoD-${COMBO}-PETS-batch-4.err &
# python3 DPF_DataGeneration-BoD.py --combo $COMBO -s 6977 460403 126613 583879 > log/DataGeneration/BoD-${COMBO}-PETS-batch-5.out 2> log/DataGeneration/BoD-${COMBO}-PETS-batch-5.err &


python3 DPF_DataGeneration-BoD.py -s 5 42 253 4112 > log/DataGeneration/BoD-PETS-batch-1.out 2> log/DataGeneration/BoD-PETS-batch-1.err &
python3 DPF_DataGeneration-BoD.py -s 32645 602627 153073 53453 > log/DataGeneration/BoD-PETS-batch-2.out 2> log/DataGeneration/BoD-PETS-batch-2.err &
python3 DPF_DataGeneration-BoD.py -s 178753 243421 767707 113647 > log/DataGeneration/BoD-PETS-batch-3.out 2> log/DataGeneration/BoD-PETS-batch-3.err &
python3 DPF_DataGeneration-BoD.py -s 796969 553067 96797 133843 > log/DataGeneration/BoD-PETS-batch-4.out 2> log/DataGeneration/BoD-PETS-batch-4.err &
python3 DPF_DataGeneration-BoD.py -s 6977 460403 126613 583879 > log/DataGeneration/BoD-PETS-batch-5.out 2> log/DataGeneration/BoD-PETS-batch-5.err &