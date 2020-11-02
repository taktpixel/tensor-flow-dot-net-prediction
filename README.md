# POODL model prediction sample

Need to install .NET Core 3.1
https://dotnet.microsoft.com/download/visual-studio-sdks?utm_source=getdotnetsdk&utm_medium=referral

## Command line arguments
```
  -t, --method        Required. Prediction mode

  -i, --image         Predicted image path list

  -l, --image-list    Predicted image path list file (*.csv)

  -m, --model         Required. Deep learning model

  -c, --label         Required. Label name

  -s, --batch-size    Batch size

  -o, --output

  -v, --verbose       Set output to verbose messages.

  --help              Display this help screen.

  --version           Display version information.
```

## single prediction

command example in (bin\Debug\netcoreapp3.1 directory)
```
predict.exe --method single --image images\tpxtech_00000224.m_00000000.png,images\tpxtech_00000224.m_00000001.png --model models\trained.pb --label models\label.txt --output predict_result.csv --verbose
```

by dotnet command in (repository top directory)
```
dotnet run --project src/TensorFlowNET.Examples --method single --image src/TensorFlowNET.Examples/bin/Debug/netcoreapp3.1/images/tpxtech_00000224.m_00000000.png,src/TensorFlowNET.Examples/bin/Debug/netcoreapp3.1/images/tpxtech_00000224.m_00000001.png --model src/TensorFlowNET.Examples/bin/Debug/netcoreapp3.1/models/trained.pb --label src/TensorFlowNET.Examples/bin/Debug/netcoreapp3.1/models/label.txt --output src/TensorFlowNET.Examples/bin/Debug/netcoreapp3.1/predict_result.csv --verbose
```

## batch prediction

command example in (bin\Debug\netcoreapp3.1 directory)
```
predict.exe --method batch --image-list images\list.csv --model models\trained.pb --label models\label.txt --batch-size 32 --output predict_result.csv --verbose
```

by dotnet command in (repository top directory)
```
dotnet run --project src/TensorFlowNET.Examples --method batch --image-list src/TensorFlowNET.Examples/bin/Debug/netcoreapp3.1/images/list.csv --model src/TensorFlowNET.Examples/bin/Debug/netcoreapp3.1/models/trained.pb --label src/TensorFlowNET.Examples/bin/Debug/netcoreapp3.1/models/label.txt --batch-size 32  --output src/TensorFlowNET.Examples/bin/Debug/netcoreapp3.1/predict_result.csv --verbose
```



