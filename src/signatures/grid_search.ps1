$startTime = Get-Date
$hyperparameters = @(
    @{learning_rate=0.0001; weight_decay=0.0001},
    @{learning_rate=0.0001; weight_decay=0.001},
    @{learning_rate=0.0001; weight_decay=0.01},
    @{learning_rate=0.0001; weight_decay=0.1}
    @{learning_rate=0.001; weight_decay=0},
    @{learning_rate=0.001; weight_decay=0.0001},
    @{learning_rate=0.001; weight_decay=0.001},
    @{learning_rate=0.001; weight_decay=0.01},
    @{learning_rate=0.001; weight_decay=0.1},
    @{learning_rate=0.0001; weight_decay=0},
    @{learning_rate=0.01; weight_decay=0},
    @{learning_rate=0.01; weight_decay=0.0001},
    @{learning_rate=0.01; weight_decay=0.001},
    @{learning_rate=0.01; weight_decay=0.01},
    @{learning_rate=0.01; weight_decay=0.1},
    @{learning_rate=0.1; weight_decay=0},
    @{learning_rate=0.1; weight_decay=0.0001},
    @{learning_rate=0.1; weight_decay=0.001},
    @{learning_rate=0.1; weight_decay=0.01},
    @{learning_rate=0.1; weight_decay=0.1}
)
foreach ($params in $hyperparameters) {
    $learningRate = $params.learning_rate
    $wd = $params.weight_decay

    Write-Host "Running train_all_signs.py with learning rate $learningRate and weight decay $wd"

    python C:\Users\inserm\Documents\histo_sign\src\signatures\train_all_sign.py `
    --col_signs C:\Users\inserm\Documents\histo_sign\dataset\col_names.txt `
    --export_folder "C:\Users\inserm\Documents\histo_sign\trainings\grid_search\lr_${learningRate}_wd_${wd}" `
    --lr $learningRate `
    --wd $wd
}
$endTime = Get-Date
$timeTaken = New-TimeSpan -Start $startTime -End $endTime
# Write-Host "Grid search took $($timeTaken.Hours):$($timeTaken.Minutes):$($timeTaken.Seconds)"
Write-Host "Grid search took $timeTaken"