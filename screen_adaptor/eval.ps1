# run_evaluation.ps1
param(
    [string]$LutPath = "lut.pt",
    [string]$BaseOutputDir = "output",
    [string[]]$Datasets = @(
        "D:\workspace\master\3DGS\Vulkan\display_project\screen_adaptor\datasets\genshin_impact",
        "D:\workspace\master\3DGS\Vulkan\display_project\screen_adaptor\datasets\delta_force",
        "D:\workspace\master\3DGS\Vulkan\display_project\screen_adaptor\datasets\cf",
        "D:\workspace\master\3DGS\Vulkan\display_project\screen_adaptor\datasets\dfm300",
        "D:\workspace\master\3DGS\Vulkan\display_project\screen_adaptor\datasets\miHoYo",
        "D:\workspace\master\3DGS\Vulkan\display_project\screen_adaptor\datasets\sgame0",
        "D:\workspace\master\3DGS\Vulkan\display_project\screen_adaptor\datasets\jkchess",
        "D:\workspace\master\3DGS\Vulkan\display_project\screen_adaptor\datasets\nrc"
    ),
    [string]$PowerWeights = "0.229 0.243 0.526",
    [string]$EvalMode = "--eval-mode"
    # [string]$EvalMode = ""
)

# 创建日志目录
$LogDir = "evaluation_logs"
New-Item -ItemType Directory -Force -Path $LogDir | Out-Null

# 存储所有结果的汇总
$AllResults = @()

foreach ($dataset in $Datasets) {
    # 获取数据集名称（从路径中提取最后一个文件夹名）
    $datasetName = Split-Path $dataset -Leaf
    $datasetOutputDir = Join-Path $BaseOutputDir $datasetName
    
    # 创建输出目录
    New-Item -ItemType Directory -Force -Path $datasetOutputDir | Out-Null
    
    # 设置JSON输出文件名
    $jsonOutput = Join-Path $LogDir "$datasetName.json"
    
    Write-Host "========================================" -ForegroundColor Cyan
    Write-Host "Processing dataset: $datasetName" -ForegroundColor Yellow
    Write-Host "Input: $dataset" -ForegroundColor Gray
    Write-Host "Output: $datasetOutputDir" -ForegroundColor Gray
    Write-Host "========================================" -ForegroundColor Cyan
    
    # 使用 conda 3dgs 环境中的 python
    $pythonExe = "D:\miniconda3\envs\3dgs\python.exe"
    
    # 构建命令
    $cmdArgs = @(
        "-m", "src.screen_adaptor.eval",
        "--max-images", "1",
        "--scene-manifest", "outputs\scene_manifest.json",
        "--input-dir", $dataset,
        "--output-dir", $datasetOutputDir,
        "--power-weights"
    ) + ($PowerWeights -split '\s+') + @(
        "--eval-mode",
        "--json-output", $jsonOutput
    )
    
    Write-Host "Executing: $pythonExe $($cmdArgs -join ' ')" -ForegroundColor Green
    
    # 使用 & 操作符执行并等待完成（继承当前环境）
    $output = & $pythonExe $cmdArgs 2>&1
    $output | Out-String | Write-Host
    
    # 等待文件系统同步
    Start-Sleep -Milliseconds 500
    
    # 读取JSON结果
    if (Test-Path $jsonOutput) {
        $jsonRaw = Get-Content $jsonOutput -Raw
        $jsonContent = $jsonRaw | ConvertFrom-Json
        $summary = $jsonContent.summary
        $AllResults += [PSCustomObject]@{
            Dataset = $datasetName
            TotalImages = $summary.total_images
            AvgSaving = $summary.average_saving_percent
            AvgPSNR = $summary.average_psnr
            AvgSSIM = $summary.average_ssim
            AvgMetaM = $summary.average_metametric
        }
    }
    
    Write-Host ""
}

# 显示所有数据集的汇总结果
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "SUMMARY FOR ALL DATASETS" -ForegroundColor Yellow
Write-Host "========================================" -ForegroundColor Cyan

$AllResults | Format-Table -AutoSize

# # 保存汇总结果到CSV
# $CsvOutput = Join-Path $LogDir "all_datasets_summary.csv"
# $AllResults | Export-Csv -Path $CsvOutput -NoTypeInformation
# Write-Host "Summary saved to: $CsvOutput" -ForegroundColor Green

# # 可选：生成简单的Markdown报告
# $MdOutput = Join-Path $LogDir "all_datasets_summary.md"
# @"
# # Evaluation Results Summary

# | Dataset | Total Images | Avg Saving (%) | Avg PSNR (dB) | Avg SSIM | Avg MetaM |
# |---------|-------------|---------------|---------------|----------|-----------|
# $($AllResults | ForEach-Object { "| $($_.Dataset) | $($_.TotalImages) | $($_.AvgSaving.ToString('F2')) | $($_.AvgPSNR.ToString('F2')) | $($_.AvgSSIM.ToString('F4')) | $($_.AvgMetaM.ToString('F6')) |" })
# "@ | Out-File -FilePath $MdOutput -Encoding UTF8

# Write-Host "Markdown report saved to: $MdOutput" -ForegroundColor Green