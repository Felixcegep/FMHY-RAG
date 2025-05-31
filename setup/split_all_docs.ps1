# split_all_docs.ps1 (run from setup\ but uses ../docs and ../data)

$projectRoot = Split-Path -Path $PSScriptRoot -Parent
$inputDir = Join-Path $projectRoot "docs"
$outputDir = Join-Path $projectRoot "data"
New-Item -ItemType Directory -Force -Path $outputDir | Out-Null

Get-ChildItem -Path $inputDir -Filter *.md | ForEach-Object {
    $lines = Get-Content $_.FullName
    $currentFile = $null

    foreach ($line in $lines) {
        if ($line -match "►\s*(.+)") {
            $rawTitle = $matches[1]
            $title = $rawTitle -replace '\[([^\]]+)\]\([^)]+\)', '$1'
            $safeName = ($title -replace ' ', '_' -replace '[^\w_]', '') + ".md"
            $currentFile = Join-Path $outputDir $safeName
            "# $title" | Out-File -FilePath $currentFile -Encoding utf8
        } elseif ($currentFile) {
            $line | Out-File -FilePath $currentFile -Append -Encoding utf8
        }
    }
}

Write-Host "✅ All data saved in: $outputDir"
