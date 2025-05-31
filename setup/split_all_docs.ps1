# Navigate to the folder containing this script (i.e., setup/)
Set-Location -Path $PSScriptRoot

$InputDir = Join-Path $PSScriptRoot "..\docs"
$OutputDir = Join-Path $PSScriptRoot "..\sections"
New-Item -ItemType Directory -Force -Path $OutputDir | Out-Null

# Loop through all .md files
Get-ChildItem -Path $InputDir -Filter *.md | ForEach-Object {
    $file = $_.FullName
    Write-Host "🔍 Processing $file..."

    $currentFile = $null

    Get-Content $file | ForEach-Object {
        $line = $_
        if ($line -match "►\s*(.+)") {
            $rawTitle = $matches[1]
            $title = $rawTitle -replace "\[([^\]]+)\]\([^)]+\)", '$1'
            $filename = ($title -replace '\s', '_') -replace '[^a-zA-Z0-9_]', ''
            $currentFile = Join-Path $OutputDir "$filename.md"
            "# $title" | Out-File -FilePath $currentFile -Encoding UTF8
        }
        elseif ($null -ne $currentFile) {
            $line | Out-File -FilePath $currentFile -Encoding UTF8 -Append
        }
    }
}

Write-Host "✅ All sections saved in: $OutputDir"
