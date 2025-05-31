# download_docs.ps1 (run from setup\ but acts on ../docs)

$projectRoot = Split-Path -Path $PSScriptRoot -Parent
$docsDir = Join-Path $projectRoot "docs"
New-Item -ItemType Directory -Force -Path $docsDir | Out-Null

$apiUrl = "https://api.github.com/repos/fmhy/edit/contents/docs"
$response = Invoke-RestMethod -Uri $apiUrl

foreach ($file in $response) {
    if ($file.name -like "*.md") {
        $filename = $file.name
        $url = $file.download_url
        $destPath = Join-Path $docsDir $filename
        Invoke-WebRequest -Uri $url -OutFile $destPath
        Write-Host "✅ Downloaded $filename"
    }
}
