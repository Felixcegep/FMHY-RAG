# Navigate to the folder containing this script (i.e., setup/)
Set-Location -Path $PSScriptRoot

# Create ../docs directory if it doesn't exist
$docsDir = Join-Path $PSScriptRoot "..\docs"
New-Item -ItemType Directory -Force -Path $docsDir | Out-Null

# Get list of Markdown file URLs from GitHub API
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
