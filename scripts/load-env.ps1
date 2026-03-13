<#
  Load repository runtime environment variables from .env or src/env.json.

  Usage:
    . .\scripts\load-env.ps1
    . .\scripts\load-env.ps1 -SourcePath .env
    . .\scripts\load-env.ps1 -SourcePath src\env.json
    & .\scripts\load-env.ps1

  Notes:
  - The script prefers .env by default, then falls back to src/env.json.
  - Supported .env format: KEY=value, optional leading "export ", blank lines, and # comments.
  - Supported JSON format: { "Variables": { "KEY": "value" } }
  - Do not use "powershell -File" if you want variables to stay in the current shell session.
#>

param(
  [string]$SourcePath = ""
)

$ErrorActionPreference = "Stop"

$RepoRoot = Resolve-Path (Join-Path $PSScriptRoot "..")
Set-Location $RepoRoot

function Resolve-EnvSourcePath {
  param([string]$RequestedPath)

  if ($RequestedPath) {
    $candidate = $RequestedPath
    if (-not [System.IO.Path]::IsPathRooted($candidate)) {
      $candidate = Join-Path $RepoRoot $candidate
    }
    if (-not (Test-Path $candidate)) {
      throw "Environment source not found: $RequestedPath"
    }
    return (Resolve-Path $candidate).Path
  }

  $defaultCandidates = @(
    (Join-Path $RepoRoot ".env"),
    (Join-Path $RepoRoot "src\env.json")
  )

  foreach ($candidate in $defaultCandidates) {
    if (Test-Path $candidate) {
      return (Resolve-Path $candidate).Path
    }
  }

  throw "No environment source found. Expected .env or src\env.json under $RepoRoot"
}

function Import-DotEnvFile {
  param([string]$Path)

  $loaded = New-Object System.Collections.Generic.List[string]

  foreach ($rawLine in Get-Content $Path) {
    $line = $rawLine.Trim()
    if (-not $line -or $line.StartsWith("#")) {
      continue
    }

    if ($line.StartsWith("export ")) {
      $line = $line.Substring(7).Trim()
    }

    $parts = $line -split "=", 2
    if ($parts.Count -ne 2) {
      throw "Invalid .env line in ${Path}: $rawLine"
    }

    $name = $parts[0].Trim()
    $value = $parts[1].Trim()

    if (($value.StartsWith('"') -and $value.EndsWith('"')) -or ($value.StartsWith("'") -and $value.EndsWith("'"))) {
      $value = $value.Substring(1, $value.Length - 2)
    }

    [Environment]::SetEnvironmentVariable($name, $value, "Process")
    $loaded.Add($name) | Out-Null
  }

  return $loaded
}

function Import-EnvJsonFile {
  param([string]$Path)

  $doc = Get-Content $Path -Raw | ConvertFrom-Json
  if (-not $doc.Variables) {
    throw "Expected JSON object with top-level 'Variables' property in $Path"
  }

  $loaded = New-Object System.Collections.Generic.List[string]
  foreach ($property in $doc.Variables.PSObject.Properties) {
    [Environment]::SetEnvironmentVariable($property.Name, [string]$property.Value, "Process")
    $loaded.Add($property.Name) | Out-Null
  }

  return $loaded
}

$resolvedSourcePath = Resolve-EnvSourcePath -RequestedPath $SourcePath
$extension = [System.IO.Path]::GetExtension($resolvedSourcePath).ToLowerInvariant()

if ($extension -eq ".json") {
  $loadedKeys = Import-EnvJsonFile -Path $resolvedSourcePath
}
else {
  $loadedKeys = Import-DotEnvFile -Path $resolvedSourcePath
}

$loadedKeys = $loadedKeys | Sort-Object -Unique

Write-Host "Loaded $($loadedKeys.Count) environment variables from $resolvedSourcePath"
Write-Host ($loadedKeys -join ", ")

