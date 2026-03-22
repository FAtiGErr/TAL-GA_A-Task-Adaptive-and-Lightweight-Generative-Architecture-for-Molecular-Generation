param(
    [string]$RepoUrl = "https://github.com/FAtiGErr/TAL-GA_A-Task-Adaptive-and-Lightweight-Generative-Architecture-for-Molecular-Generation.git",
    [string]$Branch = "main",
    [string]$CommitMessage = "chore: sync minimal open-source snapshot",
    [switch]$NoPush
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

function Run-Git {
    param([Parameter(Mandatory = $true)][string[]]$Args)
    & git --no-pager @Args
    if ($LASTEXITCODE -ne 0) {
        throw "Git command failed: git $($Args -join ' ')"
    }
}

function Disable-GitProxy {
    Write-Host "[publish] Disabling git proxy (local/global)..."
    & git --no-pager config --unset http.proxy 2>$null | Out-Null
    & git --no-pager config --unset https.proxy 2>$null | Out-Null
    & git --no-pager config --global --unset http.proxy 2>$null | Out-Null
    & git --no-pager config --global --unset https.proxy 2>$null | Out-Null
}

function Is-AllowedPath {
    param([Parameter(Mandatory = $true)][string]$Path)

    $p = $Path.Replace('/', '\\')
    $lower = $p.ToLowerInvariant()
    $name = [System.IO.Path]::GetFileName($p).ToLowerInvariant()

    if ($lower.EndsWith(".npz")) { return $false }
    if ($lower -like "*\__pycache__\*" -or $lower.StartsWith("__pycache__\\")) { return $false }

    $blockedDirs = @(
        "chembl_31\\",
        "zinc\\",
        "jointcorpus\\",
        "pytdc\\data\\",
        "model\\logs\\",
        "embedding\\weights\\"
    )
    foreach ($dir in $blockedDirs) {
        if ($lower.StartsWith($dir)) { return $false }
    }

    if ($lower.StartsWith("results\\")) {
        if (-not $lower.StartsWith("results\\pso\\")) { return $false }
        if ($name -eq "evaluation_summary.csv") { return $true }
        if ($name -eq "moses_metrics.csv") { return $true }
        if ($name.StartsWith("moses_metrics_") -and $name.EndsWith(".csv")) { return $true }
        if ($lower.StartsWith("results\\pso\\reports\\") -and $lower.EndsWith(".csv")) { return $true }
        return $false
    }

    if ($lower -eq "readme.md" -or
        $lower -eq "requirements.txt" -or
        $lower -eq "environment.yml" -or
        $lower -eq ".gitignore" -or
        $lower -eq "publish_github.ps1") {
        return $true
    }

    if ($lower.StartsWith("prepare\\") -or $lower.StartsWith("optalgo\\")) { return $true }

    if ($lower.StartsWith("embedding\\")) {
        if ($lower.StartsWith("embedding\\weights\\")) { return $false }
        return $true
    }

    if ($lower.EndsWith(".py")) { return $true }

    return $false
}

Write-Host "[publish] Working directory: $PWD"
Disable-GitProxy

if (-not (Get-Command git -ErrorAction SilentlyContinue)) {
    throw "git is not available in PATH. Install Git for Windows or fix PATH first."
}

if (-not (Test-Path ".git")) {
    Write-Host "[publish] Git repo not found, initializing..."
    Run-Git -Args @("init")
    Run-Git -Args @("branch", "-M", $Branch)
}

$existingOrigin = ""
try {
    $existingOrigin = (& git remote get-url origin 2>$null).Trim()
} catch {
    $existingOrigin = ""
}

if ([string]::IsNullOrWhiteSpace($existingOrigin)) {
    Write-Host "[publish] Setting remote origin -> $RepoUrl"
    Run-Git -Args @("remote", "add", "origin", $RepoUrl)
} elseif ($existingOrigin -ne $RepoUrl) {
    throw "Remote origin mismatch. Current: $existingOrigin ; Expected: $RepoUrl"
}

Write-Host "[publish] Staging all changes..."
Run-Git -Args @("add", "-A")

$stagedFiles = (& git --no-pager diff --cached --name-only) | Where-Object { $_ -and $_.Trim() -ne "" }
$filteredOut = New-Object System.Collections.Generic.List[string]

foreach ($file in $stagedFiles) {
    if (-not (Is-AllowedPath -Path $file)) {
        & git --no-pager reset -q HEAD -- "$file"
        $filteredOut.Add($file)
    }
}

if ($filteredOut.Count -gt 0) {
    Write-Host "[publish] Filtered out files (not in minimal open-source list):"
    $filteredOut | ForEach-Object { Write-Host "  - $_" }
}

$remaining = (& git --no-pager diff --cached --name-only) | Where-Object { $_ -and $_.Trim() -ne "" }
if (-not $remaining -or $remaining.Count -eq 0) {
    Write-Host "[publish] No committable changes after filters. Nothing to upload."
    exit 0
}

Write-Host "[publish] Committing..."
Run-Git -Args @("commit", "-m", $CommitMessage)

if ($NoPush) {
    Write-Host "[publish] Commit created. Push skipped because -NoPush was set."
    exit 0
}

Write-Host "[publish] Fetching origin..."
& git --no-pager fetch origin --prune | Out-Null

$remoteBranchExists = (& git --no-pager ls-remote --heads origin $Branch)
if (-not [string]::IsNullOrWhiteSpace($remoteBranchExists)) {
    Write-Host "[publish] Rebasing local branch on origin/$Branch"
    & git --no-pager pull --rebase origin $Branch
    if ($LASTEXITCODE -ne 0) {
        throw "git pull --rebase failed; please resolve conflicts first."
    }
}

Write-Host "[publish] Pushing to origin/$Branch ..."
& git --no-pager push -u origin $Branch
if ($LASTEXITCODE -ne 0) {
    throw "git push failed"
}

Write-Host "[publish] Done."

