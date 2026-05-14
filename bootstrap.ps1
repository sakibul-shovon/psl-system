<#
.SYNOPSIS
    One-command bootstrap for the PSL Document Intelligence stack.

.DESCRIPTION
    This script:
      1. Verifies Docker Desktop is running
      2. Checks that .env exists with real API keys
      3. Builds and starts the full stack (Qdrant + API + UI) via Docker Compose
      4. Waits for the API container to become healthy (ML models take ~60-90s to load)
      5. Runs the seed script inside the api container to ingest example documents
         and create starter patterns for the demo
      6. Prints the URLs to open in your browser

    Prerequisites:
      - Docker Desktop installed and running
      - .env file in this directory with GEMINI_API_KEY and GROQ_API_KEY filled in
        (copy from .env.example and replace the placeholder values)

.EXAMPLE
    .\bootstrap.ps1

.EXAMPLE
    .\bootstrap.ps1 -SkipSeed    # Start stack without running the seed script
#>

param(
    [switch]$SkipSeed,
    [int]$HealthTimeoutSeconds = 180
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

# ── Colour helpers ─────────────────────────────────────────────────────────────
function Write-Step  { param($Msg) Write-Host "`n[PSL] $Msg" -ForegroundColor Cyan }
function Write-Ok    { param($Msg) Write-Host "  ✓  $Msg" -ForegroundColor Green }
function Write-Warn  { param($Msg) Write-Host "  ⚠  $Msg" -ForegroundColor Yellow }
function Write-Fail  { param($Msg) Write-Host "`n  ✗  $Msg" -ForegroundColor Red; exit 1 }

Write-Host ""
Write-Host "╔══════════════════════════════════════════════════════════════╗" -ForegroundColor Magenta
Write-Host "║       Pearson Specter Litt — Document Intelligence           ║" -ForegroundColor Magenta
Write-Host "║                  Bootstrap Script v1.0                       ║" -ForegroundColor Magenta
Write-Host "╚══════════════════════════════════════════════════════════════╝" -ForegroundColor Magenta
Write-Host ""

# ── Step 1: Check Docker is running ──────────────────────────────────────────
Write-Step "1/5  Checking Docker..."
try {
    $dockerInfo = docker info 2>&1
    if ($LASTEXITCODE -ne 0) { throw "docker info returned exit code $LASTEXITCODE" }
    Write-Ok "Docker is running."
} catch {
    Write-Fail "Docker is not running. Please start Docker Desktop and retry.`n  Error: $_"
}

# ── Step 2: Check .env exists and has real keys ───────────────────────────────
Write-Step "2/5  Checking .env..."

$envPath = Join-Path $PSScriptRoot ".env"
if (-not (Test-Path $envPath)) {
    Write-Fail ".env not found.`n  Run: Copy-Item .env.example .env`n  Then fill in GEMINI_API_KEY and GROQ_API_KEY."
}

$envContent = Get-Content $envPath -Raw

if ($envContent -match "your_gemini_api_key_here" -or $envContent -match "your_groq_api_key_here") {
    Write-Fail ".env still contains placeholder values.`n  Edit .env and replace GEMINI_API_KEY and GROQ_API_KEY with real keys."
}

if ($envContent -notmatch "GEMINI_API_KEY=\S+") {
    Write-Warn "GEMINI_API_KEY appears to be empty — draft generation will fail."
}

if ($envContent -notmatch "GROQ_API_KEY=\S+") {
    Write-Warn "GROQ_API_KEY appears to be empty — judge scoring will be disabled."
}

Write-Ok ".env found with API keys."

# ── Step 3: Build and start containers ───────────────────────────────────────
Write-Step "3/5  Building and starting containers (this may take a few minutes on first run)..."
Write-Host "     Building: api (FastAPI + Tesseract) and ui (Streamlit)" -ForegroundColor Gray
Write-Host "     NOTE: First run downloads ~1.5 GB of ML models into a Docker volume." -ForegroundColor Gray
Write-Host "           Subsequent runs start in seconds from the cached volume." -ForegroundColor Gray

Set-Location $PSScriptRoot
docker compose up --build -d

if ($LASTEXITCODE -ne 0) {
    Write-Fail "docker compose up failed. Check the output above for errors."
}

Write-Ok "Containers started (running in background)."

# ── Step 4: Wait for API to be healthy ────────────────────────────────────────
Write-Step "4/5  Waiting for API to be ready (ML models loading — up to ${HealthTimeoutSeconds}s)..."

$apiUrl    = "http://localhost:8000/health"
$elapsed   = 0
$interval  = 5
$dots      = 0
$ready     = $false

while ($elapsed -lt $HealthTimeoutSeconds) {
    try {
        $response = Invoke-RestMethod -Uri $apiUrl -Method Get -TimeoutSec 5
        if ($response.status -eq "ok") {
            $ready = $true
            break
        }
    } catch {
        # Not ready yet — continue polling
    }

    $dots++
    Write-Host ("  " + ("." * $dots) + " ${elapsed}s / ${HealthTimeoutSeconds}s") -ForegroundColor Gray -NoNewline
    Write-Host "`r" -NoNewline
    Start-Sleep -Seconds $interval
    $elapsed += $interval
}

if (-not $ready) {
    Write-Host ""
    Write-Warn "API did not become healthy within ${HealthTimeoutSeconds}s."
    Write-Warn "The containers are still running. Check logs with:"
    Write-Warn "  docker compose logs api --tail=50"
    Write-Host ""
    Write-Host "  Open http://localhost:8000/health in your browser to check manually." -ForegroundColor Yellow
    exit 1
}

Write-Host ""
Write-Ok "API is ready at http://localhost:8000"

# ── Step 5: Run seed script ────────────────────────────────────────────────────
if (-not $SkipSeed) {
    Write-Step "5/5  Seeding example documents and patterns..."
    Write-Host "     Running scripts/seed.py inside the api container..." -ForegroundColor Gray

    docker compose exec api python -m scripts.seed

    if ($LASTEXITCODE -ne 0) {
        Write-Warn "Seed script encountered an error (see output above)."
        Write-Warn "The stack is still running — you can seed manually:"
        Write-Warn "  docker compose exec api python -m scripts.seed"
    } else {
        Write-Ok "Seed complete — example document ingested and starter patterns created."
    }
} else {
    Write-Step "5/5  Skipping seed (--SkipSeed flag set)."
    Write-Ok  "You can seed later with: docker compose exec api python -m scripts.seed"
}

# ── Done ───────────────────────────────────────────────────────────────────────
Write-Host ""
Write-Host "╔══════════════════════════════════════════════════════════════╗" -ForegroundColor Green
Write-Host "║                    PSL stack is live!                        ║" -ForegroundColor Green
Write-Host "╠══════════════════════════════════════════════════════════════╣" -ForegroundColor Green
Write-Host "║  UI (Streamlit)      http://localhost:8501                   ║" -ForegroundColor Green
Write-Host "║  API (FastAPI)       http://localhost:8000                   ║" -ForegroundColor Green
Write-Host "║  API docs (Swagger)  http://localhost:8000/docs              ║" -ForegroundColor Green
Write-Host "║  Qdrant dashboard    http://localhost:6333/dashboard         ║" -ForegroundColor Green
Write-Host "╠══════════════════════════════════════════════════════════════╣" -ForegroundColor Green
Write-Host "║  Stop stack:   docker compose down                           ║" -ForegroundColor Green
Write-Host "║  View logs:    docker compose logs -f                        ║" -ForegroundColor Green
Write-Host "║  Wipe data:    docker compose down -v                        ║" -ForegroundColor Green
Write-Host "╚══════════════════════════════════════════════════════════════╝" -ForegroundColor Green
Write-Host ""
