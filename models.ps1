# Download pre-quantized MiniMax Music 3 GGUF models from HuggingFace
#
# Usage: .\models.ps1 [options]
#   default:    Q8_0 set (LM Q8 + depth Q8 + DiT Q8 + cond F32 + VAE F32)
#   --all:      all models, all quants
#   --quant X:  use quant X (Q4_K_M, Q5_K_M, Q6_K, Q8_0, BF16, F32)

param(
    [switch]$All,
    [string]$Quant = "Q8_0"
)

$ErrorActionPreference = "Stop"

$REPO = "Serveurperso/MiniMax-Music3-GGUF"
$DIR = "models"

# Resolve quant to best available for each component.
# Matches the quantize.sh matrix exactly:
#   LM:    BF16, Q5_K_M, Q6_K, Q8_0  (no Q4)
#   Depth: BF16, Q8_0
#   DiT:   F32, Q4_K_M, Q5_K_M, Q6_K, Q8_0
#   Cond and VAE ship as F32 only (small, quality-critical).
# If requested quant unavailable, picks the next larger available.
function Resolve-Quant {
    param([string]$Requested, [string]$Component)

    switch ($Component) {
        "lm" {
            switch ($Requested) {
                "BF16" { "BF16" }
                "F32"  { "BF16" }
                "Q6_K" { "Q6_K" }
                "Q5_K_M" { "Q5_K_M" }
                "Q4_K_M" { "Q5_K_M" }
                default { "Q8_0" }
            }
        }
        "depth" {
            switch ($Requested) {
                "BF16" { "BF16" }
                "F32"  { "BF16" }
                default { "Q8_0" }
            }
        }
        "dit" {
            switch ($Requested) {
                "BF16" { "F32" }
                "F32"  { "F32" }
                default { $Requested }
            }
        }
    }
}

function Download-Model {
    param([string]$File)

    $path = Join-Path $DIR $File
    if (Test-Path $path) {
        Write-Host "[OK] $File"
        return
    }
    Write-Host "[Download] $File"
    hf download --quiet $REPO $File --local-dir $DIR
}

New-Item -ItemType Directory -Force -Path $DIR | Out-Null

if ($All) {
    Download-Model "MiniMax-Music3-condition_encoder-F32.gguf"
    Download-Model "MiniMax-Music3-language_model-BF16.gguf"
    Download-Model "MiniMax-Music3-language_model-Q5_K_M.gguf"
    Download-Model "MiniMax-Music3-language_model-Q6_K.gguf"
    Download-Model "MiniMax-Music3-language_model-Q8_0.gguf"
    Download-Model "MiniMax-Music3-rvq_depth_decoder-BF16.gguf"
    Download-Model "MiniMax-Music3-rvq_depth_decoder-Q8_0.gguf"
    Download-Model "MiniMax-Music3-transformer-F32.gguf"
    Download-Model "MiniMax-Music3-transformer-Q4_K_M.gguf"
    Download-Model "MiniMax-Music3-transformer-Q5_K_M.gguf"
    Download-Model "MiniMax-Music3-transformer-Q6_K.gguf"
    Download-Model "MiniMax-Music3-transformer-Q8_0.gguf"
    Download-Model "MiniMax-Music3-vocoder-F32.gguf"
    exit 0
}

Download-Model "MiniMax-Music3-condition_encoder-F32.gguf"
Download-Model "MiniMax-Music3-vocoder-F32.gguf"
Download-Model "MiniMax-Music3-language_model-$(Resolve-Quant $Quant lm).gguf"
Download-Model "MiniMax-Music3-rvq_depth_decoder-$(Resolve-Quant $Quant depth).gguf"
Download-Model "MiniMax-Music3-transformer-$(Resolve-Quant $Quant dit).gguf"