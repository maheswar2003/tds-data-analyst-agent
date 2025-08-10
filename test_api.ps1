# PowerShell script to test the TDS Data Analyst Agent API
Write-Host "🚀 Testing TDS Data Analyst Agent API" -ForegroundColor Green
Write-Host "=" * 50

# Test 1: Health Check
Write-Host "`n🏥 Testing Health Endpoint..." -ForegroundColor Yellow
try {
    $healthResponse = Invoke-RestMethod -Uri "http://127.0.0.1:8000/health" -Method GET
    Write-Host "✅ Health Check: Success" -ForegroundColor Green
    Write-Host "Response: $($healthResponse | ConvertTo-Json)" -ForegroundColor Gray
} catch {
    Write-Host "❌ Health Check Failed: $_" -ForegroundColor Red
    Write-Host "Make sure the server is running: python start_server.py" -ForegroundColor Yellow
    exit 1
}

# Test 2: Root Endpoint
Write-Host "`n🏠 Testing Root Endpoint..." -ForegroundColor Yellow
try {
    $rootResponse = Invoke-RestMethod -Uri "http://127.0.0.1:8000/" -Method GET
    Write-Host "✅ Root Endpoint: Success" -ForegroundColor Green
    Write-Host "Available endpoints:" -ForegroundColor Gray
    $rootResponse.endpoints.PSObject.Properties | ForEach-Object {
        Write-Host "  - $($_.Name): $($_.Value)" -ForegroundColor Cyan
    }
} catch {
    Write-Host "❌ Root Endpoint Failed: $_" -ForegroundColor Red
}

# Test 3: File Upload (Wikipedia Question)
Write-Host "`n📤 Testing File Upload Endpoint..." -ForegroundColor Yellow
try {
    if (Test-Path "test_questions\wikipedia_question.txt") {
        Write-Host "Uploading Wikipedia question..." -ForegroundColor Gray
        
        # Create multipart form data for file upload
        $filePath = "test_questions\wikipedia_question.txt"
        $fileBytes = [System.IO.File]::ReadAllBytes($filePath)
        $fileName = [System.IO.Path]::GetFileName($filePath)
        
        # Use curl-style upload (PowerShell 7+ compatible)
        if ($PSVersionTable.PSVersion.Major -ge 7) {
            $uploadResponse = Invoke-RestMethod -Uri "http://127.0.0.1:8000/api/" -Method POST -Form @{
                question = Get-Item $filePath
            }
        } else {
            # Fallback for older PowerShell
            Write-Host "⚠️  PowerShell version < 7 detected. Using basic test..." -ForegroundColor Yellow
            Write-Host "For full file upload testing, please use: python test_server.py" -ForegroundColor Yellow
            exit 0
        }
        
        Write-Host "✅ File Upload: Success" -ForegroundColor Green
        
        if ($uploadResponse.summary) {
            Write-Host "📊 Analysis Summary: $($uploadResponse.summary)" -ForegroundColor Cyan
        }
        
        if ($uploadResponse.status) {
            Write-Host "📈 Status: $($uploadResponse.status)" -ForegroundColor Cyan
        }
        
        if ($uploadResponse.data) {
            Write-Host "📁 Data Keys: $($uploadResponse.data.PSObject.Properties.Name -join ', ')" -ForegroundColor Cyan
        }
        
    } else {
        Write-Host "❌ Test file not found: test_questions\wikipedia_question.txt" -ForegroundColor Red
    }
} catch {
    Write-Host "❌ File Upload Failed: $_" -ForegroundColor Red
    if ($_.Exception.Message -like "*API*key*") {
        Write-Host "💡 Hint: Set your OpenAI API key with: `$env:OPENAI_API_KEY='your-key-here'" -ForegroundColor Yellow
    }
}

Write-Host "`n" + "=" * 50
Write-Host "✅ API Testing Complete!" -ForegroundColor Green
Write-Host "💡 For interactive testing, visit: http://127.0.0.1:8000/docs" -ForegroundColor Yellow