docker stop osrm 2>$null
docker rm osrm 2>$null

$currentPath = (Get-Location).Path + '\osrm-data'

docker run -d `
  --name osrm `
  -p 5000:5000 `
  -v "${currentPath}:/data" `
  osrm/osrm-backend `
  osrm-routed --algorithm mld --max-table-size 10000 /data/sudeste-260212.osrm

Start-Sleep -Seconds 3

$status = docker ps --filter "name=osrm" --format "{{.Status}}"
if ($status) {
    Write-Host "OSRM running on http://localhost:5000" -ForegroundColor Green
    
    Start-Sleep -Seconds 2
    $test = Invoke-RestMethod "http://localhost:5000/route/v1/driving/-46.6333,-23.5505;-46.6389,-23.5489" -ErrorAction SilentlyContinue
    if ($test) {
        Write-Host "Test successful! Distance: $([math]::Round($test.routes[0].distance/1000,2)) km" -ForegroundColor Green
    }
} else {
    Write-Host "Failed to start OSRM" -ForegroundColor Red
    docker logs osrm
}
