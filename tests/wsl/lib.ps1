if ($PSVersionTable.PSEdition -ne "Core") {
    throw "The tests require PowerShell Core."
}

if ($IsWindows -eq $false) {
    throw "The tests require real Windows with WSL2."
}

function Remove-Escapes {
    param(
        [parameter(ValueFromPipeline = $true)]
        [string[]]$InputObject
    )

    process {
        $InputObject | ForEach-Object {
            $_ -replace '\x1b(\[(\?..|.)|.)', ''
        }
    }
}

class CcproxyWslDistro {
    [string]$Id
    [string]$TempDir

    CcproxyWslDistro([string]$Artifact) {
        $this.Id = (New-Guid).ToString()
        $this.TempDir = Join-Path ([System.IO.Path]::GetTempPath()) $this.Id
        New-Item -ItemType Directory -Path $this.TempDir | Out-Null

        Write-Host "> wsl.exe --import $($this.Id) $($this.TempDir) $Artifact --version 2"
        & wsl.exe --import $this.Id $this.TempDir $Artifact --version 2 | Write-Host
        if ($LASTEXITCODE -ne 0) {
            throw "Failed to import distro"
        }

        $distros = @(& wsl.exe --list -q)
        if ($distros -notcontains $this.Id) {
            throw "Imported distro $($this.Id) was not listed by wsl.exe"
        }
    }

    [Array]Launch([string]$Command) {
        Write-Host "> $Command"
        $result = & wsl.exe -d $this.Id -- bash -lc $Command 2>&1
        $code = $LASTEXITCODE
        $clean = @($result | Remove-Escapes)
        $clean | Write-Host
        if ($code -ne 0) {
            throw "Command failed with exit code $code"
        }
        return $clean
    }

    [int]ExitCode([string]$Command) {
        Write-Host "> $Command"
        $result = & wsl.exe -d $this.Id -- bash -lc $Command 2>&1
        $code = $LASTEXITCODE
        @($result | Remove-Escapes) | Write-Host
        return $code
    }

    [void]Terminate() {
        Write-Host "> wsl.exe -t $($this.Id)"
        & wsl.exe -t $this.Id | Write-Host
    }

    [void]Uninstall() {
        Write-Host "> wsl.exe --unregister $($this.Id)"
        & wsl.exe --unregister $this.Id | Write-Host
        if ($LASTEXITCODE -ne 0) {
            throw "Failed to unregister distro"
        }

        if (Test-Path $this.TempDir) {
            Remove-Item $this.TempDir -Recurse -Force
        }
    }
}
