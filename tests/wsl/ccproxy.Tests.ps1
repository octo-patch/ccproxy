BeforeAll {
    . $PSScriptRoot/lib.ps1
    $script:Distro = $null

    if (-not $env:CCPROXY_WSL_ARTIFACT) {
        throw "CCPROXY_WSL_ARTIFACT must point at ccproxy.wsl"
    }

    Write-Host "> wsl.exe --update"
    & wsl.exe --update | Write-Host

    Write-Host "> wsl.exe --version"
    & wsl.exe --version | Write-Host
    if ($LASTEXITCODE -ne 0) {
        throw "Store WSL is required; wsl.exe --version failed"
    }

    $script:Distro = [CcproxyWslDistro]::new($env:CCPROXY_WSL_ARTIFACT)
}

AfterAll {
    if ($script:Distro) {
        $script:Distro.Uninstall()
    }
}

Describe "ccproxy.wsl" {
    It "runs the namespace inspector path in an imported WSL2 distro" {
        $distro = $script:Distro
        $daemonPid = $null

        try {
            $distro.Launch("ccproxy --help >/dev/null")

            $systemdCode = $distro.ExitCode("systemctl is-system-running --wait")
            if ($systemdCode -ne 0) {
                $distro.ExitCode("systemctl --failed --no-pager")
                $distro.ExitCode("journalctl -b -n 200 --no-pager")
            }
            $systemdCode | Should -Be 0

            $configDir = ($distro.Launch("mktemp -d /tmp/ccproxy-wsl.XXXXXX") | Select-Object -Last 1).Trim()
            $distro.Launch("CCPROXY_CONFIG_DIR=$configDir ccproxy init")

            $startCommand = 'CCPROXY_CONFIG_DIR={0} nohup ccproxy start >{0}/ccproxy.log 2>&1 & echo $!' -f $configDir
            $daemonPid = ($distro.Launch($startCommand) | Select-Object -Last 1).Trim()

            $ready = $false
            foreach ($i in 1..90) {
                if ($distro.ExitCode("CCPROXY_CONFIG_DIR=$configDir ccproxy status --proxy") -eq 0 -and
                    $distro.ExitCode("test -s $configDir/.inspector-wireguard-client.conf") -eq 0) {
                    $ready = $true
                    break
                }
                Start-Sleep -Seconds 1
            }

            if (-not $ready) {
                $distro.ExitCode("tail -200 $configDir/ccproxy.log")
            }
            $ready | Should -BeTrue

            $statusJson = $distro.Launch("CCPROXY_CONFIG_DIR=$configDir ccproxy namespace status --json") -join "`n"
            $status = $statusJson | ConvertFrom-Json
            $status.kernel.is_wsl | Should -BeTrue
            $status.tools.slirp4netns.present | Should -BeTrue
            $status.tools.wg.present | Should -BeTrue
            $status.tools.sysctl.present | Should -BeTrue
            $status.devices.dev_net_tun.present | Should -BeTrue

            $doctorJson = $distro.Launch("CCPROXY_CONFIG_DIR=$configDir ccproxy namespace doctor --json") -join "`n"
            $doctor = $doctorJson | ConvertFrom-Json
            @($doctor.failures).Count | Should -Be 0

            $distro.Launch("CCPROXY_CONFIG_DIR=$configDir ccproxy run --capture -- curl -fsS https://example.com -o /dev/null")
        }
        finally {
            if ($daemonPid) {
                $distro.ExitCode("kill $daemonPid >/dev/null 2>&1 || true")
            }
        }
    }
}
