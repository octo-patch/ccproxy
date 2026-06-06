#!/usr/bin/env bash
set -euo pipefail

artifact="${1:-tmp/ccproxy-wsl-smoke/ccproxy.wsl}"
root="${CCPROXY_WSL_KVM_DIR:-tmp/wsl-kvm-smoke}"
downloads_dir="${XDG_DOWNLOAD_DIR:-$HOME/downloads}"
iso_url="${WIN11_ISO_URL:-}"
win_iso="${WIN11_ISO:-$downloads_dir/Win11_25H2_English_x64_v2.iso}"
disk="${WIN11_DISK:-$root/windows.qcow2}"
answer_iso="$root/autounattend.iso"
payload_iso="$root/payload.iso"
share="$root/share"
answer_dir="$root/autounattend"
payload_dir="$root/payload"
vars="$root/OVMF_VARS.fd"
monitor="$root/qemu.monitor"
qemu_log="$root/qemu.log"
result="$share/result.json"
share_marker="ccproxy-wsl-smoke-share.txt"
collector_port_file="$root/collector-port.txt"
collector_log="$root/collector.log"
timeout_seconds="${CCPROXY_WSL_KVM_TIMEOUT_SECONDS:-14400}"
disk_size="${CCPROXY_WSL_KVM_DISK_SIZE:-96G}"
reuse_disk="${CCPROXY_WSL_KVM_REUSE_DISK:-0}"
memory="${CCPROXY_WSL_KVM_MEMORY:-16G}"
cpus="${CCPROXY_WSL_KVM_CPUS:-8}"
vnc_display="${CCPROXY_WSL_KVM_VNC:-127.0.0.1:9}"
qemu_pid=""
collector_pid=""

cleanup() {
  if [[ -n "$qemu_pid" ]] && kill -0 "$qemu_pid" >/dev/null 2>&1; then
    kill "$qemu_pid" >/dev/null 2>&1 || true
    wait "$qemu_pid" >/dev/null 2>&1 || true
  fi
  if [[ -n "$collector_pid" ]] && kill -0 "$collector_pid" >/dev/null 2>&1; then
    kill "$collector_pid" >/dev/null 2>&1 || true
    wait "$collector_pid" >/dev/null 2>&1 || true
  fi
}
trap cleanup EXIT

if [[ ! -f "$artifact" ]]; then
  echo "ERROR: WSL artifact not found: $artifact" >&2
  exit 1
fi

if [[ ! -r "${OVMF_CODE:?OVMF_CODE must be set by the flake wrapper}" ]]; then
  echo "ERROR: OVMF_CODE is not readable: $OVMF_CODE" >&2
  exit 1
fi

if [[ ! -r "${OVMF_VARS_TEMPLATE:?OVMF_VARS_TEMPLATE must be set by the flake wrapper}" ]]; then
  echo "ERROR: OVMF_VARS_TEMPLATE is not readable: $OVMF_VARS_TEMPLATE" >&2
  exit 1
fi

if [[ ! -e /dev/kvm ]]; then
  echo "ERROR: /dev/kvm is required for the Windows WSL2 smoke VM" >&2
  exit 1
fi

is_iso_image() {
  local image="$1"
  local size
  [[ -f "$image" ]] || return 1
  size="$(stat -c %s "$image")"
  (( size > 1000000000 )) || return 1
  xorriso -indev "$image" -toc >/dev/null 2>&1
}

start_collector() {
  rm -f "$collector_port_file"
  : > "$collector_log"
  python3 -u - "$share" "$collector_port_file" >>"$collector_log" 2>&1 <<'PY' &
import http.server
import pathlib
import sys

out_dir = pathlib.Path(sys.argv[1])
port_file = pathlib.Path(sys.argv[2])
out_dir.mkdir(parents=True, exist_ok=True)

class Handler(http.server.BaseHTTPRequestHandler):
    def log_message(self, format, *args):
        return

    def do_GET(self):
        if self.path == "/health":
            body = b"ok\n"
            self.send_response(200)
            self.send_header("Content-Type", "text/plain")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return
        self.send_error(404)

    def do_POST(self):
        targets = {
            "/result": "result.json",
            "/bootstrap-log": "bootstrap.log",
            "/stage": "bootstrap-stage.txt",
        }
        target = targets.get(self.path)
        if target is None:
            self.send_error(404)
            return
        length = int(self.headers.get("Content-Length", "0"))
        body = self.rfile.read(length)
        (out_dir / target).write_bytes(body)
        self.send_response(204)
        self.end_headers()

server = http.server.ThreadingHTTPServer(("0.0.0.0", 0), Handler)
port_file.write_text(f"{server.server_port}\n", encoding="ascii")
server.serve_forever()
PY
  collector_pid="$!"
  for _ in $(seq 1 100); do
    [[ -s "$collector_port_file" ]] && break
    sleep 0.1
  done
  if [[ ! -s "$collector_port_file" ]]; then
    echo "ERROR: collector did not start; see $collector_log" >&2
    exit 1
  fi
}

mkdir -p "$root" "$share" "$answer_dir" "$payload_dir" "$downloads_dir"
: > "$qemu_log"
rm -f "$root/serial.log" "$collector_log"

if ! is_iso_image "$win_iso" && [[ -n "$iso_url" ]]; then
  echo "[wsl-kvm] Downloading Windows 11 Enterprise evaluation ISO"
  rm -f "$win_iso"
  curl --fail --location --continue-at - --output "$win_iso" "$iso_url"
fi

if ! is_iso_image "$win_iso"; then
  cat >&2 <<EOF
ERROR: Windows ISO is missing or invalid: $win_iso

Set WIN11_ISO to a locally downloaded official Windows 11 x64 ISO, for example:

  WIN11_ISO=$downloads_dir/Win11_25H2_English_x64_v2.iso just test-wsl-kvm $artifact

The Microsoft Evaluation Center fwlink currently returns an HTML landing page
to headless curl, so the harness refuses to boot it as installation media.
EOF
  exit 1
fi

if [[ "$reuse_disk" != "1" ]]; then
  rm -f "$disk" "$vars"
fi

if [[ ! -f "$disk" ]]; then
  echo "[wsl-kvm] Creating Windows disk: $disk"
  qemu-img create -f qcow2 "$disk" "$disk_size"
fi

if [[ ! -f "$vars" ]]; then
  cp "$OVMF_VARS_TEMPLATE" "$vars"
  chmod 0644 "$vars"
fi

echo "[wsl-kvm] Preparing host-visible share"
printf 'ccproxy WSL smoke result share\n' > "$share/$share_marker"
rm -f "$result" "$share/bootstrap.log" "$share/bootstrap-stage.txt" "$share/ccproxy.wsl" "$share/ccproxy-wsl-smoke-write-test.txt"

echo "[wsl-kvm] Starting result collector"
start_collector
collector_port="$(cat "$collector_port_file")"

echo "[wsl-kvm] Preparing payload ISO"
rm -rf "$payload_dir"
mkdir -p "$payload_dir"
cp "$artifact" "$payload_dir/ccproxy.wsl"
xorriso -as mkisofs -quiet -iso-level 4 -volid CCPROXYWSL -o "$payload_iso" "$payload_dir"
printf 'http://10.0.2.2:%s\n' "$collector_port" > "$answer_dir/CollectorUrl.txt"

mkdir -p "$answer_dir/\$OEM\$/\$\$/Setup/Scripts"

cat > "$answer_dir/autounattend.xml" <<'XML'
<?xml version="1.0" encoding="utf-8"?>
<unattend xmlns="urn:schemas-microsoft-com:unattend"
  xmlns:wcm="http://schemas.microsoft.com/WMIConfig/2002/State"
  xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance">
  <settings pass="windowsPE">
    <component name="Microsoft-Windows-International-Core-WinPE" processorArchitecture="amd64"
      publicKeyToken="31bf3856ad364e35" language="neutral" versionScope="nonSxS">
      <SetupUILanguage>
        <UILanguage>en-US</UILanguage>
      </SetupUILanguage>
      <InputLocale>en-US</InputLocale>
      <SystemLocale>en-US</SystemLocale>
      <UILanguage>en-US</UILanguage>
      <UserLocale>en-US</UserLocale>
    </component>
    <component name="Microsoft-Windows-Setup" processorArchitecture="amd64"
      publicKeyToken="31bf3856ad364e35" language="neutral" versionScope="nonSxS">
      <RunSynchronous>
        <RunSynchronousCommand wcm:action="add">
          <Order>1</Order>
          <Path>reg add HKLM\SYSTEM\Setup\LabConfig /v BypassTPMCheck /t REG_DWORD /d 1 /f</Path>
        </RunSynchronousCommand>
        <RunSynchronousCommand wcm:action="add">
          <Order>2</Order>
          <Path>reg add HKLM\SYSTEM\Setup\LabConfig /v BypassSecureBootCheck /t REG_DWORD /d 1 /f</Path>
        </RunSynchronousCommand>
        <RunSynchronousCommand wcm:action="add">
          <Order>3</Order>
          <Path>reg add HKLM\SYSTEM\Setup\LabConfig /v BypassRAMCheck /t REG_DWORD /d 1 /f</Path>
        </RunSynchronousCommand>
      </RunSynchronous>
      <DiskConfiguration>
        <Disk wcm:action="add">
          <DiskID>0</DiskID>
          <WillWipeDisk>true</WillWipeDisk>
          <CreatePartitions>
            <CreatePartition wcm:action="add">
              <Order>1</Order>
              <Type>EFI</Type>
              <Size>100</Size>
            </CreatePartition>
            <CreatePartition wcm:action="add">
              <Order>2</Order>
              <Type>MSR</Type>
              <Size>16</Size>
            </CreatePartition>
            <CreatePartition wcm:action="add">
              <Order>3</Order>
              <Type>Primary</Type>
              <Extend>true</Extend>
            </CreatePartition>
          </CreatePartitions>
          <ModifyPartitions>
            <ModifyPartition wcm:action="add">
              <Order>1</Order>
              <PartitionID>1</PartitionID>
              <Format>FAT32</Format>
              <Label>System</Label>
            </ModifyPartition>
            <ModifyPartition wcm:action="add">
              <Order>2</Order>
              <PartitionID>3</PartitionID>
              <Format>NTFS</Format>
              <Label>Windows</Label>
              <Letter>C</Letter>
            </ModifyPartition>
          </ModifyPartitions>
        </Disk>
        <WillShowUI>OnError</WillShowUI>
      </DiskConfiguration>
      <ImageInstall>
        <OSImage>
          <InstallFrom>
            <MetaData wcm:action="add">
              <Key>/IMAGE/INDEX</Key>
              <Value>1</Value>
            </MetaData>
          </InstallFrom>
          <InstallTo>
            <DiskID>0</DiskID>
            <PartitionID>3</PartitionID>
          </InstallTo>
          <WillShowUI>OnError</WillShowUI>
        </OSImage>
      </ImageInstall>
      <UserData>
        <AcceptEula>true</AcceptEula>
        <FullName>ccproxy</FullName>
        <Organization>ccproxy</Organization>
        <ProductKey>
          <WillShowUI>Never</WillShowUI>
        </ProductKey>
      </UserData>
    </component>
  </settings>
  <settings pass="specialize">
    <component name="Microsoft-Windows-Shell-Setup" processorArchitecture="amd64"
      publicKeyToken="31bf3856ad364e35" language="neutral" versionScope="nonSxS">
      <ComputerName>CCPROXY-WSL</ComputerName>
      <TimeZone>UTC</TimeZone>
    </component>
    <component name="Microsoft-Windows-Deployment" processorArchitecture="amd64"
      publicKeyToken="31bf3856ad364e35" language="neutral" versionScope="nonSxS">
      <RunSynchronous>
        <RunSynchronousCommand wcm:action="add">
          <Order>1</Order>
          <Path>cmd.exe /c for %D in (D E F G H I J K L M N O P Q R S T U V W X Y Z) do if exist %D:\Bootstrap.cmd call %D:\Bootstrap.cmd</Path>
        </RunSynchronousCommand>
      </RunSynchronous>
    </component>
  </settings>
  <settings pass="oobeSystem">
    <component name="Microsoft-Windows-International-Core" processorArchitecture="amd64"
      publicKeyToken="31bf3856ad364e35" language="neutral" versionScope="nonSxS">
      <InputLocale>en-US</InputLocale>
      <SystemLocale>en-US</SystemLocale>
      <UILanguage>en-US</UILanguage>
      <UserLocale>en-US</UserLocale>
    </component>
    <component name="Microsoft-Windows-Shell-Setup" processorArchitecture="amd64"
      publicKeyToken="31bf3856ad364e35" language="neutral" versionScope="nonSxS">
      <OOBE>
        <HideEULAPage>true</HideEULAPage>
        <HideLocalAccountScreen>true</HideLocalAccountScreen>
        <HideOEMRegistrationScreen>true</HideOEMRegistrationScreen>
        <HideOnlineAccountScreens>true</HideOnlineAccountScreens>
        <HideWirelessSetupInOOBE>true</HideWirelessSetupInOOBE>
        <NetworkLocation>Work</NetworkLocation>
        <ProtectYourPC>3</ProtectYourPC>
      </OOBE>
      <UserAccounts>
        <LocalAccounts>
          <LocalAccount wcm:action="add">
            <Name>ccproxy</Name>
            <DisplayName>ccproxy</DisplayName>
            <Group>Administrators</Group>
            <Password>
              <Value>ccproxy</Value>
              <PlainText>true</PlainText>
            </Password>
          </LocalAccount>
        </LocalAccounts>
      </UserAccounts>
      <AutoLogon>
        <Username>ccproxy</Username>
        <Enabled>true</Enabled>
        <LogonCount>999</LogonCount>
        <Password>
          <Value>ccproxy</Value>
          <PlainText>true</PlainText>
        </Password>
      </AutoLogon>
      <FirstLogonCommands>
        <SynchronousCommand wcm:action="add">
          <Order>1</Order>
          <CommandLine>cmd.exe /c for %D in (D E F G H I J K L M N O P Q R S T U V W X Y Z) do if exist %D:\Bootstrap.cmd call %D:\Bootstrap.cmd</CommandLine>
        </SynchronousCommand>
      </FirstLogonCommands>
    </component>
  </settings>
</unattend>
XML

cat > "$answer_dir/Bootstrap.cmd" <<'CMD'
@echo off
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0Bootstrap.ps1"
CMD

cat > "$answer_dir/Bootstrap.ps1" <<'POWERSHELL'
$ErrorActionPreference = "Stop"
$stateDir = "C:\ccproxy-wsl-smoke"
$stagePath = Join-Path $stateDir "stage.txt"
$logPath = Join-Path $stateDir "bootstrap.log"
$bootstrapPath = Join-Path $stateDir "Bootstrap.ps1"
New-Item -ItemType Directory -Force -Path $stateDir | Out-Null
if ($PSCommandPath -and ($PSCommandPath -ne $bootstrapPath)) {
    Copy-Item -Path $PSCommandPath -Destination $bootstrapPath -Force
}
Start-Transcript -Path $logPath -Append | Out-Null

function Set-BootstrapRunKey {
    $command = "powershell.exe -NoProfile -ExecutionPolicy Bypass -File `"$bootstrapPath`""
    New-Item -Path "HKLM:\Software\Microsoft\Windows\CurrentVersion\Run" -Force | Out-Null
    New-ItemProperty -Path "HKLM:\Software\Microsoft\Windows\CurrentVersion\Run" -Name "CcproxyWslSmoke" -Value $command -PropertyType String -Force | Out-Null
}

function Remove-BootstrapRunKey {
    Remove-ItemProperty -Path "HKLM:\Software\Microsoft\Windows\CurrentVersion\Run" -Name "CcproxyWslSmoke" -ErrorAction SilentlyContinue
}

function Get-SmokeArtifactPath {
    for ($i = 0; $i -lt 180; $i++) {
        foreach ($drive in Get-PSDrive -PSProvider FileSystem) {
            $candidate = Join-Path $drive.Root "ccproxy.wsl"
            if (Test-Path $candidate) {
                return $candidate
            }
        }
        Start-Sleep -Seconds 2
    }
    throw "Could not find attached payload containing ccproxy.wsl"
}

function Get-CollectorBase {
    $persisted = Join-Path $stateDir "CollectorUrl.txt"
    if (Test-Path $persisted) {
        return (Get-Content $persisted -Raw).Trim()
    }

    for ($i = 0; $i -lt 180; $i++) {
        foreach ($drive in Get-PSDrive -PSProvider FileSystem) {
            $candidate = Join-Path $drive.Root "CollectorUrl.txt"
            if (Test-Path $candidate) {
                Copy-Item -Path $candidate -Destination $persisted -Force
                return (Get-Content $persisted -Raw).Trim()
            }
        }
        Start-Sleep -Seconds 2
    }
    throw "Could not find collector URL on attached answer media"
}

function Send-CollectorText {
    param(
        [string]$CollectorBase,
        [string]$Path,
        [string]$Body,
        [string]$ContentType = "text/plain"
    )

    $uri = "$CollectorBase/$Path"
    Invoke-WebRequest -Uri $uri -Method Post -Body $Body -ContentType $ContentType -UseBasicParsing | Out-Null
}

function Send-CollectorFile {
    param(
        [string]$CollectorBase,
        [string]$Path,
        [string]$FilePath,
        [string]$ContentType = "text/plain"
    )

    if (Test-Path $FilePath) {
        $body = Get-Content -Path $FilePath -Raw
        Send-CollectorText -CollectorBase $CollectorBase -Path $Path -Body $body -ContentType $ContentType
    }
}

function Invoke-Step {
    param(
        [string]$Name,
        [scriptblock]$Script
    )

    Write-Host "[ccproxy-wsl-smoke] $Name"
    $output = & $Script 2>&1
    $code = if ($null -eq $LASTEXITCODE) { 0 } else { $LASTEXITCODE }
    $script:steps += [ordered]@{
        name = $Name
        exit_code = $code
        output = @($output | ForEach-Object { "$_" })
    }
    if ($code -ne 0) {
        throw "Step failed: $Name ($code)"
    }
}

function ConvertTo-NativeArgument {
    param([string]$Argument)

    if ($null -eq $Argument) {
        return '""'
    }
    if ($Argument -eq "") {
        return '""'
    }
    if ($Argument -notmatch '[\s"]') {
        return $Argument
    }

    $result = '"'
    $backslashes = 0
    foreach ($character in $Argument.ToCharArray()) {
        if ($character -eq '\') {
            $backslashes += 1
        }
        elseif ($character -eq '"') {
            $result += '\' * (($backslashes * 2) + 1)
            $result += '"'
            $backslashes = 0
        }
        else {
            if ($backslashes -gt 0) {
                $result += '\' * $backslashes
                $backslashes = 0
            }
            $result += $character
        }
    }
    if ($backslashes -gt 0) {
        $result += '\' * ($backslashes * 2)
    }
    $result += '"'
    return $result
}

function Invoke-Native {
    param(
        [string]$FilePath,
        [string[]]$Arguments = @(),
        [int]$TimeoutSeconds = 300
    )

    $psi = [System.Diagnostics.ProcessStartInfo]::new()
    $psi.FileName = $FilePath
    $psi.Arguments = ($Arguments | ForEach-Object { ConvertTo-NativeArgument $_ }) -join " "
    $psi.RedirectStandardOutput = $true
    $psi.RedirectStandardError = $true
    $psi.UseShellExecute = $false
    $process = [System.Diagnostics.Process]::new()
    $process.StartInfo = $psi
    [void]$process.Start()

    if (-not $process.WaitForExit($TimeoutSeconds * 1000)) {
        try {
            $process.Kill()
        }
        catch {
        }
        throw "Timed out after $TimeoutSeconds seconds: $FilePath $($Arguments -join ' ')"
    }

    $stdout = $process.StandardOutput.ReadToEnd()
    $stderr = $process.StandardError.ReadToEnd()
    $global:LASTEXITCODE = $process.ExitCode

    $lines = @()
    if ($stdout) {
        $lines += $stdout -split "`r?`n"
    }
    if ($stderr) {
        $lines += $stderr -split "`r?`n"
    }
    $lines | Where-Object { $_ -ne "" }
}

function Write-SmokeResult {
    param(
        [string]$CollectorBase,
        [bool]$Ok,
        [string]$ErrorMessage
    )

    $payload = [ordered]@{
        ok = $Ok
        error = $ErrorMessage
        stage = if (Test-Path $stagePath) { Get-Content $stagePath -Raw } else { "" }
        timestamp = (Get-Date).ToUniversalTime().ToString("o")
        computer = $env:COMPUTERNAME
        user = "$env:USERDOMAIN\$env:USERNAME"
        steps = $script:steps
    }

    $json = $payload | ConvertTo-Json -Depth 20
    Send-CollectorText -CollectorBase $CollectorBase -Path "result" -Body $json -ContentType "application/json"
    Send-CollectorFile -CollectorBase $CollectorBase -Path "bootstrap-log" -FilePath $logPath
    Send-CollectorText -CollectorBase $CollectorBase -Path "stage" -Body $payload.stage
}

$script:steps = @()

try {
    Set-BootstrapRunKey
    $stage = if (Test-Path $stagePath) { (Get-Content $stagePath -Raw).Trim() } else { "0" }

    if ($stage -eq "0") {
        Set-Content -Path $stagePath -Value "1" -Encoding ASCII
        Invoke-Step "enable-wsl-feature" { Invoke-Native -FilePath "dism.exe" -Arguments @("/online", "/enable-feature", "/featurename:Microsoft-Windows-Subsystem-Linux", "/all", "/norestart") -TimeoutSeconds 900 }
        Invoke-Step "enable-vmp-feature" { Invoke-Native -FilePath "dism.exe" -Arguments @("/online", "/enable-feature", "/featurename:VirtualMachinePlatform", "/all", "/norestart") -TimeoutSeconds 900 }
        Restart-Computer -Force
        exit 0
    }

    $collector = Get-CollectorBase
    $artifact = Get-SmokeArtifactPath
    $installRoot = "C:\ccproxy-wsl"
    $distroRoot = Join-Path $installRoot "distro"
    New-Item -ItemType Directory -Force -Path $installRoot | Out-Null

    Invoke-Step "wsl-version-before-update" { Invoke-Native -FilePath "wsl.exe" -Arguments @("--version") -TimeoutSeconds 180 }
    Invoke-Step "wsl-update" {
        Invoke-Native -FilePath "wsl.exe" -Arguments @("--update", "--web-download") -TimeoutSeconds 1800
        if ($LASTEXITCODE -ne 0) {
            Invoke-Native -FilePath "wsl.exe" -Arguments @("--update") -TimeoutSeconds 1800
        }
    }
    Invoke-Step "wsl-set-default-version" { Invoke-Native -FilePath "wsl.exe" -Arguments @("--set-default-version", "2") -TimeoutSeconds 180 }
    Invoke-Step "wsl-unregister-old" {
        Invoke-Native -FilePath "wsl.exe" -Arguments @("--unregister", "ccproxy-smoke") -TimeoutSeconds 180
        if ($LASTEXITCODE -ne 0) {
            $global:LASTEXITCODE = 0
        }
    }
    Invoke-Step "wsl-import-ccproxy" { Invoke-Native -FilePath "wsl.exe" -Arguments @("--import", "ccproxy-smoke", $distroRoot, $artifact, "--version", "2") -TimeoutSeconds 900 }
    Invoke-Step "wsl-version-list" { Invoke-Native -FilePath "wsl.exe" -Arguments @("-l", "-v") -TimeoutSeconds 180 }
    Invoke-Step "ccproxy-help" { Invoke-Native -FilePath "wsl.exe" -Arguments @("-d", "ccproxy-smoke", "--", "bash", "-lc", "ccproxy --help >/dev/null") -TimeoutSeconds 180 }
    Invoke-Step "systemd-status" { Invoke-Native -FilePath "wsl.exe" -Arguments @("-d", "ccproxy-smoke", "--", "bash", "-lc", "systemctl is-system-running --wait") -TimeoutSeconds 300 }

    $bash = @'
set -euo pipefail
tmp="$(mktemp -d /tmp/ccproxy-wsl.XXXXXX)"
export CCPROXY_CONFIG_DIR="$tmp"
ccproxy init
nohup ccproxy start > "$tmp/ccproxy.log" 2>&1 &
daemon="$!"
cleanup() {
  kill "$daemon" >/dev/null 2>&1 || true
}
trap cleanup EXIT
for i in $(seq 1 120); do
  if ccproxy status --proxy >/dev/null 2>&1 && test -s "$tmp/.inspector-wireguard-client.conf"; then
    break
  fi
  sleep 1
done
ccproxy status --proxy
test -s "$tmp/.inspector-wireguard-client.conf"
ccproxy namespace status --json | tee "$tmp/namespace-status.json"
ccproxy namespace doctor --json | tee "$tmp/namespace-doctor.json"
ccproxy run --inspect -- curl -fsS https://example.com -o /dev/null
'@

    Invoke-Step "ccproxy-namespace-smoke" { Invoke-Native -FilePath "wsl.exe" -Arguments @("-d", "ccproxy-smoke", "--", "bash", "-lc", $bash) -TimeoutSeconds 900 }
    Write-SmokeResult -CollectorBase $collector -Ok $true -ErrorMessage ""
    Remove-BootstrapRunKey
    Stop-Transcript | Out-Null
}
catch {
    $message = $_.Exception.ToString()
    try {
        $collector = Get-CollectorBase
        Write-SmokeResult -CollectorBase $collector -Ok $false -ErrorMessage $message
    }
    catch {
        Write-Host $_.Exception.ToString()
    }
    Stop-Transcript | Out-Null
    exit 1
}
POWERSHELL
install -Dm644 "$answer_dir/Bootstrap.ps1" "$answer_dir/\$OEM\$/\$\$/Setup/Scripts/Bootstrap.ps1"

echo "[wsl-kvm] Building unattended answer ISO"
xorriso -as mkisofs -quiet -iso-level 4 -volid AUTOUNATTEND -o "$answer_iso" "$answer_dir"

send_monitor() {
  local command="$1"
  if [[ -S "$monitor" ]]; then
    printf '%s\n' "$command" | socat - "UNIX-CONNECT:$monitor" >/dev/null 2>&1 || true
  fi
}

launch_qemu() {
  local boot_order="$1"
  rm -f "$monitor"
  echo "[wsl-kvm] Launching Windows VM (boot order: $boot_order, VNC: $vnc_display)"
  qemu-system-x86_64 \
    -name ccproxy-wsl-smoke \
    -machine q35,accel=kvm,usb=off,vmport=off,hpet=off \
    -m "$memory" \
    -smp "$cpus" \
    -cpu host,migratable=off,topoext=on,svm=on,npt=on,hv-time=on,hv-relaxed=on,hv-vapic=on,hv-spinlocks=0x1fff,kvm=off \
    -drive if=pflash,format=raw,readonly=on,file="$OVMF_CODE" \
    -drive if=pflash,format=raw,file="$vars" \
    -device ich9-ahci,id=sata \
    -drive file="$disk",if=none,id=system,format=qcow2,cache=writeback,discard=unmap \
    -device ide-hd,drive=system,bus=sata.0 \
    -drive file="$win_iso",if=none,id=winiso,media=cdrom,readonly=on \
    -device ide-cd,drive=winiso,bus=sata.1 \
    -drive file="$answer_iso",if=none,id=answeriso,media=cdrom,readonly=on \
    -device ide-cd,drive=answeriso,bus=sata.2 \
    -drive file="$payload_iso",if=none,id=payloadiso,media=cdrom,readonly=on \
    -device ide-cd,drive=payloadiso,bus=sata.3 \
    -netdev user,id=net0,hostfwd=tcp:127.0.0.1:22222-:22 \
    -device e1000e,netdev=net0 \
    -boot order="$boot_order" \
    -display none \
    -vnc "$vnc_display" \
    -monitor "unix:$monitor,server,nowait" \
    -serial "file:$root/serial.log" \
    >>"$qemu_log" 2>&1 &
  qemu_pid="$!"
}

deadline=$((SECONDS + timeout_seconds))
attempt=0
while (( SECONDS < deadline )); do
  attempt=$((attempt + 1))
  if (( attempt == 1 )); then
    boot_order="d"
  else
    boot_order="c"
  fi

  launch_qemu "$boot_order"

  if (( attempt == 1 )); then
    for _ in $(seq 1 20); do
      [[ -S "$monitor" ]] && break
      sleep 1
    done
    sleep 3
    send_monitor "sendkey ret"
    sleep 3
    send_monitor "sendkey spc"
  fi

  while kill -0 "$qemu_pid" >/dev/null 2>&1; do
    if [[ -f "$result" ]]; then
      echo "[wsl-kvm] Result written: $result"
      jq . "$result" || cat "$result"
      ok="$(jq -r '.ok // false' "$result" 2>/dev/null || echo false)"
      if [[ "$ok" == "true" ]]; then
        exit 0
      fi
      exit 1
    fi
    if (( SECONDS >= deadline )); then
      echo "ERROR: timed out waiting for Windows WSL smoke result" >&2
      exit 1
    fi
    sleep 10
  done

  wait "$qemu_pid" >/dev/null 2>&1 || true
  qemu_pid=""
  echo "[wsl-kvm] VM exited before result; restarting if time remains"
  sleep 5
done

echo "ERROR: timed out waiting for Windows WSL smoke result" >&2
exit 1
