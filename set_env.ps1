<#
.SYNOPSIS
  从 config.yaml（单层键值）中读取配置，仅在当前 PowerShell 进程设置环境变量（Process作用域），并更新 PYTHONPATH。

.DESCRIPTION
  - 依赖模块：powershell-yaml
  - 遍历 YAML 键值对
  - 将相对路径转换为脚本目录下的绝对路径
  - 使用 .NET API 设置到当前进程，不写入注册表
  - 最后在当前会话中更新 PYTHONPATH，添加脚本所在目录
#>

# 1. 导入 powershell-yaml
if (-not (Get-Module -ListAvailable -Name powershell-yaml)) {
    Install-Module -Name powershell-yaml -Force -Scope CurrentUser
}
Import-Module powershell-yaml

# 2. 读取 config.yaml
$scriptDir  = Split-Path $MyInvocation.MyCommand.Definition -Parent
$configPath = Join-Path $scriptDir 'config.yaml'
$config     = Get-Content $configPath -Raw | ConvertFrom-Yaml

# 3. 使用 GetEnumerator 遍历键值对，仅设置到当前进程
foreach ($kv in $config.GetEnumerator()) {
    $key   = $kv.Key
    $value = $kv.Value

    # 如果是字符串且包含路径分隔符，且非绝对路径，则转绝对路径
    if ($value -is [string] -and $value -match '[\\/]' -and -not ([IO.Path]::IsPathRooted($value))) {
        $value = Join-Path $scriptDir $value
    }

    # 设置到 Process 作用域
    [System.Environment]::SetEnvironmentVariable($key, $value, 'Process')

    # 验证并输出真实值
    $current = [System.Environment]::GetEnvironmentVariable($key, 'Process')
    Write-Host "[Process] $key = $current"
}

# 4. 更新 PYTHONPATH，将脚本目录添加到前缀
$env:PYTHONPATH = "$scriptDir;$scriptDir/core;$env:PYTHONPATH"
Write-Host "[Process] PYTHONPATH = $env:PYTHONPATH"
