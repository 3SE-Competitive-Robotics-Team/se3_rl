# wuyingyun 机器备忘录

> 通用操作流程见父目录 `SKILL.md`，本文件只记录 wuyingyun 特有的参数和坑。

## 基本信息

| 项目 | 值 |
|---|---|
| IP | 172.31.1.74 |
| 用户名 | zhongjin_lu |
| SSH 别名 | `wuyingyun` |
| 云厂商 | 无影云 |
| 系统 | Ubuntu 22.04，Linux 5.15.0-125-generic |
| GPU | NVIDIA RTX 5880 Ada 48GB，sm_89 |
| CUDA toolkit | 12.8（`/usr/local/cuda-12.8`） |
| CUDA driver | 570.172.18 |
| 主机名 | pntlgaolhyg69qw |
| 仓库路径 | `~/project/se3_rl` |
| uv 路径 | `~/.local/bin/uv`（需 `source ~/.local/bin/env` 激活） |
| zellij | `~/.local/bin/zellij`（已安装） |
| GitHub SSH key | `~/.ssh/github_wuyinyun`（ed25519，账号 `XiaoPengYouCode`） |

## 本机 SSH 配置

本机密钥：`~/.ssh/wuyinyun`（ed25519，当前仅保留密钥文件名），公钥已写入远程 `~/.ssh/authorized_keys`。

`~/.ssh/config`：
```
Host wuyingyun
    HostName 172.31.1.74
    User zhongjin_lu
    IdentityFile ~/.ssh/wuyinyun
    StrictHostKeyChecking no
```

## 远程机器 GitHub SSH 配置

GitHub 22 端口被防火墙封锁，走 443 端口。远程 `~/.ssh/config` 已配置：

```
Host github.com
    HostName ssh.github.com
    Port 443
    User git
    IdentityFile ~/.ssh/github_wuyinyun
    StrictHostKeyChecking no
```

验证：
```bash
ssh wuyingyun "timeout 10 ssh -o ConnectTimeout=8 -T git@github.com 2>&1; echo exit:$?"
# 期望：Hi XiaoPengYouCode! You've successfully authenticated...
```

## 代理端口

| 用途 | 端口 |
|---|---|
| SSH 反向隧道（本机 tinyproxy → 远程） | `17890` |
| 机器本地 Mihomo（备用，不稳定） | `7897` |

## 隧道管理（boring）

使用 [boring](https://github.com/alebeck/boring) 管理反向隧道，配置在 `~/.boring.toml`，支持自动重连和 keep-alive。

```bash
# 开启隧道（后台运行）
boring open wuyingyun-proxy

# 查看状态
boring list

# 关闭隧道
boring close wuyingyun-proxy
```

切换网络后 boring 会自动重连，无需手动重建。

## wuyingyun 特有的依赖问题

### nvshmem

nvshmem 通过机器本地 Mihomo 代理安装（不走 SSH 隧道）：
```bash
ssh wuyingyun "HTTPS_PROXY=http://127.0.0.1:7897 HTTP_PROXY=http://127.0.0.1:7897 uv add nvshmem"
```

### warp-lang 必须锁定 < 1.13.0

warp 1.13.0 移除了 `wp.context` API，mjlab 1.3.0 依赖此 API。`pyproject.toml` 已锁定 `"warp-lang>=1.12.0,<1.13.0"`。
macOS CPU 路径不触发此问题。

验证：
```bash
ssh wuyingyun "source ~/.local/bin/env && cd ~/project/se3_rl && uv pip show warp-lang | grep Version"
```

### PyTorch 版本选择

`pyproject.toml` 按平台自动选择：
- macOS → `pytorch-cpu` index
- Linux → `pytorch-cu128` index（CUDA 12.8，torch>=2.7，兼容 mjlab>=1.3.0）

## ContactSensor body 名称

碰撞传感器匹配：`base_link|lf0_Link|lf1_Link|rf0_Link|rf1_Link`
轮子传感器匹配：`l_wheel_Link|r_wheel_Link`
