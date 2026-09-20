# ~/project/test_proxy_ip.py

import time
from collections import Counter

import requests

# 本机代理端口。
# 如果你使用的是 Clash / Mihomo 之类的软件，
# 7897 很可能是 HTTP 或 mixed 代理端口。
PROXY_URL = "http://127.0.0.1:7897"

PROXIES = {
    "http": PROXY_URL,
    "https": PROXY_URL,
}

# 测试次数
NUM_REQUESTS = 10

# 每次请求之间间隔，避免连续请求过快
INTERVAL_SECONDS = 1


# ---------------------------------------------------------
# 获取当前请求所使用的公网出口 IP。
#
# 请求会通过 PROXIES 指定的本机代理发送。
# api.ipify.org 返回请求来源的公网 IP。
# ---------------------------------------------------------
def get_proxy_ip() -> str:
    response = requests.get(
        "https://api.ipify.org",
        proxies=PROXIES,
        timeout=10,
    )

    # 如果服务器返回 4xx / 5xx，这里直接抛出异常。
    response.raise_for_status()

    return response.text.strip()


# ---------------------------------------------------------
# 连续请求多次，并统计每次看到的出口 IP。
#
# 如果代理是固定节点：
#     很可能每次都是同一个 IP。
#
# 如果代理软件后面的节点组配置成负载均衡 / 随机：
#     可能出现多个不同 IP。
# ---------------------------------------------------------
def main() -> None:
    ips = []

    print(f"代理地址: {PROXY_URL}")
    print(f"测试次数: {NUM_REQUESTS}")
    print("-" * 50)

    for i in range(NUM_REQUESTS):
        try:
            ip = get_proxy_ip()
            ips.append(ip)

            print(f"[{i + 1:02d}] {ip}")

        except requests.RequestException as exc:
            print(f"[{i + 1:02d}] 请求失败: {exc}")

        time.sleep(INTERVAL_SECONDS)

    print("-" * 50)

    counter = Counter(ips)

    print(f"成功请求次数: {len(ips)}")
    print(f"不同出口 IP 数量: {len(counter)}")

    print("\nIP 出现次数：")

    for ip, count in counter.items():
        print(f"  {ip}: {count} 次")

    if len(counter) == 0:
        print("\n没有成功获得出口 IP。")

    elif len(counter) == 1:
        print("\n结果：目前观察到的出口 IP 是固定的。")

    else:
        print("\n结果：观察到了多个出口 IP，代理可能正在进行节点轮换。")


if __name__ == "__main__":
    main()
