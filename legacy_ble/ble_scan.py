import asyncio
from bleak import BleakScanner

MIN_RSSI = -75   # 數字越接近 0 訊號越強，例如 -45 很強、-90 很弱

async def main():
    print("Scanning BLE devices...")
    devices = await BleakScanner.discover(
        timeout=10,
        return_adv=True
    )

    results = []

    for device, adv_data in devices.values():
        name = device.name or adv_data.local_name or "Unknown"
        rssi = adv_data.rssi
        service_uuids = adv_data.service_uuids

        # 過濾掉太遠的裝置
        if rssi < MIN_RSSI:
            continue

        results.append((rssi, name, device.address, service_uuids))

    # RSSI 由強到弱排序
    results.sort(reverse=True)

    for rssi, name, address, service_uuids in results:
        print("--------------------------------")
        print("Name:", name)
        print("Address:", address)
        print("RSSI:", rssi)
        print("Service UUIDs:", service_uuids)

asyncio.run(main())