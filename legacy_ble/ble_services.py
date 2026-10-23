import asyncio
from bleak import BleakClient

DEVICE_ADDRESS = "F8:5B:B6:4E:26:67"   # 改成 ble_scan.py 掃到的 TriAnswer 位址

async def main():
    print("Connecting...")
    async with BleakClient(DEVICE_ADDRESS) as client:
        print("Connected:", client.is_connected)

        services = client.services

        for service in services:
            print("\nService:", service.uuid)
            for char in service.characteristics:
                print("  Characteristic:", char.uuid)
                print("  Properties:", char.properties)

asyncio.run(main())