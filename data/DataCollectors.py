import asyncio
import websockets
import json
import csv
import os
import requests
import zipfile
import time
import pandas as pd

RAW_DATA_FOLDER = "data/raw_data"
if not os.path.exists(RAW_DATA_FOLDER):
    os.makedirs(RAW_DATA_FOLDER)

class RealTimeDataCollector:
    def __init__(self, pairs, duration_hours, update_interval):
        self.pairs = [p.lower() for p in pairs]
        self.duration = duration_hours * 3600
        self.update_interval = update_interval
        self.data_queues = {pair: asyncio.Queue() for pair in self.pairs}
        self.stop_event = asyncio.Event()

    async def connect_and_listen(self):
        streams = '/'.join([f"{pair}@trade" for pair in self.pairs])
        url = f"wss://stream.binance.com:9443/stream?streams={streams}"
        print(f"[SYSTEM] Connection to Binance WebSocket")

        try:
            async with websockets.connect(url) as websocket:
                while not self.stop_event.is_set():
                    try:
                        message = await asyncio.wait_for(websocket.recv(), timeout=30)
                        data = json.loads(message)

                        stream = data.get("stream", "")
                        payload = data.get("data", {})

                        pair = stream.split("@")[0]
                        ts = payload.get("T", int(time.time() * 1000))
                        price = payload.get("p")
                        qty = payload.get("q")

                        if pair in self.data_queues:
                            await self.data_queues[pair].put((ts, price, qty))

                    except asyncio.TimeoutError:
                        print("[SYSTEM] No data found since 30 seconds.")
        except Exception as e:
            print(f"[SYSTEM] Websocket error : {e}")

    async def periodic_saver(self, pair):
        filename = os.path.join(RAW_DATA_FOLDER, f"REAL_TIME_{pair.upper()}.csv")

        if not os.path.isfile(filename):
            with open(filename, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["timestamp", "price", "volume"])

        phase = 1
        while not self.stop_event.is_set():
            await asyncio.sleep(self.update_interval)
            items = []

            while not self.data_queues[pair].empty():
                items.append(await self.data_queues[pair].get())

            if items:
                with open(filename, "a", newline="") as f:
                    writer = csv.writer(f)
                    writer.writerows(items)
                print(f"[{pair.upper()}] Phase {phase} : {len(items)} transactions saved.")
                phase += 1

        final_items = []
        while not self.data_queues[pair].empty():
            final_items.append(await self.data_queues[pair].get())

        if final_items:
            with open(filename, "a", newline="") as f:
                writer = csv.writer(f)
                writer.writerows(final_items)
            print(f"[{pair.upper()}] Final update : {len(final_items)} remaining transactions saved.")

    async def run(self):
        listener_task = asyncio.create_task(self.connect_and_listen())
        saver_tasks = [asyncio.create_task(self.periodic_saver(pair)) for pair in self.pairs]

        await asyncio.sleep(self.duration)
        self.stop_event.set()

        await listener_task
        await asyncio.gather(*saver_tasks)


class HistoricalDataCollector:
    BASE_URL = "https://data.binance.vision/data/spot"

    def __init__(self, pairs, year, month, day=None):
        self.crypto_pairs = pairs
        self.year = str(year)
        self.month = f"{int(month):02d}"
        self.day = f"{int(day):02d}" if day else None

        os.makedirs(RAW_DATA_FOLDER, exist_ok=True)

    def _build_file_paths(self, pair):
        if self.day:
            zip_filename = f"{pair}-trades-{self.year}-{self.month}-{self.day}.zip"
            csv_filename = f"{pair}-trades-{self.year}-{self.month}-{self.day}.csv"
        else:
            zip_filename = f"{pair}-trades-{self.year}-{self.month}.zip"
            csv_filename = f"{pair}-trades-{self.year}-{self.month}.csv"

        zip_filepath = os.path.join(".", zip_filename)
        csv_output_path = os.path.join(RAW_DATA_FOLDER, csv_filename)
        return zip_filename, csv_filename, zip_filepath, csv_output_path

    def _download_zip(self, pair, zip_filename, zip_filepath):
        if self.day:
            url = f"{self.BASE_URL}/daily/trades/{pair}/{zip_filename}"
        else:
            url = f"{self.BASE_URL}/monthly/trades/{pair}/{zip_filename}"

        print(f"[SYSTEM] Downloading {zip_filename} from {url} ...")
        response = requests.get(url)
        if response.status_code == 200:
            with open(zip_filepath, 'wb') as f:
                f.write(response.content)
            print(f"[SYSTEM] File downloaded: {zip_filename}")
        else:
            raise Exception(f"Error for the pair {pair} ({response.status_code})")

    def _extract_zip(self, zip_filepath, output_dir):
        print(f"[SYSTEM] Extracting {zip_filepath} ...")
        with zipfile.ZipFile(zip_filepath, 'r') as zip_ref:
            zip_ref.extractall(output_dir)
        print(f"[SYSTEM] Extraction finished.")

    def collect(self):
        for pair in self.crypto_pairs:
            print(f"[SYSTEM] Processing {pair}...")
            zip_filename, csv_filename, zip_filepath, csv_output_path = self._build_file_paths(pair)

            if os.path.exists(csv_output_path):
                print(f"[SYSTEM] Data already available for {pair} ({csv_filename})")
                continue

            try:
                if not os.path.exists(zip_filepath):
                    self._download_zip(pair, zip_filename, zip_filepath)
                else:
                    print(f"[SYSTEM] ZIP file already exists: {zip_filename}")

                self._extract_zip(zip_filepath, RAW_DATA_FOLDER)

            finally:
                if os.path.exists(zip_filepath):
                    os.remove(zip_filepath)
                    print(f"[SYSTEM] ZIP file deleted: {zip_filename}")


if __name__ == "__main__":
    cryptos = [
        "BTCUSDT", "ETHUSDT", "BNBUSDT", "XRPUSDT", "SOLUSDT", "LTCUSDT", "BCHUSDT"
    ]
    duration_hours = 1
    update_interval = 10

    collector = RealTimeDataCollector(cryptos, duration_hours, update_interval)
    asyncio.run(collector.run())
