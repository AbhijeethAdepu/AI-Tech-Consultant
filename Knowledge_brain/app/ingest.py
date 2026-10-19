
import asyncio
import json
from pathlib import Path
from urllib.parse import urlparse
import xml.etree.ElementTree as ET

import httpx
from bs4 import BeautifulSoup

SITEMAP_URL = "https://support.brilliantdirectories.com/support/sitemap.xml"
DATA_DIR = Path(__file__).resolve().parent.parent / "data"
OUTPUT_FILE = DATA_DIR / "articles.json"


async def main():
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    headers = {"User-Agent": "AI-Tech-Consultant-Student/1.0"}

    async with httpx.AsyncClient(
        timeout=25.0,
        follow_redirects=True,
        headers=headers,
    ) as client:
        response = await client.get(SITEMAP_URL)
        response.raise_for_status()

        root = ET.fromstring(response.content)
        urls = [
            element.text
            for element in root.iter()
            if element.tag.endswith("loc") and element.text
        ]

        print(f"Found {len(urls)} sitemap URLs")

        articles = []

        # Start with 10 pages to verify the pipeline.
        for index, url in enumerate(urls[:10], start=1):
            try:
                parsed = urlparse(url)
                if parsed.scheme not in {"http", "https"}:
                    continue

                page = await client.get(url)
                page.raise_for_status()

                soup = BeautifulSoup(page.text, "lxml")

                for element in soup(
                    ["script", "style", "nav", "footer", "header"]
                ):
                    element.decompose()

                main_content = (
                    soup.find("article")
                    or soup.find("main")
                    or soup.find(id="content")
                    or soup.body
                )

                if not main_content:
                    print(f"[{index}] No content found: {url}")
                    continue

                text = " ".join(main_content.stripped_strings)

                if len(text) < 100:
                    print(f"[{index}] Too little content: {url}")
                    continue

                articles.append({
                    "url": url,
                    "title": soup.title.get_text(strip=True)
                    if soup.title else "",
                    "content": text,
                })

                print(f"[{index}] Ingested: {url}")

                # Be polite to the source website.
                await asyncio.sleep(1)

            except Exception as exc:
                print(f"[{index}] Failed: {url} ({exc})")

        OUTPUT_FILE.write_text(
            json.dumps(articles, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

        print(f"\nSaved {len(articles)} articles to {OUTPUT_FILE}")


if __name__ == "__main__":
    asyncio.run(main())
