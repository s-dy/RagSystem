import os
import re
import json
import uuid
import asyncio
from typing import List, Dict
from docx import Document
import aiohttp

from crawl4ai import AsyncWebCrawler, CacheMode, BrowserConfig, CrawlerRunConfig
from crawl4ai.extraction_strategy import JsonCssExtractionStrategy

from src.crawler.Saver import MetaDataSaver


class WangyiNewsCrawler:
    def __init__(self, saver:MetaDataSaver=None):
        self.saver = saver
        # 浏览器配置
        self.browser_config = BrowserConfig(
            headless=True,
            java_script_enabled=True,
        )

        # 文章页提取策略
        self.article_config = CrawlerRunConfig(
            cache_mode=CacheMode.WRITE_ONLY,
            excluded_tags=["style", "script"],
            remove_overlay_elements=True,
            extraction_strategy=self._get_article_extraction_strategy(),
            delay_before_return_html=1,
        )

    def _get_article_extraction_strategy(self):
        schema = {
            "name": "WangyiNewsContext",
            "baseSelector": ".post_main",
            "fields": [
                {"name": "title", "selector": "h1", "type": "text"},
                {"name": "source", "selector": ".post_info", "type": "text"},
                {"name": "context", "selector": ".post_body", "type": "text"},
            ],
        }
        return JsonCssExtractionStrategy(schema)


    def _parse_source_info(self, source_text: str) -> Dict[str, str]:
        """解析来源字符串中的时间、来源、地点"""
        pattern = r'^(?P<time>\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2}:\d{2})\u3000来源:(?P<source>.*?)(?P<location>[\u4e00-\u9fa5]{2})举报$'
        match = re.search(pattern, source_text or "", re.VERBOSE)
        if match:
            return {
                "time": match.group("time"),
                "source": match.group("source"),
                "location": match.group("location")
            }
        return {"time": "", "source": "", "location": ""}

    async def _fetch_article_list(self, page: int) -> List[str]:
        """获取第 page 页的文章 URL 列表"""
        if page == 1:
            url = "https://news.163.com/special/cm_guonei/?callback=data_callback"
        else:
            url = f"https://news.163.com/special/cm_guonei_0{page}/?callback=data_callback"

        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(url, headers={"User-Agent": "Mozilla/5.0"}) as resp:
                    text = await resp.text()

            match = re.search(r'data_callback\((.*)\)', text,re.S)
            if not match:
                print(f"❌ Page {page}: Invalid JSONP response")
                return []

            json_data = json.loads(match.group(1))
            urls = [item["docurl"] for item in json_data if "docurl" in item]
            print(f"✅ Page {page}: Found {len(urls)} articles")
            return urls

        except Exception as e:
            print(f"⚠️ Failed to fetch page {page}: {e}")
            return []

    def _save_article(self, title: str, content: str, source_info: Dict[str, str]):
        """保存文章到 {id}.docx，并追加元信息到 metadata.json"""
        if not self.saver:
            print('为配置存储器')
            return
        article_id = str(uuid.uuid4())
        filename = f"{article_id}.docx"
        filepath = os.path.join(self.saver.data_dir_path, filename)

        # 保存 Word 文档（仅正文）
        doc = Document()
        doc.add_paragraph(content)
        doc.save(filepath)

        # 构建元信息
        meta = {
            "title": title,
            "time": source_info["time"],
            "source": source_info["source"],
            "location": source_info["location"],
            "file_type": 'docx',
            "total_chars": len(content)
        }

        self.saver.save_item(metadata=meta,item_id=article_id)


    async def _crawl_article(self, url: str):
        """爬取单篇文章"""
        try:
            async with AsyncWebCrawler(config=self.browser_config) as crawler:
                result = await crawler.arun(url=url, config=self.article_config)

            if not result.extracted_content:
                print(f"⚠️ Empty content: {url}")
                return

            content = json.loads(result.extracted_content) if isinstance(result.extracted_content, str) else result.extracted_content
            if isinstance(content, list):
                content = content[0] if content else {}

            title = content.get("title", "No Title")
            context = content.get("context", "")
            source = content.get("source", "")

            source_info = self._parse_source_info(source)
            self._save_article(title, context, source_info)

        except Exception as e:
            print(f"❌ Failed to crawl {url}: {e}")

    async def run(self, max_pages: int = 2, max_concurrent: int = 5):
        """主运行函数"""
        print("🔍 Starting Wangyi News Crawler...")

        # 1. 获取所有文章 URL
        all_urls = []
        tasks = [self._fetch_article_list(page) for page in range(1, max_pages + 1)]
        page_results = await asyncio.gather(*tasks)
        for urls in page_results:
            all_urls.extend(urls)

        print(f"📥 Total articles to crawl: {len(all_urls)}")
        if not all_urls:
            print("⚠️ No articles found. Exiting.")
            return

        # 2. 并发爬取文章
        semaphore = asyncio.Semaphore(max_concurrent)

        async def crawl_with_limit(url):
            async with semaphore:
                await self._crawl_article(url)

        crawl_tasks = [crawl_with_limit(url) for url in all_urls]
        await asyncio.gather(*crawl_tasks)

        print("✅ Crawling completed!")


# 运行入口
if __name__ == "__main__":
    with MetaDataSaver() as saver:
        crawler = WangyiNewsCrawler(saver)
        asyncio.run(crawler.run(max_pages=3, max_concurrent=5))
