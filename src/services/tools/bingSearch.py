import json

import requests
from bs4 import BeautifulSoup
import re
import random
from urllib.parse import urlparse, parse_qs, urlencode
from typing import Optional, Dict, Any, List
from pydantic import Field, BaseModel
from langchain.tools import BaseTool
import trafilatura
from trafilatura.settings import use_config

from src.monitoring.logger import monitor_task_status


class BingSearch:
    def __init__(self):
        self.headers = {
            'accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.7',
            'accept-language': 'zh-CN,zh;q=0.9',
            'cache-control': 'no-cache',
            'ect': '4g',
            'pragma': 'no-cache',
            'priority': 'u=0, i',
            'referer': 'https://cn.bing.com/',
            'sec-ch-ua': '"Google Chrome";v="143", "Chromium";v="143", "Not A(Brand";v="24"',
            'sec-ch-ua-arch': '"arm"',
            'sec-ch-ua-bitness': '"64"',
            'sec-ch-ua-full-version': '"143.0.7499.148"',
            'sec-ch-ua-full-version-list': '"Google Chrome";v="143.0.7499.148", "Chromium";v="143.0.7499.148", "Not A(Brand";v="24.0.0.0"',
            'sec-ch-ua-mobile': '?0',
            'sec-ch-ua-model': '""',
            'sec-ch-ua-platform': '"macOS"',
            'sec-ch-ua-platform-version': '"15.5.0"',
            'sec-fetch-dest': 'document',
            'sec-fetch-mode': 'navigate',
            'sec-fetch-site': 'same-origin',
            'sec-fetch-user': '?1',
            'upgrade-insecure-requests': '1',
            'user-agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/143.0.0.0 Safari/537.36',
        }
        self.base_url = 'https://cn.bing.com'
        self.session = requests.Session()
        self.session.headers.update(self.headers)

        # 添加更多随机User-Agent
        self.user_agents = [
            'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36',
            'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:109.0) Gecko/20100101 Firefox/120.0',
            'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/118.0.0.0 Safari/537.36 Edg/118.0.2088.76',
        ]

    def search(self, query, max_results=10):
        """
        搜索关键词，支持翻页获取更多结果

        Args:
            query: 搜索关键词
            max_results: 最大结果数量

        Returns:
            搜索结果列表
        """
        all_results = []
        current_page = 1
        seen_links = set()  # 用于去重
        next_url = None

        while len(all_results) < max_results:
            try:
                # 随机切换User-Agent
                self._rotate_user_agent()

                if current_page == 1:
                    # 第一页，构造初始请求
                    page_results, next_url = self._get_first_page(query)
                else:
                    # 后续页面，直接使用next_url
                    if not next_url:
                        break
                    page_results, next_url = self._get_next_page(next_url)

                if not page_results:
                    break

                # 去重并添加新结果
                new_results = []
                for result in page_results:
                    link = result.get('link', '')
                    if link and link not in seen_links:
                        seen_links.add(link)
                        new_results.append(result)

                if not new_results:
                    # 如果没有新结果，可能到达末尾或遇到反爬
                    break

                # 添加当前页结果到总结果中
                remaining_needed = max_results - len(all_results)
                if len(new_results) > remaining_needed:
                    all_results.extend(new_results[:remaining_needed])
                else:
                    all_results.extend(new_results)

                # 如果已经达到目标数量，停止
                if len(all_results) >= max_results:
                    break

                # 如果没有下一页链接，停止
                if not next_url:
                    break

                current_page += 1

                # 添加随机延迟，避免被识别为爬虫
                # delay = random.uniform(2, 5)
                # time.sleep(delay)

                # 限制最大翻页数量
                if current_page > 10:  # 最多翻10页
                    break

            except Exception as e:
                # print(f"获取第 {current_page} 页时出错: {e}")
                break

        return all_results[:max_results]

    def _rotate_user_agent(self):
        """随机切换User-Agent"""
        if random.random() < 0.3:  # 30%的概率切换User-Agent
            new_ua = random.choice(self.user_agents)
            self.session.headers.update({'User-Agent': new_ua})

    def _get_first_page(self, query):
        """
        获取第一页结果
        """
        url = f'{self.base_url}/search'
        params = {
            'q': query,
            'form': 'QBLH',
            'sp': '-1',
            'lq': '0',
            'pq': query,
            'sc': '12-3',
            'qs': 'n',
            'sk': '',
            'cvid': self._generate_random_cvid(),  # 随机cvid
            'first': 1,
            'FORM': 'PERE1'
        }

        try:
            response = self.session.get(url, params=params)
            response.raise_for_status()
            return self._parse_page(response.text, current_url=response.url)

        except requests.RequestException as e:
            # print(f"请求失败: {e}")
            return [], None

    def _get_next_page(self, next_url):
        """
        获取下一页结果
        """
        try:
            # 确保URL是完整的
            if next_url.startswith('/'):
                next_url = self.base_url + next_url
            elif not next_url.startswith('http'):
                next_url = self.base_url + '/' + next_url

            # # 随机延迟
            # time.sleep(random.uniform(1, 3))

            response = self.session.get(next_url)
            response.raise_for_status()
            return self._parse_page(response.text, current_url=response.url)

        except requests.RequestException as e:
            # print(f"请求下一页失败: {e}")
            return [], None

    def _generate_random_cvid(self):
        """生成随机的cvid参数"""
        import uuid
        return str(uuid.uuid4()).upper().replace('-', '')

    def _parse_page(self, html_content, current_url=None):
        """
        解析页面内容，提取搜索结果和下一页链接
        """
        soup = BeautifulSoup(html_content, 'html.parser')

        # 解析搜索结果
        results = self._parse_search_result(html_content)
        monitor_task_status('parse_result',results)

        # 查找下一页链接 - 使用更智能的方法
        next_page_url = self._find_next_page_link_smart(soup, current_url)

        return results, next_page_url

    def _find_next_page_link_smart(self, soup, current_url=None):
        """
        智能查找下一页链接
        """
        # 方法1：查找标准的下一页链接
        next_links = soup.find_all('a', class_=lambda x: x and ('sb_pagN' in x or 'b_widePag' in x or 'sw_next' in x))

        for link in next_links:
            href = link.get('href', '')
            if href:
                # 检查是否是下一页
                link_text = link.get_text().strip().lower()
                aria_label = link.get('aria-label', '').lower()

                if '下一页' in link_text or 'next' in link_text or '下一页' in aria_label or 'next' in aria_label:
                    return self._normalize_url(href, current_url)

        # 方法2：查找分页按钮中的链接
        pagination_div = soup.find('div', class_='pgn_next')
        if pagination_div and pagination_div.parent and pagination_div.parent.name == 'a':
            href = pagination_div.parent.get('href', '')
            if href:
                return self._normalize_url(href, current_url)

        # 方法3：查找包含分页数字的链接
        page_links = soup.find_all('a', href=True)
        for link in page_links:
            href = link['href']
            # 检查URL中是否包含分页参数
            parsed = urlparse(href)
            query_params = parse_qs(parsed.query)

            # 检查是否有明显的分页参数
            if 'first' in query_params or 'page' in query_params or 'start' in query_params:
                link_text = link.get_text().strip()
                # 检查链接文本是否是数字或下一页
                if link_text.isdigit() or '下一页' in link_text or 'next' in link_text:
                    return self._normalize_url(href, current_url)

        # 方法4：尝试构建下一页URL
        if current_url:
            parsed = urlparse(current_url)
            query_params = parse_qs(parsed.query)

            # 尝试增加first参数
            if 'first' in query_params:
                try:
                    first_value = int(query_params['first'][0])
                    next_first = first_value + 10  # Bing通常每页10个结果

                    # 更新参数
                    query_params['first'] = [str(next_first)]

                    # 重新构建URL
                    new_query = urlencode(query_params, doseq=True)
                    next_url = f"{parsed.scheme}://{parsed.netloc}{parsed.path}?{new_query}"
                    return next_url
                except:
                    pass

        return None

    def _normalize_url(self, href, referer_url=None):
        """
        规范化URL
        """
        if not href:
            return None

        # 如果已经是完整URL，直接返回
        if href.startswith('http://') or href.startswith('https://'):
            return href

        # 如果是相对路径
        if href.startswith('/'):
            return self.base_url + href

        # 其他情况，尝试基于referer_url构建
        if referer_url:
            parsed_referer = urlparse(referer_url)
            base_url = f"{parsed_referer.scheme}://{parsed_referer.netloc}"

            if href.startswith('?'):
                # 只有查询参数的情况
                return f"{base_url}{parsed_referer.path}{href}"
            elif href.startswith('./'):
                # 相对当前目录
                path = parsed_referer.path
                if path.endswith('/'):
                    return f"{base_url}{path}{href[2:]}"
                else:
                    dir_path = '/'.join(path.split('/')[:-1])
                    return f"{base_url}{dir_path}/{href[2:]}"

        # 默认情况
        return self.base_url + '/' + href

    def _parse_search_result(self, html_content):
        """解析搜索结果"""
        results = []
        soup = BeautifulSoup(html_content, 'html.parser')

        # 查找搜索结果列表
        search_results = soup.find('ol', attrs={'id': 'b_results'})
        if not search_results:
            return results

        li_results = search_results.find_all('li', recursive=False)

        for index, li in enumerate(li_results):
            # 跳过空项
            if not li.get_text().strip():
                continue

            # 检查是否是分页组件
            if 'b_pag' in li.get('class', []):
                continue

            result_item = self._parse_li_item(li, index)
            if result_item:
                results.append(result_item)

        return results

    def _parse_li_item(self, li, index):
        """解析单个搜索结果项"""
        result_item = {
            'index': index,
            'type': 'unknown',
            'title': '',
            'link': '',
            'description': '',
            'site_info': {},
            'meta_info': {}
        }

        # 检查是否是常规搜索结果 (b_algo)
        if 'b_algo' in li.get('class', []):
            result_item['type'] = 'search_result'

            # 解析标题和链接
            title_elem = li.find('h2')
            if title_elem:
                link_elem = title_elem.find('a')
                if link_elem:
                    result_item['title'] = link_elem.get_text().strip()
                    result_item['link'] = link_elem.get('href', '')
                    # 提取iid信息
                    h_attr = link_elem.get('h', '')
                    if h_attr:
                        match = re.search(r'ID=SERP,(\d+\.\d+)', h_attr)
                        if match:
                            result_item['meta_info']['iid'] = match.group(1)

            # 解析描述
            caption_elem = li.find('div', class_='b_caption')
            if caption_elem:
                desc_elem = caption_elem.find('p')
                if desc_elem:
                    result_item['description'] = desc_elem.get_text().strip()

            # 解析网站信息
            tpcn_elem = li.find('div', class_='b_tpcn')
            if tpcn_elem:
                # 网站名称
                site_name_elem = tpcn_elem.find('div', class_='tptt')
                if site_name_elem:
                    result_item['site_info']['name'] = site_name_elem.get_text().strip()

                # 网站URL
                cite_elem = tpcn_elem.find('cite')
                if cite_elem:
                    result_item['site_info']['url'] = cite_elem.get_text().strip()

                # 网站图标
                icon_elem = tpcn_elem.find('div', class_='rms_iac')
                if icon_elem:
                    result_item['site_info']['icon'] = icon_elem.get('data-src', '')

            # 解析日期
            date_patterns = ['b_lineclamp2', 'b_caption']
            for pattern in date_patterns:
                elem = li.find(class_=pattern)
                if elem:
                    text = elem.get_text()
                    # 尝试提取日期
                    date_match = re.search(r'(\d{4}年\d{1,2}月\d{1,2}日)', text)
                    if date_match:
                        result_item['meta_info']['date'] = date_match.group(1)
                        break

        # 其他类型的列表项
        else:
            # 检查是否有链接
            links = li.find_all('a')
            if links:
                result_item['type'] = 'link_item'
                main_link = links[0]
                result_item['title'] = main_link.get_text().strip()
                result_item['link'] = main_link.get('href', '')
                result_item['description'] = li.get_text().strip()

        return result_item

    def close(self):
        """关闭session"""
        if self.session:
            self.session.close()


class BingSearchTool(BaseTool):
    """Bing搜索工具，可以搜索网页内容并返回结构化结果"""

    name: str = "bing_search"
    description: str = """
    使用Bing搜索引擎进行网页搜索。当需要获取最新信息、查找资料或搜索特定内容时使用此工具。

    输入格式：
    - 搜索关键词或问题

    输出格式：
    - 搜索结果列表，每个结果包含标题、链接和摘要
    """

    # 工具参数
    max_results: int = Field(default=10, description="最大返回结果数量")
    extract: bool = Field(default=True,description="是否请求搜索到的url，并从中提取主要内容")
    filter_fields: list = Field(default=['link','title','text'], description="过滤字段")

    # 内部使用的Bing搜索实例
    _search_engine: Any = None

    class InputSchema(BaseModel):
        """工具输入模式"""
        query: str = Field(description="搜索关键词或问题")
        max_results: Optional[int] = Field(default=None, description="最大结果数量，可选")

    args_schema = InputSchema

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._search_engine = BingSearch()

    def _run(self,query: str,max_results: Optional[int] = None) -> str:
        """
        执行搜索并返回格式化结果

        Args:
            query: 搜索关键词
            max_results: 最大结果数量

        Returns:
            格式化后的搜索结果字符串
        """
        monitor_task_status('开始调用bing_search工具')
        monitor_task_status('正在搜索...',query)
        # 使用传入参数或默认值
        max_results = max_results if max_results is not None else self.max_results

        # 执行搜索
        results = self._search_engine.search(query, max_results)

        # 获取详细内容
        if self.extract:
            results = self.extract_docs(results)

        # 格式化结果
        formatted_result = self._format_results(results)
        monitor_task_status('bing_search工具调用完成')
        return formatted_result

    def _format_results(self, results: List[Dict]) -> str:
        """
        格式化搜索结果

        Args:
            results: 搜索结果列表

        Returns:
            格式化后的字符串
        """
        if not results:
            return "没有找到相关结果。"

        formatted_lines = []

        for i, result in enumerate(results):
            if result['type'] == 'search_result':
                if result.get('title') and 'title' in self.filter_fields:
                    formatted_lines.append(f"{i + 1}. {result['title']}")
                if result.get('link') and 'link' in self.filter_fields:
                    formatted_lines.append(f"   链接: {result['link']}")
                # 如果没有获取到正文，拿摘要来填充
                if result.get('text') and 'text' in self.filter_fields:
                    formatted_lines.append(f"   正文: {result['text']}")
                elif result.get('description') and ('description' in self.filter_fields or not result.get('text')):
                    formatted_lines.append(f"   摘要: {result['description']}")

                if result.get('site_info') and 'site_info' in self.filter_fields:
                    site = result['site_info']
                    formatted_lines.append(f"   来源: {site.get('name', '未知网站')}")
                if result.get('meta_info') and 'meta_info' in self.filter_fields:
                    formatted_lines.append(f"   日期: {result['meta_info']['date']}")

                formatted_lines.append("")

        return "\n".join(formatted_lines)

    def extract_docs(self, url_results: List[Dict]) -> List:
        """请求搜索到的url，并从中提取主要内容"""
        results = []

        for item in url_results:
            url = item['link']
            try:
                downloaded = trafilatura.fetch_url(url)

                # 如果下载的内容是bytes，尝试解码
                if isinstance(downloaded, bytes):
                    try:
                        downloaded = downloaded.decode('utf-8')
                    except:
                        try:
                            downloaded = downloaded.decode('gbk', errors='ignore')
                        except:
                            downloaded = downloaded.decode('utf-8', errors='ignore')

                # 提取内容
                obj = trafilatura.extract(downloaded, output_format='json', with_metadata=True)
                if obj:
                    result = json.loads(obj)
                    text = result.get('text', '')

                    # 乱码检测
                    if text:
                        # 统计正常中英文字符
                        normal_chars = sum(1 for char in text[:500]  # 只检查前500字符
                                           if ('\u4e00' <= char <= '\u9fff') or  # 中文
                                           char.isalnum() or  # 字母数字
                                           char in ' .,;:!?\'"()-_')

                        # 如果正常字符比例太低，可能是乱码
                        if len(text[:500]) > 0 and normal_chars / len(text[:500]) < 0.3:
                            monitor_task_status(f"警告：检测到可能乱码 {url}",level='WARNING')
                            result['text'] = ''  # 清空乱码文本
                            result['encoding_issue'] = True

                    results.append({**result, **item})
                else:
                    results.append({**item, 'text': '', 'error': 'No content extracted'})

            except Exception as e:
                results.append({**item, 'text': '', 'error': str(e)})

        return results


if __name__ == '__main__':
    bing_tool = BingSearchTool(
        name="bing_search",
        description="使用Bing搜索引擎进行网页搜索",
        max_results=5,  # 默认返回5个结果
        extract=True,
    )

    # 使用工具
    result = bing_tool.invoke("人工智能")
    print(result)
