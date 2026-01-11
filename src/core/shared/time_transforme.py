import re
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta
import jionlp as jio

from src.monitoring.logger import monitor_task_status


class EnhancedTimeParser:
    """时间解析器"""
    def __init__(self):
        self.base_parser = jio.parse_time
        self.now = datetime.now()
        # 预编译正则表达式
        self._compile_patterns()
        # 配置
        self.config = {
            'default_time_span': 24 * 60 * 60,  # 默认时间跨度1天
            'strict_mode': False
        }

    def _compile_patterns(self):
        """预编译常用的正则表达式"""
        # 日期格式模式
        self.patterns = {
            # 年-月-日
            'date_ymd': re.compile(r'(\d{4})[-/年](\d{1,2})[-/月](\d{1,2})[日]?'),
            # 月-日
            'date_md': re.compile(r'(\d{1,2})[-/月](\d{1,2})[日]?'),
            # 年份
            'year': re.compile(r'(\d{4})年'),
            # 月份
            'month': re.compile(r'(\d{1,2})月'),
            # 带前后缀的时间表达式
            'prefix_time': re.compile(r'(前|后|上|下|本|这)(个?)(天|日|周|星期|月|季度|年)'),
            # 时间段
            'time_period': re.compile(r'(凌晨|早晨|早上|上午|中午|下午|傍晚|晚上|深夜|午夜)'),
            # 星期几
            'weekday': re.compile(r'(周|星期)[一二三四五六日天]'),
            # 数字星期几
            'weekday_num': re.compile(r'(周|星期)([1-7])'),
            # 最近/未来
            'recent_future': re.compile(
                r'(最近|过去|以往|未来|今后|接下来)([一二三四五六七八九十\d]+)(天|周|月|季度|年)'),
            # 季节
            'season': re.compile(r'(春|夏|秋|冬)(天|季)'),
            # 简单数字+N天/周/月/年
            'simple_recent_future': re.compile(r'([一二三四五六七八九十\d]+)(天|周|月|季度|年)(内|前|后|之前|之后)?'),
        }

    def parse(self, text: str, time_base: datetime = None, **kwargs) -> List[Dict[str, Any]]:
        """
        解析文本中的时间表达式

        Args:
            text: 包含时间表达式的文本
            time_base: 时间基准点，默认为当前时间
            **kwargs: 配置参数
                - strict_mode: 严格模式，只返回高精度结果
                - include_source: 是否包含源信息

        Returns:
            统一格式的时间解析结果列表
        """
        # 更新配置
        config = {**self.config, **kwargs}
        time_base = time_base or self.now

        results = []

        # 1. 尝试使用jionlp解析
        try:
            jio_results = self._parse_with_jionlp(text, time_base, config)
            if jio_results:
                results.extend(jio_results)
        except Exception as e:
            if config['strict_mode']:
                monitor_task_status(f"jionlp解析失败: {e}",'ERROR')

        # 2. 解析标准日期格式
        std_results = self._parse_standard_formats(text, time_base)
        if std_results:
            results.extend(std_results)

        # 3. 解析相对时间
        relative_results = self._parse_relative_time(text, time_base)
        if relative_results:
            results.extend(relative_results)

        # 4. 解析时间段
        period_results = self._parse_time_periods(text, time_base)
        if period_results:
            results.extend(period_results)

        # 5. 解析特殊表达式
        special_results = self._parse_special_expressions(text, time_base)
        if special_results:
            results.extend(special_results)

        # 6. 解析星期几
        weekday_results = self._parse_weekday_expressions(text, time_base)
        if weekday_results:
            results.extend(weekday_results)

        # 7. 解析简单相对时间（如"三天内"）
        simple_relative_results = self._parse_simple_relative_time(text, time_base)
        if simple_relative_results:
            results.extend(simple_relative_results)

        # 8. 合并和排序结果
        if results:
            results = self._merge_and_sort_results(results, config)

        return results

    def _parse_with_jionlp(self, text: str, time_base: datetime, config: Dict) -> List[Dict[str, Any]]:
        """使用jionlp解析时间"""
        results = []

        try:
            jio_result = self.base_parser(text, time_base)
            monitor_task_status('jio_result', jio_result,'WARNING')
            if isinstance(jio_result, dict):
                formatted = self._convert_jionlp_format(jio_result, config)
                if formatted:
                    results.append(formatted)
            elif isinstance(jio_result, list):
                for item in jio_result:
                    if isinstance(item, dict):
                        formatted = self._convert_jionlp_format(item, config)
                        if formatted:
                            results.append(formatted)

        except Exception as e:
            if config['strict_mode']:
                raise

        return results

    def _convert_jionlp_format(self, jio_result: Dict, config: Dict) -> Optional[Dict]:
        """转换jionlp格式到统一格式"""
        try:
            if 'time' not in jio_result or not jio_result['time']:
                return None

            time_value = jio_result['time']
            definition = jio_result.get('definition', 'blur')

            # 严格模式下只接受准确结果
            if config.get('strict_mode') and definition == 'blur':
                return None

            if isinstance(time_value, list):
                if len(time_value) >= 2:
                    start_time, end_time = time_value[0], time_value[1]
                else:
                    start_time = time_value[0]
                    end_time = self._get_end_time(start_time)
            else:
                start_time = time_value
                end_time = self._get_end_time(start_time)

            result = {
                'type': 'time_span',
                'definition': definition,
                'time': [start_time, end_time],
                'source': 'jionlp',
            }

            if config.get('include_source', False):
                result['original'] = jio_result

            return result

        except Exception:
            return None

    def _parse_standard_formats(self, text: str, time_base: datetime) -> List[Dict[str, Any]]:
        """解析标准日期格式"""
        results = []

        # YYYY-MM-DD
        match = self.patterns['date_ymd'].search(text)
        if match:
            year, month, day = map(int, match.groups())
            try:
                dt = datetime(year, month, day)
                results.append(self._create_time_span(dt, dt + timedelta(days=1), 'accurate', 'std_ymd'))
            except ValueError:
                pass

        # MM-DD
        match = self.patterns['date_md'].search(text)
        if match:
            month, day = map(int, match.groups())
            try:
                dt = datetime(time_base.year, month, day)
                # 如果日期已经过去，则认为是明年
                if dt < time_base:
                    dt = datetime(time_base.year + 1, month, day)
                results.append(self._create_time_span(dt, dt + timedelta(days=1), 'accurate', 'std_md'))
            except ValueError:
                pass

        # 年份
        match = self.patterns['year'].search(text)
        if match:
            year = int(match.group(1))
            start = datetime(year, 1, 1)
            end = datetime(year + 1, 1, 1)
            results.append(self._create_time_span(start, end, 'accurate', 'year'))

        # 月份
        match = self.patterns['month'].search(text)
        if match:
            month = int(match.group(1))
            year = time_base.year
            # 如果月份已经过去，则认为是明年
            if month < time_base.month:
                year += 1
            start = datetime(year, month, 1)
            end = start + relativedelta(months=1)
            results.append(self._create_time_span(start, end, 'accurate', 'month'))

        return results

    def _parse_relative_time(self, text: str, time_base: datetime) -> List[Dict[str, Any]]:
        """解析相对时间"""
        results = []

        # 昨天、今天、明天
        if '昨天' in text or '昨日' in text:
            start = (time_base - timedelta(days=1)).replace(hour=0, minute=0, second=0)
            end = start + timedelta(days=1)
            results.append(self._create_time_span(start, end, 'accurate', 'yesterday'))

        if '今天' in text or '今日' in text:
            start = time_base.replace(hour=0, minute=0, second=0)
            end = start + timedelta(days=1)
            results.append(self._create_time_span(start, end, 'accurate', 'today'))

        if '明天' in text or '明日' in text:
            start = (time_base + timedelta(days=1)).replace(hour=0, minute=0, second=0)
            end = start + timedelta(days=1)
            results.append(self._create_time_span(start, end, 'accurate', 'tomorrow'))

        if '后天' in text:
            start = (time_base + timedelta(days=2)).replace(hour=0, minute=0, second=0)
            end = start + timedelta(days=1)
            results.append(self._create_time_span(start, end, 'accurate', 'day_after_tomorrow'))

        # 上周、本周、下周
        if '上周' in text or '上星期' in text:
            start = time_base - timedelta(days=time_base.weekday() + 7)
            start = start.replace(hour=0, minute=0, second=0)
            end = start + timedelta(days=7)
            results.append(self._create_time_span(start, end, 'accurate', 'last_week'))

        if '本周' in text or '这周' in text or '本星期' in text:
            start = time_base - timedelta(days=time_base.weekday())
            start = start.replace(hour=0, minute=0, second=0)
            end = start + timedelta(days=7)
            results.append(self._create_time_span(start, end, 'accurate', 'this_week'))

        if '下周' in text or '下星期' in text:
            start = time_base - timedelta(days=time_base.weekday()) + timedelta(days=7)
            start = start.replace(hour=0, minute=0, second=0)
            end = start + timedelta(days=7)
            results.append(self._create_time_span(start, end, 'accurate', 'next_week'))

        # 上月、本月、下月
        if '上月' in text:
            start = (time_base.replace(day=1) - timedelta(days=1)).replace(day=1)
            start = start.replace(hour=0, minute=0, second=0)
            end = start + relativedelta(months=1)
            results.append(self._create_time_span(start, end, 'accurate', 'last_month'))

        if '本月' in text or '这个月' in text:
            start = time_base.replace(day=1, hour=0, minute=0, second=0)
            end = start + relativedelta(months=1)
            results.append(self._create_time_span(start, end, 'accurate', 'this_month'))

        if '下月' in text or '下个月' in text:
            start = (time_base.replace(day=1) + relativedelta(months=1))
            start = start.replace(hour=0, minute=0, second=0)
            end = start + relativedelta(months=1)
            results.append(self._create_time_span(start, end, 'accurate', 'next_month'))

        # 去年、今年、明年
        if '去年' in text:
            start = datetime(time_base.year - 1, 1, 1)
            end = datetime(time_base.year, 1, 1)
            results.append(self._create_time_span(start, end, 'accurate', 'last_year'))

        if '今年' in text:
            start = datetime(time_base.year, 1, 1)
            end = datetime(time_base.year + 1, 1, 1)
            results.append(self._create_time_span(start, end, 'accurate', 'this_year'))

        if '明年' in text:
            start = datetime(time_base.year + 1, 1, 1)
            end = datetime(time_base.year + 2, 1, 1)
            results.append(self._create_time_span(start, end, 'accurate', 'next_year'))

        return results

    def _parse_time_periods(self, text: str, time_base: datetime) -> List[Dict[str, Any]]:
        """解析时间段"""
        results = []

        period_map = {
            '凌晨': (0, 6),
            '早晨': (6, 8),
            '早上': (6, 9),
            '上午': (8, 12),  # 修正：上午是8-12点，不是0-12点
            '中午': (11, 14),
            '下午': (12, 18),
            '傍晚': (17, 19),
            '晚上': (18, 23),
            '深夜': (22, 24),
            '午夜': (23, 1),
        }

        for period, (start_hour, end_hour) in period_map.items():
            if period in text:
                # 确定是哪一天的时间段
                base_date = time_base.replace(hour=0, minute=0, second=0)

                # 处理跨天的情况（如午夜）
                if end_hour < start_hour:
                    start_time = base_date.replace(hour=start_hour)
                    end_time = base_date.replace(hour=end_hour) + timedelta(days=1)
                else:
                    start_time = base_date.replace(hour=start_hour)
                    end_time = base_date.replace(hour=end_hour)

                results.append(self._create_time_span(start_time, end_time, 'accurate', f'period_{period}'))

        return results

    def _parse_special_expressions(self, text: str, time_base: datetime) -> List[Dict[str, Any]]:
        """解析特殊表达式"""
        results = []

        # 最近N天/周/月/年
        match = self.patterns['recent_future'].search(text)
        if match:
            prefix, num_str, unit = match.groups()

            # 转换中文数字
            num = self._chinese_to_number(num_str)
            if num is None:
                try:
                    num = int(num_str)
                except ValueError:
                    num = 1

            if prefix in ['最近', '过去', '以往']:
                # 过去的时间段
                end = time_base
                if unit == '天':
                    start = time_base - timedelta(days=num)
                elif unit == '周':
                    start = time_base - timedelta(weeks=num)
                elif unit == '月':
                    start = time_base - relativedelta(months=num)
                elif unit == '季度':
                    start = time_base - relativedelta(months=num * 3)
                elif unit == '年':
                    start = self._safe_subtract_years(time_base, num)
                else:
                    start = time_base - timedelta(days=num)

                results.append(self._create_time_span(start, end, 'accurate', f'recent_{num}{unit}'))

            elif prefix in ['未来', '今后', '接下来']:
                # 未来的时间段
                start = time_base
                if unit == '天':
                    end = time_base + timedelta(days=num)
                elif unit == '周':
                    end = time_base + timedelta(weeks=num)
                elif unit == '月':
                    end = time_base + relativedelta(months=num)
                elif unit == '季度':
                    end = time_base + relativedelta(months=num * 3)
                elif unit == '年':
                    end = time_base + relativedelta(years=num)
                else:
                    end = time_base + timedelta(days=num)

                results.append(self._create_time_span(start, end, 'accurate', f'future_{num}{unit}'))

        # 季节
        match = self.patterns['season'].search(text)
        if match:
            season, _ = match.groups()
            season_map = {
                '春': (3, 5),
                '夏': (6, 8),
                '秋': (9, 11),
                '冬': (12, 2),
            }

            if season in season_map:
                start_month, end_month = season_map[season]
                year = time_base.year

                # 处理冬季跨年的情况
                if start_month > end_month:
                    if time_base.month >= start_month:
                        # 今年冬季
                        start_date = datetime(year, start_month, 1)
                        end_date = datetime(year + 1, end_month + 1, 1)
                    else:
                        # 去年冬季
                        start_date = datetime(year - 1, start_month, 1)
                        end_date = datetime(year, end_month + 1, 1)
                else:
                    start_date = datetime(year, start_month, 1)
                    end_date = datetime(year, end_month + 1, 1)

                results.append(self._create_time_span(start_date, end_date, 'accurate', f'season_{season}'))

        # 季度
        quarter_patterns = {
            '一季度': (1, 3), '二季度': (4, 6), '三季度': (7, 9), '四季度': (10, 12),
            '第一季度': (1, 3), '第二季度': (4, 6), '第三季度': (7, 9), '第四季度': (10, 12),
            'Q1': (1, 3), 'Q2': (4, 6), 'Q3': (7, 9), 'Q4': (10, 12),
        }

        for pattern, (start_month, end_month) in quarter_patterns.items():
            if pattern in text:
                year = time_base.year
                start_date = datetime(year, start_month, 1)
                if end_month == 12:
                    end_date = datetime(year + 1, 1, 1)
                else:
                    end_date = datetime(year, end_month + 1, 1)

                results.append(self._create_time_span(start_date, end_date, 'accurate', f'quarter_{pattern}'))

        return results

    def _parse_weekday_expressions(self, text: str, time_base: datetime) -> List[Dict[str, Any]]:
        """解析星期几表达式"""
        results = []

        # 中文星期几
        match = self.patterns['weekday'].search(text)
        if match:
            weekday_str = match.group(0)
            # 转换中文星期几到数字
            weekday_map = {
                '周一': 0, '周二': 1, '周三': 2, '周四': 3, '周五': 4, '周六': 5, '周日': 6,
                '星期一': 0, '星期二': 1, '星期三': 2, '星期四': 3, '星期五': 4, '星期六': 5, '星期日': 6,
            }

            if weekday_str in weekday_map:
                weekday_num = weekday_map[weekday_str]
                # 找到最近的下一个该星期几
                current_weekday = time_base.weekday()
                days_ahead = weekday_num - current_weekday
                if days_ahead <= 0:
                    days_ahead += 7

                target_date = time_base + timedelta(days=days_ahead)
                start = target_date.replace(hour=0, minute=0, second=0)
                end = start + timedelta(days=1)

                results.append(self._create_time_span(start, end, 'accurate', f'weekday_{weekday_str}'))

        # 数字星期几
        match = self.patterns['weekday_num'].search(text)
        if match:
            weekday_num = int(match.group(2)) - 1  # 转换为0-6的格式
            # 找到最近的下一个该星期几
            current_weekday = time_base.weekday()
            days_ahead = weekday_num - current_weekday
            if days_ahead <= 0:
                days_ahead += 7

            target_date = time_base + timedelta(days=days_ahead)
            start = target_date.replace(hour=0, minute=0, second=0)
            end = start + timedelta(days=1)

            results.append(self._create_time_span(start, end, 'accurate', f'weekday_num_{weekday_num + 1}'))

        return results

    def _parse_simple_relative_time(self, text: str, time_base: datetime) -> List[Dict[str, Any]]:
        """解析简单相对时间（如"三天内"）"""
        results = []

        match = self.patterns['simple_recent_future'].search(text)
        if match:
            num_str, unit, suffix = match.groups()

            # 转换中文数字
            num = self._chinese_to_number(num_str)
            if num is None:
                try:
                    num = int(num_str)
                except ValueError:
                    num = 1

            # 根据后缀判断是过去还是未来
            if suffix in ['内', '之后', '后']:
                # 未来时间段
                start = time_base
                if unit == '天':
                    end = time_base + timedelta(days=num)
                elif unit == '周':
                    end = time_base + timedelta(weeks=num)
                elif unit == '月':
                    end = time_base + relativedelta(months=num)
                elif unit == '季度':
                    end = time_base + relativedelta(months=num * 3)
                elif unit == '年':
                    end = time_base + relativedelta(years=num)
                else:
                    end = time_base + timedelta(days=num)

                results.append(self._create_time_span(start, end, 'accurate', f'future_{num}{unit}'))

            elif suffix in ['前', '之前']:
                # 过去时间段
                end = time_base
                if unit == '天':
                    start = time_base - timedelta(days=num)
                elif unit == '周':
                    start = time_base - timedelta(weeks=num)
                elif unit == '月':
                    start = time_base - relativedelta(months=num)
                elif unit == '季度':
                    start = time_base - relativedelta(months=num * 3)
                elif unit == '年':
                    start = self._safe_subtract_years(time_base, num)
                else:
                    start = time_base - timedelta(days=num)

                results.append(self._create_time_span(start, end, 'accurate', f'past_{num}{unit}'))
            else:
                # 没有后缀，默认是过去
                end = time_base
                if unit == '天':
                    start = time_base - timedelta(days=num)
                elif unit == '周':
                    start = time_base - timedelta(weeks=num)
                elif unit == '月':
                    start = time_base - relativedelta(months=num)
                elif unit == '季度':
                    start = time_base - relativedelta(months=num * 3)
                elif unit == '年':
                    start = self._safe_subtract_years(time_base, num)
                else:
                    start = time_base - timedelta(days=num)

                results.append(self._create_time_span(start, end, 'accurate', f'past_{num}{unit}'))

        return results

    def _safe_subtract_years(self, time_base: datetime, years: int) -> datetime:
        """安全地减去年份，避免年份为0"""
        new_year = time_base.year - years
        # 确保年份不小于1
        if new_year < 1:
            new_year = 1
        return time_base.replace(year=new_year)

    def _create_time_span(self, start: datetime, end: datetime, definition: str, source: str) -> Dict[str, Any]:
        """创建标准的时间范围字典"""
        return {
            'type': 'time_span',
            'definition': definition,
            'time': [
                start.strftime('%Y-%m-%d %H:%M:%S'),
                end.strftime('%Y-%m-%d %H:%M:%S')
            ],
            'source': source,
            'duration_seconds': int((end - start).total_seconds())
        }

    def _get_end_time(self, time_str: str) -> str:
        """根据开始时间获取结束时间"""
        try:
            dt = datetime.strptime(time_str, '%Y-%m-%d %H:%M:%S')
            end_dt = dt + timedelta(seconds=self.config['default_time_span'] - 1)
            return end_dt.strftime('%Y-%m-%d %H:%M:%S')
        except:
            return time_str

    def _chinese_to_number(self, chinese_num: str) -> Optional[int]:
        """中文数字转阿拉伯数字"""
        chinese_map = {
            '一': 1, '二': 2, '三': 3, '四': 4, '五': 5,
            '六': 6, '七': 7, '八': 8, '九': 9, '十': 10,
            '两': 2, '几': 3,  # "几"通常表示3左右
        }

        if chinese_num in chinese_map:
            return chinese_map[chinese_num]
        elif chinese_num.isdigit():
            return int(chinese_num)
        else:
            # 处理"十几"、"二十几"等
            try:
                if '十' in chinese_num:
                    parts = chinese_num.split('十')
                    if len(parts) == 2:
                        tens = chinese_map.get(parts[0], 0)
                        ones = chinese_map.get(parts[1], 0)
                        return tens * 10 + ones
            except:
                pass

        return None

    def _merge_and_sort_results(self, results: List[Dict], config: Dict) -> List[Dict]:
        """合并和排序结果"""
        if not results:
            return []

        # 去重
        seen = set()
        unique_results = []

        for result in results:
            key = (result['time'][0], result['time'][1], result['source'])
            if key not in seen:
                seen.add(key)
                unique_results.append(result)

        # 排序：按时间跨度从小到大，精度从高到低
        unique_results.sort(key=lambda x: (
            x.get('duration_seconds', float('inf')),
            0 if x['definition'] == 'accurate' else 1,
        ))

        # 如果严格模式，只保留高精度结果
        if config.get('strict_mode'):
            unique_results = [r for r in unique_results if r['definition'] == 'accurate']

        return unique_results

    def validate_time_string(self, time_str: str) -> bool:
        """验证时间字符串格式"""
        try:
            datetime.strptime(time_str, '%Y-%m-%d %H:%M:%S')
            return True
        except ValueError:
            return False


class TimeParseTool:
    """时间解析工具"""
    def __init__(self, config: Optional[Dict] = None):
        self.parser = EnhancedTimeParser()
        if config:
            self.parser.config.update(config)

    def __call__(self, text: str, time_base: datetime = None, **kwargs) -> Dict[str, Any]:
        """
        解析文本中的时间表达式
        Args:
            text: 包含时间表达式的文本
            time_base: 时间基准点
            **kwargs: 配置参数
        """
        result = self.parser.parse(text, time_base, **kwargs)
        if not result:
            return {}

        time_result = {
            'definition': result[0]['definition'],
            'time': result[0]['time'],
        }
        monitor_task_status(f'time_parse_tool: 【text】 => {text} 【result】=> {time_result}')
        return time_result

    def validate_time_string(self, time_str: str) -> bool:
        """验证时间字符串格式"""
        return self.parser.validate_time_string(time_str)



if __name__ == '__main__':
    tool = TimeParseTool(config={'include_source':True,'strict_mode':False})
    # 测试用例
    test_groups = {
        "基本时间": ["今天", "明天", "昨天", "后天"],
        "周相关": ["上周", "本周", "下周", "周一", "星期五"],
        "月相关": ["上月", "本月", "下月"],
        "年相关": ["去年", "今年", "明年"],
        "时间段": ["上午", "下午", "晚上"],
        "完整日期": ["2024-01-15", "2024年1月15日", "1月15日"],
        "简单相对时间": ["三天内", "一周后", "两个月前"],
        "复合表达": ["今天上午", "明天下午", "上周五"],
        "复杂文本": [
            "请查询昨天到今天的销售数据",
            "分析2023年Q4的业绩",
            "安排下周一的会议，每天上午9点",
            "总结最近三个月的工作",
        ]
    }
    print(tool("总结最近三个月的工作"))
    print(tool("今天"))
    print(tool("现在"))
    print(tool.validate_time_string('2026-01-07 13:56:15'))
