"""
Generic Document Parser Utility

This module provides functionality for parsing PDF and image documents using PaddleOCR,
and converts the parsing results into markdown and JSON formats
"""

import argparse
import hashlib
import logging
import subprocess
import tempfile
from pathlib import Path
from typing import (
    Dict,
    List,
    Optional,
    Union,
    Tuple,
    Any,
    Iterator,
    TypeVar,
)

T = TypeVar("T")


class Parser:
    """
    Base class for document parsing utilities.

    Defines common functionality and constants for parsing different document types.
    """

    # Define common file formats
    OFFICE_FORMATS = {".doc", ".docx", ".ppt", ".pptx", ".xls", ".xlsx"}
    IMAGE_FORMATS = {".png", ".jpeg", ".jpg", ".bmp", ".tiff", ".tif", ".gif", ".webp"}
    TEXT_FORMATS = {".txt", ".md"}

    # Class-level logger
    logger = logging.getLogger(__name__)

    def __init__(self) -> None:
        """Initialize the base parser."""
        pass

    @staticmethod
    def _unique_output_dir(
        base_dir: Union[str, Path], file_path: Union[str, Path]
    ) -> Path:
        """Create a unique output subdirectory for a file to prevent same-name collisions.

        When multiple files share the same name (e.g. dir1/paper.pdf and dir2/paper.pdf),
        their parser output would collide in the same output directory. This creates a
        unique subdirectory by appending a short hash of the file's absolute path. (Fixes #51)

        Args:
            base_dir: The base output directory
            file_path: Path to the input file

        Returns:
            Path like base_dir/paper_a1b2c3d4/ unique per absolute file path.
        """
        file_path = Path(file_path).resolve()
        stem = file_path.stem
        path_hash = hashlib.md5(str(file_path).encode()).hexdigest()[:8]
        return Path(base_dir) / f"{stem}_{path_hash}"

    @classmethod
    def convert_office_to_pdf(
        cls, doc_path: Union[str, Path], output_dir: Optional[str] = None
    ) -> Path:
        """
        Convert Office document (.doc, .docx, .ppt, .pptx, .xls, .xlsx) to PDF.
        Requires LibreOffice to be installed.

        Args:
            doc_path: Path to the Office document file
            output_dir: Output directory for the PDF file

        Returns:
            Path to the generated PDF file
        """
        try:
            # Convert to Path object for easier handling
            doc_path = Path(doc_path)
            if not doc_path.exists():
                raise FileNotFoundError(f"Office document does not exist: {doc_path}")

            name_without_suff = doc_path.stem

            # Prepare output directory
            if output_dir:
                base_output_dir = Path(output_dir)
            else:
                base_output_dir = doc_path.parent / "libreoffice_output"

            base_output_dir.mkdir(parents=True, exist_ok=True)

            # Create temporary directory for PDF conversion
            with tempfile.TemporaryDirectory() as temp_dir:
                temp_path = Path(temp_dir)

                # Convert to PDF using LibreOffice
                cls.logger.info(
                    f"Converting {doc_path.name} to PDF using LibreOffice..."
                )

                # Prepare subprocess parameters to hide console window on Windows
                import platform

                # Try LibreOffice commands in order of preference
                commands_to_try = ["libreoffice", "soffice"]

                conversion_successful = False
                for cmd in commands_to_try:
                    try:
                        convert_cmd = [
                            cmd,
                            "--headless",
                            "--convert-to",
                            "pdf",
                            "--outdir",
                            str(temp_path),
                            str(doc_path),
                        ]

                        # Prepare conversion subprocess parameters
                        convert_subprocess_kwargs = {
                            "capture_output": True,
                            "text": True,
                            "timeout": 60,  # 60 second timeout
                            "encoding": "utf-8",
                            "errors": "ignore",
                        }

                        # Hide console window on Windows
                        if platform.system() == "Windows":
                            convert_subprocess_kwargs["creationflags"] = (
                                subprocess.CREATE_NO_WINDOW
                            )

                        result = subprocess.run(
                            convert_cmd, **convert_subprocess_kwargs
                        )

                        if result.returncode == 0:
                            conversion_successful = True
                            cls.logger.info(
                                f"Successfully converted {doc_path.name} to PDF using {cmd}"
                            )
                            break
                        else:
                            cls.logger.warning(
                                f"LibreOffice command '{cmd}' failed: {result.stderr}"
                            )
                    except FileNotFoundError:
                        cls.logger.warning(f"LibreOffice command '{cmd}' not found")
                    except subprocess.TimeoutExpired:
                        cls.logger.warning(f"LibreOffice command '{cmd}' timed out")
                    except Exception as e:
                        cls.logger.error(
                            f"LibreOffice command '{cmd}' failed with exception: {e}"
                        )

                if not conversion_successful:
                    raise RuntimeError(
                        f"LibreOffice conversion failed for {doc_path.name}. "
                        f"Please ensure LibreOffice is installed:\n"
                        "- Windows: Download from https://www.libreoffice.org/download/download/\n"
                        "- macOS: brew install --cask libreoffice\n"
                        "- Ubuntu/Debian: sudo apt-get install libreoffice\n"
                        "- CentOS/RHEL: sudo yum install libreoffice\n"
                        "Alternatively, convert the document to PDF manually."
                    )

                # Find the generated PDF
                pdf_files = list(temp_path.glob("*.pdf"))
                if not pdf_files:
                    raise RuntimeError(
                        f"PDF conversion failed for {doc_path.name} - no PDF file generated. "
                        f"Please check LibreOffice installation or try manual conversion."
                    )

                pdf_path = pdf_files[0]
                cls.logger.info(
                    f"Generated PDF: {pdf_path.name} ({pdf_path.stat().st_size} bytes)"
                )

                # Validate the generated PDF
                if pdf_path.stat().st_size < 100:  # Very small file, likely empty
                    raise RuntimeError(
                        "Generated PDF appears to be empty or corrupted. "
                        "Original file may have issues or LibreOffice conversion failed."
                    )

                # Copy PDF to final output directory
                final_pdf_path = base_output_dir / f"{name_without_suff}.pdf"
                import shutil

                shutil.copy2(pdf_path, final_pdf_path)

                return final_pdf_path

        except Exception as e:
            cls.logger.error(f"Error in convert_office_to_pdf: {str(e)}")
            raise

    @classmethod
    def convert_text_to_pdf(
        cls, text_path: Union[str, Path], output_dir: Optional[str] = None
    ) -> Path:
        """
        Convert text file (.txt, .md) to PDF using ReportLab with full markdown support.

        Args:
            text_path: Path to the text file
            output_dir: Output directory for the PDF file

        Returns:
            Path to the generated PDF file
        """
        try:
            text_path = Path(text_path)
            if not text_path.exists():
                raise FileNotFoundError(f"Text file does not exist: {text_path}")

            # Supported text formats
            supported_text_formats = {".txt", ".md"}
            if text_path.suffix.lower() not in supported_text_formats:
                raise ValueError(f"Unsupported text format: {text_path.suffix}")

            # Read the text content
            try:
                with open(text_path, "r", encoding="utf-8") as f:
                    text_content = f.read()
            except UnicodeDecodeError:
                # Try with different encodings
                for encoding in ["gbk", "latin-1", "cp1252"]:
                    try:
                        with open(text_path, "r", encoding=encoding) as f:
                            text_content = f.read()
                        cls.logger.info(
                            f"Successfully read file with {encoding} encoding"
                        )
                        break
                    except UnicodeDecodeError:
                        continue
                else:
                    raise RuntimeError(
                        f"Could not decode text file {text_path.name} with any supported encoding"
                    )

            # Prepare output directory
            if output_dir:
                base_output_dir = Path(output_dir)
            else:
                base_output_dir = text_path.parent / "reportlab_output"

            base_output_dir.mkdir(parents=True, exist_ok=True)
            pdf_path = base_output_dir / f"{text_path.stem}.pdf"

            # Convert text to PDF
            cls.logger.info(f"Converting {text_path.name} to PDF...")

            try:
                from reportlab.lib.pagesizes import A4
                from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
                from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
                from reportlab.lib.units import inch
                from reportlab.pdfbase import pdfmetrics
                from reportlab.pdfbase.ttfonts import TTFont

                support_chinese = True
                try:
                    if "WenQuanYi" not in pdfmetrics.getRegisteredFontNames():
                        if not Path(
                            "/usr/share/fonts/wqy-microhei/wqy-microhei.ttc"
                        ).exists():
                            support_chinese = False
                            cls.logger.warning(
                                "WenQuanYi font not found at /usr/share/fonts/wqy-microhei/wqy-microhei.ttc. Chinese characters may not render correctly."
                            )
                        else:
                            pdfmetrics.registerFont(
                                TTFont(
                                    "WenQuanYi",
                                    "/usr/share/fonts/wqy-microhei/wqy-microhei.ttc",
                                )
                            )
                except Exception as e:
                    support_chinese = False
                    cls.logger.warning(
                        f"Failed to register WenQuanYi font: {e}. Chinese characters may not render correctly."
                    )

                # Create PDF document
                doc = SimpleDocTemplate(
                    str(pdf_path),
                    pagesize=A4,
                    leftMargin=inch,
                    rightMargin=inch,
                    topMargin=inch,
                    bottomMargin=inch,
                )

                # Get styles
                styles = getSampleStyleSheet()
                normal_style = styles["Normal"]
                heading_style = styles["Heading1"]
                if support_chinese:
                    normal_style.fontName = "WenQuanYi"
                    heading_style.fontName = "WenQuanYi"

                # Try to register a font that supports Chinese characters
                try:
                    # Try to use system fonts that support Chinese
                    import platform

                    system = platform.system()
                    if system == "Windows":
                        # Try common Windows fonts
                        for font_name in ["SimSun", "SimHei", "Microsoft YaHei"]:
                            try:
                                from reportlab.pdfbase.cidfonts import (
                                    UnicodeCIDFont,
                                )

                                pdfmetrics.registerFont(UnicodeCIDFont(font_name))
                                normal_style.fontName = font_name
                                heading_style.fontName = font_name
                                break
                            except Exception:
                                continue
                    elif system == "Darwin":  # macOS
                        for font_name in ["STSong-Light", "STHeiti"]:
                            try:
                                from reportlab.pdfbase.cidfonts import (
                                    UnicodeCIDFont,
                                )

                                pdfmetrics.registerFont(UnicodeCIDFont(font_name))
                                normal_style.fontName = font_name
                                heading_style.fontName = font_name
                                break
                            except Exception:
                                continue
                except Exception:
                    pass  # Use default fonts if Chinese font setup fails

                # Build content
                story = []

                # Handle markdown or plain text
                if text_path.suffix.lower() == ".md":
                    # Handle markdown content - simplified implementation
                    lines = text_content.split("\n")
                    for line in lines:
                        line = line.strip()
                        if not line:
                            story.append(Spacer(1, 12))
                            continue

                        # Headers
                        if line.startswith("#"):
                            level = len(line) - len(line.lstrip("#"))
                            header_text = line.lstrip("#").strip()
                            if header_text:
                                header_style = ParagraphStyle(
                                    name=f"Heading{level}",
                                    parent=heading_style,
                                    fontSize=max(16 - level, 10),
                                    spaceAfter=8,
                                    spaceBefore=16 if level <= 2 else 12,
                                )
                                story.append(Paragraph(header_text, header_style))
                        else:
                            # Regular text
                            story.append(Paragraph(line, normal_style))
                            story.append(Spacer(1, 6))
                else:
                    # Handle plain text files (.txt)
                    cls.logger.info(
                        f"Processing plain text file with {len(text_content)} characters..."
                    )

                    # Split text into lines and process each line
                    lines = text_content.split("\n")
                    line_count = 0

                    for line in lines:
                        line = line.rstrip()
                        line_count += 1

                        # Empty lines
                        if not line.strip():
                            story.append(Spacer(1, 6))
                            continue

                        # Regular text lines
                        # Escape special characters for ReportLab
                        safe_line = (
                            line.replace("&", "&amp;")
                            .replace("<", "&lt;")
                            .replace(">", "&gt;")
                        )

                        # Create paragraph
                        story.append(Paragraph(safe_line, normal_style))
                        story.append(Spacer(1, 3))

                    cls.logger.info(f"Added {line_count} lines to PDF")

                    # If no content was added, add a placeholder
                    if not story:
                        story.append(Paragraph("(Empty text file)", normal_style))

                # Build PDF
                doc.build(story)
                cls.logger.info(
                    f"Successfully converted {text_path.name} to PDF ({pdf_path.stat().st_size / 1024:.1f} KB)"
                )

            except ImportError:
                raise RuntimeError(
                    "reportlab is required for text-to-PDF conversion. "
                    "Please install it using: pip install reportlab"
                )
            except Exception as e:
                raise RuntimeError(
                    f"Failed to convert text file {text_path.name} to PDF: {str(e)}"
                )

            # Validate the generated PDF
            if not pdf_path.exists() or pdf_path.stat().st_size < 100:
                raise RuntimeError(
                    f"PDF conversion failed for {text_path.name} - generated PDF is empty or corrupted."
                )

            return pdf_path

        except Exception as e:
            cls.logger.error(f"Error in convert_text_to_pdf: {str(e)}")
            raise

    @classmethod
    def _process_inline_markdown(cls, text: str) -> str:
        """
        Process inline markdown formatting (bold, italic, code, links)

        Args:
            text: Raw text with markdown formatting

        Returns:
            Text with ReportLab markup
        """
        import re

        # Escape special characters for ReportLab
        text = text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")

        # Bold text: **text** or __text__
        text = re.sub(r"\*\*(.*?)\*\*", r"<b>\1</b>", text)
        text = re.sub(r"__(.*?)__", r"<b>\1</b>", text)

        # Italic text: *text* or _text_ (but not in the middle of words)
        text = re.sub(r"(?<!\w)\*([^*\n]+?)\*(?!\w)", r"<i>\1</i>", text)
        text = re.sub(r"(?<!\w)_([^_\n]+?)_(?!\w)", r"<i>\1</i>", text)

        # Inline code: `code`
        text = re.sub(
            r"`([^`]+?)`",
            r'<font name="Courier" size="9" color="darkred">\1</font>',
            text,
        )

        # Links: [text](url) - convert to text with URL annotation
        def link_replacer(match):
            link_text = match.group(1)
            url = match.group(2)
            return f'<link href="{url}" color="blue"><u>{link_text}</u></link>'

        text = re.sub(r"\[([^\]]+?)\]\(([^)]+?)\)", link_replacer, text)

        # Strikethrough: ~~text~~
        text = re.sub(r"~~(.*?)~~", r"<strike>\1</strike>", text)

        return text

    def parse_pdf(
        self,
        pdf_path: Union[str, Path],
        output_dir: Optional[str] = None,
        method: str = "auto",
        lang: Optional[str] = None,
        **kwargs,
    ) -> List[Dict[str, Any]]:
        """
        Abstract method to parse PDF document.
        Must be implemented by subclasses.
        """
        raise NotImplementedError("parse_pdf must be implemented by subclasses")

    def parse_image(
        self,
        image_path: Union[str, Path],
        output_dir: Optional[str] = None,
        lang: Optional[str] = None,
        **kwargs,
    ) -> List[Dict[str, Any]]:
        """
        Abstract method to parse image document.
        Must be implemented by subclasses.
        """
        raise NotImplementedError("parse_image must be implemented by subclasses")

    def parse_document(
        self,
        file_path: Union[str, Path],
        method: str = "auto",
        output_dir: Optional[str] = None,
        lang: Optional[str] = None,
        **kwargs,
    ) -> List[Dict[str, Any]]:
        """
        Abstract method to parse a document.
        Must be implemented by subclasses.
        """
        raise NotImplementedError("parse_document must be implemented by subclasses")

    def check_installation(self) -> bool:
        """
        Abstract method to check if the parser is properly installed.
        Must be implemented by subclasses.
        """
        raise NotImplementedError(
            "check_installation must be implemented by subclasses"
        )


class PaddleOCRParser(Parser):
    """PaddleOCR document parser with optional PDF page rendering support."""

    def __init__(self, default_lang: str = "ch") -> None:
        super().__init__()
        self.default_lang = default_lang
        self._ocr_instances: Dict[str, Any] = {}

        # Set environment variables to reduce likelihood of segmentation faults
        import os

        os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
        os.environ.setdefault("OMP_NUM_THREADS", "1")
        os.environ.setdefault("PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK", "True")

    def _require_paddleocr(self):
        try:
            from paddleocr import PaddleOCR
        except ImportError as exc:
            raise ImportError(
                "PaddleOCR parser requires optional dependency `paddleocr`. "
                "Install with `pip install -e '.[paddleocr]'` or "
                "`uv sync --extra paddleocr`. "
                "PaddleOCR also needs `paddlepaddle`; install it from "
                "https://www.paddlepaddle.org.cn/install/quick."
            ) from exc
        return PaddleOCR

    def _get_ocr(self, lang: Optional[str] = None):
        PaddleOCR = self._require_paddleocr()
        language = (lang or self.default_lang).strip() or self.default_lang
        cached = self._ocr_instances.get(language)
        if cached is not None:
            return cached

        # More comprehensive initialization options
        init_candidates = [
            {
                "lang": language,
                "show_log": False,
                "use_angle_cls": False,
                "det": True,
                "rec": True,
            },
            {"lang": language, "show_log": False, "use_angle_cls": True},
            {"lang": language, "show_log": False},
            {"lang": language},
            {"lang": "en", "show_log": False, "use_angle_cls": False},
            {},  # Last resort fallback
        ]
        last_exception = None
        for candidate_kwargs in init_candidates:
            try:
                self.logger.debug(
                    f"Attempting to initialize PaddleOCR with params: {candidate_kwargs}"
                )
                ocr = PaddleOCR(**candidate_kwargs)
                self._ocr_instances[language] = ocr
                self.logger.debug("Successfully initialized PaddleOCR instance")
                return ocr
            except Exception as exc:  # pragma: no cover - defensive fallback
                last_exception = exc
                self.logger.warning(
                    f"PaddleOCR initialization failed with params {candidate_kwargs}: {exc}"
                )
                continue

        raise RuntimeError(
            f"Unable to initialize PaddleOCR for language '{language}'. "
            f"Last exception: {last_exception}. "
            f"Please check your PaddleOCR installation and dependencies."
        )

    def _extract_text_lines(self, result: Any) -> List[str]:
        lines: List[str] = []

        def append_text(text: str) -> None:
            clean_text = text.strip()
            if clean_text:
                lines.append(clean_text)

        if isinstance(result, str):
            append_text(result)
            return lines

        def visit(node: Any) -> None:
            if node is None:
                return

            if hasattr(node, "to_dict"):
                try:
                    visit(node.to_dict())
                    return
                except Exception:
                    pass

            if isinstance(node, dict):
                rec_texts = node.get("rec_texts")
                if isinstance(rec_texts, list):
                    for item in rec_texts:
                        if isinstance(item, str):
                            append_text(item)
                        else:
                            visit(item)

                text_value = node.get("text")
                if isinstance(text_value, str):
                    append_text(text_value)

                texts_value = node.get("texts")
                if isinstance(texts_value, list):
                    for item in texts_value:
                        if isinstance(item, str):
                            append_text(item)
                        else:
                            visit(item)

                # Avoid double-visiting keys we already handled above; this prevents
                # accidental duplication without content-level deduplication.
                for key, value in node.items():
                    if key in {"rec_texts", "text", "texts"}:
                        continue
                    visit(value)
                return

            if isinstance(node, (list, tuple)):
                if node and all(isinstance(item, str) for item in node):
                    for item in node:
                        append_text(item)
                    return

                if (
                    len(node) >= 2
                    and isinstance(node[1], (list, tuple))
                    and len(node[1]) >= 1
                    and isinstance(node[1][0], str)
                ):
                    append_text(node[1][0])
                    return

                if (
                    len(node) >= 1
                    and isinstance(node[0], str)
                    and (len(node) == 1 or isinstance(node[1], (int, float)))
                ):
                    append_text(node[0])
                    return

                for item in node:
                    visit(item)
                return

            if isinstance(node, str):
                append_text(node)
                return

        visit(result)
        return lines

    def _ocr_input(
        self, input_data: Any, lang: Optional[str] = None, cls_enabled: bool = True
    ) -> List[str]:
        ocr = self._get_ocr(lang=lang)

        if hasattr(ocr, "ocr"):
            try:
                # First try with cls parameter
                result = ocr.ocr(input_data, cls=cls_enabled)
            except TypeError:
                # Fallback without cls parameter
                result = ocr.ocr(input_data)
            except Exception as e:
                # Handle any other exceptions including segmentation faults
                self.logger.warning(f"Direct OCR call failed: {e}")
                # Try alternative approach with predict method if available
                if hasattr(ocr, "predict"):
                    try:
                        result = ocr.predict(input_data)
                    except Exception as pred_e:
                        self.logger.error(f"Predict method also failed: {pred_e}")
                        raise RuntimeError(f"Both OCR methods failed: {e}, {pred_e}")
                else:
                    raise RuntimeError(
                        f"OCR call failed and no predict method available: {e}"
                    )

            # Add safety check for result
            if result is None:
                self.logger.warning("OCR returned None result")
                return []

            return self._extract_text_lines(result)

        if hasattr(ocr, "predict"):
            try:
                result = ocr.predict(input_data)
                # Add safety check for result
                if result is None:
                    self.logger.warning("Predict returned None result")
                    return []
                return self._extract_text_lines(result)
            except Exception as e:
                self.logger.error(f"Predict method failed: {e}")
                raise RuntimeError(f"Predict method failed: {e}")

        raise RuntimeError(
            "Unsupported PaddleOCR API: expected `ocr` or `predict` method."
        )

    def _extract_pdf_page_inputs(self, pdf_path: Path) -> Iterator[Tuple[int, Any]]:
        try:
            import pypdfium2 as pdfium
        except ImportError as exc:
            raise ImportError(
                "PDF parsing with parser='paddleocr' requires `pypdfium2`. "
                "Install with `pip install -e '.[paddleocr]'` or "
                "`uv sync --extra paddleocr`."
            ) from exc

        pdf = pdfium.PdfDocument(str(pdf_path))
        try:
            total_pages = len(pdf)
            for page_idx in range(total_pages):
                page = pdf[page_idx]
                try:
                    rendered = page.render(scale=2.0)
                    if hasattr(rendered, "to_pil"):
                        yield (page_idx, rendered.to_pil())
                    elif hasattr(rendered, "to_numpy"):
                        yield (page_idx, rendered.to_numpy())
                    else:
                        raise RuntimeError(
                            "Unsupported rendered page format from pypdfium2."
                        )
                finally:
                    if hasattr(page, "close"):
                        page.close()
        finally:
            if hasattr(pdf, "close"):
                pdf.close()

    def _ocr_rendered_page(
        self, rendered_page: Any, lang: Optional[str] = None, cls_enabled: bool = True
    ) -> List[str]:
        if hasattr(rendered_page, "save"):
            temp_image_path: Optional[Path] = None
            try:
                with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as temp:
                    temp_image_path = Path(temp.name)
                rendered_page.save(temp_image_path)
                return self._ocr_input(
                    str(temp_image_path), lang=lang, cls_enabled=cls_enabled
                )
            finally:
                if temp_image_path is not None and temp_image_path.exists():
                    try:
                        temp_image_path.unlink()
                    except Exception:
                        pass

        return self._ocr_input(rendered_page, lang=lang, cls_enabled=cls_enabled)

    def parse_pdf(
        self,
        pdf_path: Union[str, Path],
        output_dir: Optional[str] = None,
        method: str = "auto",
        lang: Optional[str] = None,
        **kwargs,
    ) -> List[Dict[str, Any]]:
        del output_dir, method
        pdf_path = Path(pdf_path)
        if not pdf_path.exists():
            raise FileNotFoundError(f"PDF file does not exist: {pdf_path}")

        cls_enabled = kwargs.get("cls", True)
        content_list: List[Dict[str, Any]] = []
        page_inputs = self._extract_pdf_page_inputs(pdf_path)
        try:
            for page_idx, rendered_page in page_inputs:
                page_lines = self._ocr_rendered_page(
                    rendered_page, lang=lang, cls_enabled=cls_enabled
                )
                for text in page_lines:
                    content_list.append(
                        {"type": "text", "text": text, "page_idx": int(page_idx)}
                    )
        finally:
            # Ensure we promptly release PDF handles even if OCR fails mid-stream.
            close = getattr(page_inputs, "close", None)
            if callable(close):
                close()
        return content_list

    def parse_image(
        self,
        image_path: Union[str, Path],
        output_dir: Optional[str] = None,
        lang: Optional[str] = None,
        **kwargs,
    ) -> List[Dict[str, Any]]:
        del output_dir
        image_path = Path(image_path)
        if not image_path.exists():
            raise FileNotFoundError(f"Image file does not exist: {image_path}")

        ext = image_path.suffix.lower()
        if ext not in self.IMAGE_FORMATS:
            raise ValueError(
                f"Unsupported image format: {ext}. Supported formats: {', '.join(sorted(self.IMAGE_FORMATS))}"
            )

        cls_enabled = kwargs.get("cls", True)
        page_idx = int(kwargs.get("page_idx", 0))
        text_lines = self._ocr_input(
            str(image_path), lang=lang, cls_enabled=cls_enabled
        )
        return [
            {"type": "text", "text": text, "page_idx": page_idx} for text in text_lines
        ]

    def parse_office_doc(
        self,
        doc_path: Union[str, Path],
        output_dir: Optional[str] = None,
        lang: Optional[str] = None,
        **kwargs,
    ) -> List[Dict[str, Any]]:
        pdf_path = self.convert_office_to_pdf(doc_path, output_dir)
        return self.parse_pdf(
            pdf_path=pdf_path, output_dir=output_dir, lang=lang, **kwargs
        )

    def parse_text_file(
        self,
        text_path: Union[str, Path],
        output_dir: Optional[str] = None,
        lang: Optional[str] = None,
        **kwargs,
    ) -> List[Dict[str, Any]]:
        pdf_path = self.convert_text_to_pdf(text_path, output_dir)
        return self.parse_pdf(
            pdf_path=pdf_path, output_dir=output_dir, lang=lang, **kwargs
        )

    def parse_document(
        self,
        file_path: Union[str, Path],
        method: str = "auto",
        output_dir: Optional[str] = None,
        lang: Optional[str] = None,
        **kwargs,
    ) -> List[Dict[str, Any]]:
        del method
        file_path = Path(file_path)
        if not file_path.exists():
            raise FileNotFoundError(f"File does not exist: {file_path}")

        ext = file_path.suffix.lower()
        if ext == ".pdf":
            return self.parse_pdf(file_path, output_dir, lang=lang, **kwargs)
        if ext in self.IMAGE_FORMATS:
            return self.parse_image(file_path, output_dir, lang=lang, **kwargs)
        if ext in self.OFFICE_FORMATS:
            return self.parse_office_doc(file_path, output_dir, lang=lang, **kwargs)
        if ext in self.TEXT_FORMATS:
            return self.parse_text_file(file_path, output_dir, lang=lang, **kwargs)

        raise ValueError(
            f"Unsupported file format: {ext}. "
            "PaddleOCR parser supports PDF, image, office, and text formats."
        )

    def check_installation(self) -> bool:
        try:
            self._require_paddleocr()
            return True
        except ImportError:
            return False


def main():
    """
    Main function to run the document parser from command line
    """
    parser = argparse.ArgumentParser(description="Parse documents using PaddleOCR")
    parser.add_argument("file_path", help="Path to the document to parse")
    parser.add_argument("--output", "-o", help="Output directory path")
    parser.add_argument(
        "--lang",
        "-l",
        help="Document language for OCR optimization (e.g., ch, en, ja)",
    )
    parser.add_argument(
        "--stats", action="store_true", help="Display content statistics"
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Check parser installation",
    )

    args = parser.parse_args()

    # Check installation if requested
    if args.check:
        doc_parser = PaddleOCRParser()
        if doc_parser.check_installation():
            print("✅ PaddleOCR is properly installed")
            return 0
        else:
            print("❌ PaddleOCR installation check failed")
            return 1

    try:
        # Parse the document
        doc_parser = PaddleOCRParser()
        content_list = doc_parser.parse_document(
            file_path=args.file_path,
            output_dir=args.output,
            lang=args.lang,
        )

        print(f"✅ Successfully parsed: {args.file_path}")
        print(f"📊 Extracted {len(content_list)} content blocks")

        # Display statistics if requested
        if args.stats:
            print("\n📈 Document Statistics:")
            print(f"Total content blocks: {len(content_list)}")

            # Count different types of content
            content_types = {}
            for item in content_list:
                if isinstance(item, dict):
                    content_type = item.get("type", "unknown")
                    content_types[content_type] = content_types.get(content_type, 0) + 1

            if content_types:
                print("\n📋 Content Type Distribution:")
                for content_type, count in sorted(content_types.items()):
                    print(f"  • {content_type}: {count}")

    except Exception as e:
        print(f"❌ Error: {str(e)}")
        return 1

    return 0


if __name__ == "__main__":
    exit(main())
