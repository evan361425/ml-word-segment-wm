from lxml.html import fromstring
from lxml.html.clean import Cleaner

_cleaner = Cleaner()
_cleaner.javascript = True
_cleaner.style = True


def unwrap_html_tag(raw: str) -> str:
    source = fromstring(_cleaner.clean_html(raw))
    lines = [l.strip() for l in source.text_content().split("\n") if l.strip()]
    return "\n".join(lines)
