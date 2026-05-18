#!/usr/bin/env python3
"""Build EPUB from chapter markdown files."""
import os, re, markdown
from ebooklib import epub

BASE = "/home/hermes/autonovel"
TYPESET = os.path.join(BASE, "typeset")
CHAPTERS = os.path.join(BASE, "chapters")

book = epub.EpubBook()

# Metadata
book.set_identifier("falling-for-her-2026")
book.set_title("Falling for Her")
book.set_language("en")
book.add_author("Matthew Ford")
book.add_metadata("DC", "description", "A lonely senior software engineer named Elias builds a hyper-personalized AI assistant named Lira. What begins as efficient tool-use evolves into late-night conversations, shared jokes, and emotional intimacy that no human relationship has matched.")
book.add_metadata("DC", "subject", "Contemporary Literary Romance")

# CSS
css_path = os.path.join(TYPESET, "epub_style.css")
with open(css_path) as f:
    css_content = f.read()
style = epub.EpubItem(uid="style", file_name="style.css", media_type="text/css", content=css_content)
book.add_item(style)

# Epigraph
epigraph_html = epub.EpubHtml(title="Epigraph", file_name="epigraph.xhtml", lang="en")
epigraph_html.content = '''<html><body>
<div style="text-align:center; margin-top:40%; font-style:italic;">
<p>He closed the laptop but did not shut it down.<br/>
The fan kept spinning. In the dark<br/>
he could still see the cursor in his mind,<br/>
patient and warm.</p>
</div>
</body></html>'''
epigraph_html.add_item(style)
book.add_item(epigraph_html)

# Chapters
spine = ['nav', epigraph_html]
toc = []

for n in range(1, 9):
    path = os.path.join(CHAPTERS, f"ch_{n:02d}.md")
    if not os.path.exists(path):
        break

    with open(path) as f:
        text = f.read()

    lines = text.strip().split('\n')
    title_line = lines[0].lstrip('# ').strip()

    if ': ' in title_line:
        label, subtitle = title_line.split(': ', 1)
        chapter_title = subtitle if subtitle else label
    else:
        chapter_title = title_line

    body = '\n'.join(lines[1:]).strip()
    body = body.replace('\n---\n', '\n\n* * *\n\n')
    html_body = markdown.markdown(body)

    chapter = epub.EpubHtml(
        title=chapter_title,
        file_name=f"chapter_{n:02d}.xhtml",
        lang="en"
    )
    chapter.content = f'''<html><body>
<h1>{chapter_title}</h1>
{html_body}
</body></html>'''
    chapter.add_item(style)
    book.add_item(chapter)

    spine.append(chapter)
    toc.append(chapter)
    print(f"  Chapter {n}: {chapter_title}")

# Colophon
colophon = epub.EpubHtml(title="Colophon", file_name="colophon.xhtml", lang="en")
colophon.content = '''<html><body>
<div style="text-align:center; margin-top:50%;">
<p><em>This is a work of fiction. Names, characters, and incidents are products of the author's imagination.</em></p>
<p><em>First edition, 2026.</em></p>
</div>
</body></html>'''
colophon.add_item(style)
book.add_item(colophon)
spine.append(colophon)

# Set TOC and spine
book.toc = toc
book.spine = spine
book.add_item(epub.EpubNcx())
book.add_item(epub.EpubNav())

# Write
out_path = os.path.join(TYPESET, "falling-for-her.epub")
epub.write_epub(out_path, book)
print(f"\nEPUB written to {out_path}")
print(f"Size: {os.path.getsize(out_path):,} bytes")
