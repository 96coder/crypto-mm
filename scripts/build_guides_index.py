#!/usr/bin/env python3
"""
Scan guides and build an index.json with title, slug, path, description.
Looks for markdown under ./guide and ./guides. Writes index.json into the same folder.
"""
import os, json, re

def extract_meta(md_text):
    title = None
    desc = None
    for line in md_text.splitlines():
        if not title:
            m = re.match(r"^#\s+(.+)$", line.strip())
            if m:
                title = m.group(1).strip()
                continue
        if not desc and line.strip() and not line.strip().startswith('#'):
            desc = line.strip()
        if title and desc:
            break
    return title, desc

def build(folder):
    base = folder
    items = []
    if not os.path.isdir(base):
        return None
    for root, _, files in os.walk(base):
        for f in files:
            if not f.lower().endswith('.md'): continue
            p = os.path.join(root, f)
            rel = os.path.relpath(p, base)
            with open(p, 'r', encoding='utf-8', errors='ignore') as fh:
                text = fh.read()
            title, desc = extract_meta(text)
            slug = re.sub(r"[^a-z0-9]+", '-', (title or rel).lower()).strip('-')
            items.append({
                'title': title or rel,
                'slug': slug,
                'path': rel.replace('\\','/'),
                'description': desc or ''
            })
    items.sort(key=lambda x: x['title'].lower())
    with open(os.path.join(base, 'index.json'), 'w', encoding='utf-8') as out:
        json.dump(items, out, ensure_ascii=False, indent=2)
    return base

if __name__ == '__main__':
    for folder in ['guide', 'guides']:
        out = build(folder)
        if out:
            print(f"Wrote {out}/index.json")

