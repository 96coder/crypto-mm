// Guides loader + simple Markdown renderer

async function fetchManifest() {
  // Try guide/index.json then guides/index.json
  const paths = ['guide/index.json', 'guides/index.json'];
  for (const p of paths) {
    try {
      const res = await fetch(p, { cache: 'no-store' });
      if (res.ok) return { base: p.split('/')[0], items: await res.json() };
    } catch {}
  }
  return { base: null, items: [] };
}

function escapeHtml(s) { return s.replace(/[&<>]/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;'}[c])); }

function renderMarkdown(md) {
  // Minimal renderer: headings, code fences, lists, inline code, bold/italic, links, paragraphs, blockquotes
  const lines = md.replace(/\r\n?/g, '\n').split('\n');
  const out = [];
  let inCode = false; let codeLang = '';
  for (let i = 0; i < lines.length; i++) {
    let line = lines[i];
    const fence = line.match(/^```(.*)$/);
    if (fence) {
      if (!inCode) { inCode = true; codeLang = fence[1].trim(); out.push(`<pre><code class="lang-${codeLang}">`); }
      else { inCode = false; codeLang = ''; out.push(`</code></pre>`); }
      continue;
    }
    if (inCode) { out.push(escapeHtml(line)); continue; }
    if (/^\s*$/.test(line)) { out.push(''); continue; }
    const h = line.match(/^(#{1,6})\s+(.*)$/);
    if (h) { const lvl = h[1].length; const id = h[2].toLowerCase().replace(/[^a-z0-9]+/g,'-'); out.push(`<h${lvl} id="${id}">${h[2]}</h${lvl}>`); continue; }
    const bq = line.match(/^>\s?(.*)$/);
    if (bq) { out.push(`<blockquote>${bq[1]}</blockquote>`); continue; }
    const li = line.match(/^\s*[-*+]\s+(.*)$/);
    if (li) {
      // gather following list lines
      const items = [li[1]];
      while (i + 1 < lines.length && /^\s*[-*+]\s+/.test(lines[i+1])) { items.push(lines[++i].replace(/^\s*[-*+]\s+/, '')); }
      out.push('<ul>' + items.map(s=>`<li>${s}</li>`).join('') + '</ul>');
      continue;
    }
    const oli = line.match(/^\s*\d+\.\s+(.*)$/);
    if (oli) {
      const items = [oli[1]];
      while (i + 1 < lines.length && /^\s*\d+\.\s+/.test(lines[i+1])) { items.push(lines[++i].replace(/^\s*\d+\.\s+/, '')); }
      out.push('<ol>' + items.map(s=>`<li>${s}</li>`).join('') + '</ol>');
      continue;
    }
    // inline: code, bold, italic, links
    let html = escapeHtml(line)
      .replace(/\*\*(.+?)\*\*/g, '<strong>$1</strong>')
      .replace(/\*(.+?)\*/g, '<em>$1</em>')
      .replace(/`([^`]+)`/g, '<code>$1</code>')
      .replace(/\[(.+?)\]\((.+?)\)/g, '<a href="$2" target="_blank" rel="noopener noreferrer">$1</a>');
    out.push(`<p>${html}</p>`);
  }
  return out.join('\n');
}

function buildToc(container) {
  const hs = container.querySelectorAll('h1, h2, h3');
  const ul = document.createElement('ul');
  hs.forEach(h => {
    const a = document.createElement('a'); a.textContent = h.textContent; a.href = `#${h.id}`;
    const li = document.createElement('li'); li.appendChild(a); ul.appendChild(li);
  });
  return ul;
}

async function loadGuide(base, entry) {
  const body = document.getElementById('guideBody');
  const title = document.getElementById('guideTitle');
  const meta = document.getElementById('guideMeta');
  const toc = document.getElementById('guideToc');
  body.innerHTML = '<p class="guide-meta">Loading…</p>';
  try {
    const res = await fetch(`${base}/${entry.path}`, { cache: 'no-store' });
    const text = await res.text();
    title.textContent = entry.title || entry.path;
    meta.textContent = entry.description || '';
    body.innerHTML = renderMarkdown(text);
    toc.innerHTML = '';
    toc.appendChild(buildToc(body));
    history.replaceState({}, '', `?g=${encodeURIComponent(entry.slug || entry.path)}`);
    window.scrollTo({ top: 0, behavior: 'smooth' });
  } catch (e) {
    body.innerHTML = '<p class="guide-meta">Failed to load guide.</p>';
  }
}

function applySearch(listEl, items) {
  const q = document.getElementById('guideSearch');
  function render(filter) {
    listEl.innerHTML = '';
    items.filter(it => !filter || (it.title||'').toLowerCase().includes(filter) || (it.description||'').toLowerCase().includes(filter)).forEach(it => {
      const li = document.createElement('li');
      const btn = document.createElement('button');
      btn.textContent = it.title || it.path;
      btn.addEventListener('click', () => loadGuide(window.__guidesBase, it));
      li.appendChild(btn);
      listEl.appendChild(li);
    });
  }
  q.addEventListener('input', () => render(q.value.trim().toLowerCase()));
  render('');
}

(async () => {
  const { base, items } = await fetchManifest();
  window.__guidesBase = base || 'guide';
  const listEl = document.getElementById('guideList');
  applySearch(listEl, items);
  // Auto-open guide if query param present
  const params = new URLSearchParams(location.search);
  const slug = params.get('g');
  if (slug) {
    const entry = items.find(x => (x.slug || x.path) === slug) || items[0];
    if (entry) loadGuide(window.__guidesBase, entry);
  } else if (items[0]) {
    loadGuide(window.__guidesBase, items[0]);
  } else {
    // Fallback: try default README/index markdown if no index.json exists
    const candidates = [
      { base: 'guide', path: 'README.md', title: 'Getting Started' },
      { base: 'guide', path: 'index.md', title: 'Getting Started' },
      { base: 'guides', path: 'README.md', title: 'Getting Started' },
      { base: 'guides', path: 'index.md', title: 'Getting Started' },
    ];
    let loaded = false;
    for (const c of candidates) {
      try {
        const res = await fetch(`${c.base}/${c.path}`, { cache: 'no-store' });
        if (res.ok) {
          window.__guidesBase = c.base;
          await loadGuide(c.base, { title: c.title, path: c.path, description: '' });
          loaded = true;
          break;
        }
      } catch {}
    }
    if (!loaded) {
      const body = document.getElementById('guideBody');
      const title = document.getElementById('guideTitle');
      const meta = document.getElementById('guideMeta');
      title.textContent = 'No guides found';
      meta.textContent = '';
      body.innerHTML = `<p class="guide-meta">No guide index detected. To enable the guides sidebar, add markdown files under <code>guide/</code> or <code>guides/</code> and build an index:</p>
<pre><code>python3 scripts/build_guides_index.py</code></pre>
<p class="guide-meta">If you’re opening the site directly from your filesystem, please serve over HTTP to allow fetch():</p>
<pre><code>python3 -m http.server 5173
open http://localhost:5173/guides.html</code></pre>`;
    }
  }
})();
