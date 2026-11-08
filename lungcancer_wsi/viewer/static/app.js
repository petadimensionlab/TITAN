/* TITAN WSI 統合ビューワ フロントエンド */
let SLIDES = [];
let viewer = null;
let current = null;          // slide record + info
let patchData = null;        // {coords, colors, patch_size} (PCA色)
let patchCluster = null;     // 組織クラスタ色 (mode=cluster 用)
let patchConch = null;       // CONCH zero-shot 色 (mode=conch 用)
let TISSUE = null;           // /api/tissue: クラスタ表 + スライド指標
let CONCH = null;            // /api/conch: クラス表
let overlayEl = null;
let labelColors = {};

const $ = (id) => document.getElementById(id);

init();

async function init() {
  const data = await (await fetch('/api/slides')).json();
  SLIDES = data.slides;
  $('counts').textContent =
    `${SLIDES.length} スライド (画像 ${SLIDES.filter(s=>s.has_image).length} / 結果 ${SLIDES.filter(s=>s.has_h5).length})`;

  try { TISSUE = await (await fetch('/api/tissue')).json(); } catch { TISSUE = null; }
  try { CONCH = await (await fetch('/api/conch')).json(); } catch { CONCH = null; }

  const cf = $('cohortFilter');
  cf.innerHTML = '<option value="">全コホート</option>' +
    data.cohorts.map(c=>`<option>${c}</option>`).join('');
  cf.onchange = renderList;
  $('search').oninput = renderList;
  renderList();
}

function renderList() {
  const q = $('search').value.toLowerCase();
  const c = $('cohortFilter').value;
  const list = SLIDES.filter(s =>
    (!c || s.cohort === c) &&
    (!q || s.name.toLowerCase().includes(q) || (s.label||'').toLowerCase().includes(q))
  );
  $('slideList').innerHTML = list.map(s => {
    const lab = s.label || '?';
    if (!(lab in labelColors)) {
      const palette = ['#4e79a7','#f28e2b','#e15759','#76b7b2','#59a14f','#edc948','#b07aa1','#ff9da7'];
      labelColors[lab] = palette[Object.keys(labelColors).length % palette.length];
    }
    const flags = [];
    if (!s.has_image) flags.push('⚠画像なし');
    if (!s.has_h5) flags.push('⚠結果なし');
    return `<div class="slideItem" data-id="${s.id}">
      <span class="dot" style="background:${labelColors[lab]}"></span>
      <div class="meta">
        <div class="nm" title="${s.name}">${s.name}</div>
        <div class="sub">${s.cohort} · ${lab} · ${s.n_patches||'-'} patches ${flags.join(' ')}</div>
      </div></div>`;
  }).join('');
  document.querySelectorAll('.slideItem').forEach(el =>
    el.onclick = () => selectSlide(+el.dataset.id));
}

async function selectSlide(id) {
  document.querySelectorAll('.slideItem').forEach(el =>
    el.classList.toggle('active', +el.dataset.id === id));
  const info = await (await fetch(`/api/slide/${id}`)).json();
  current = info;
  patchData = null;
  $('viewerPlaceholder').style.display = info.has_image ? 'none' : 'block';
  renderInfo(info);
  const rl = $('reportLink');
  rl.hidden = !info.has_report;
  rl.href = `/api/report/${encodeURIComponent(info.cohort)}`;
  await openSlide(info);
  $('togglePatches').checked = false;
  patchCluster = null;
  patchConch = null;
  atlasFiles = null;
  loadAtlas();
  if (info.has_h5) {
    patchData = await (await fetch(`/api/slide/${id}/patches`)).json();
    patchCluster = await (await fetch(`/api/slide/${id}/patches?mode=cluster`)).json();
    patchConch = await (await fetch(`/api/slide/${id}/patches?mode=conch`)).json();
    $('togglePatches').checked = true;
  }
  applyOverlay();
}

/* ---- アトラス (patch クローズアップ一覧) ---------------------------- */
let atlasFiles = null;  // {stem: manifest} を api_slides 時に読み込み

async function loadAtlas() {
  if (atlasFiles) return;
  try {
    const res = await fetch('/reports/atlas_manifest.json');
    if (res.ok) atlasFiles = await res.json();
  } catch (e) { /* 未生成なら非表示 */ }
  if (!atlasFiles) atlasFiles = {};
  if (current) renderInfo(current);
}

function atlasFor(info) {
  return atlasFiles && atlasFiles[info.key];
}

function renderAtlas(info) {
  const m = atlasFor(info);
  if (!m || !m.files || !m.files.length) return '';
  const groups = {};
  for (const f of m.files) (groups[f.cls_ja] ||= []).push(f);
  const opts = Object.entries(groups).map(([cls, arr]) =>
    `<optgroup label="${cls} (${arr.length})">` +
    arr.map((f, i) =>
      `<option value="${f.file}">#${f.rank + 1} ${(f.score * 100).toFixed(1)}% ` +
      `x=${f.x} y=${f.y}</option>`).join('') +
    '</optgroup>').join('');
  return `<h3>アトラス <span class="muted">(patch クローズアップ ${m.files.length} 点)</span></h3>
    <select id="atlasSel" style="width:100%;margin-bottom:6px">${opts}</select>
    <div class="muted" id="atlasMeta" style="margin-bottom:6px"></div>
    <img id="atlasImg" alt="" style="width:100%;border:1px solid #444;background:#fff">
    <a id="atlasDl" href="#" download style="font-size:12px">画像ダウンロード</a>
    <div class="muted" style="margin-top:4px">「patch 色分け: アトラス patch」で位置を確認できます</div>`;
}

function refreshAtlasButtons() {
  const sel = $('atlasSel');
  if (!sel || !current) return;
  const m = atlasFor(current);
  if (!m || !m.files) return;
  const go = (f) => {
    if (!f) return;
    $('atlasImg').src = `/reports/${encodeURIComponent(f.file)}`;
    $('atlasMeta').textContent =
      `${f.cls_ja}・patch #${f.patch_index}・中心 (${(f.x * m.magnification).toFixed(0)}, ` +
      `${(f.y * m.magnification).toFixed(0)}) px @10x・score ${(f.score * 100).toFixed(1)}%`;
    $('atlasDl').href = `/reports/${encodeURIComponent(f.file)}`;
  };
  const pick = () => {
    go(m.files.find((f) => f.file === sel.value));
    if ($('togglePatches').checked) applyOverlay();
  };
  sel.onchange = pick;
  if (!sel.value) sel.value = m.files[0].file;
  pick();
}

function openSlide(info) {
  if (viewer) { viewer.destroy(); viewer = null; overlayEl = null; }
  if (!info.has_image) return;
  const levels = info.levels;
  const w = levels[0].width, h = levels[0].height;
  const maxLevel = levels.length - 1;
  // OSD: level 0 = 最も縮小 / maxLevel = 原寸。server の level は downsample 指数なので反転
  const tsLevels = levels.map((l, i) => ({
    level: maxLevel - i,
    width: l.width,
    height: l.height,
    x: { min: 0, max: Math.max(0, Math.ceil(l.width / 256) - 1) },
    y: { min: 0, max: Math.max(0, Math.ceil(l.height / 256) - 1) },
  }));
  const ts = new OpenSeadragon.TileSource({
    width: w, height: h,
    tileWidth: 256, tileHeight: 256,
    minLevel: 0, maxLevel: maxLevel,
    levels: tsLevels,
    showNavigationControl: false,
  });
  ts.getTileUrl = (level, x, y) =>
    `/slide/${info.id}/tiles/${maxLevel - level}/${x}_${y}.png`;
  viewer = OpenSeadragon({ id: 'osd', showNavigationControl: false, tileSources: ts });
}

function renderInfo(info) {
  $('infoName').textContent = info.name;
  const rows = [];
  rows.push(['コホート', info.cohort]);
  if (info.label) rows.push(['label', info.label]);
  rows.push(['解析 status', info.status || '(なし)']);
  rows.push(['patch 数', info.n_patches || (patchData ? patchData.coords.length : '-')]);
  if (info.size_mb) rows.push(['ファイルサイズ', info.size_mb + ' MB']);
  if (info.seconds) rows.push(['解析時間', info.seconds + ' s']);
  if (info.mpp) rows.push(['MPP', info.mpp.toFixed(4) + ' µm/px']);
  if (info.levels) rows.push(['サイズ(L0)', `${info.levels[0].width} × ${info.levels[0].height}`]);
  if (info.has_pt) rows.push(['埋め込み (.pt)', 'あり']);
  if (info.has_h5) rows.push(['特徴量 (.h5)', 'あり']);
  rows.push(['ファイル', info.path]);
  if (info.error) rows.push(['error', info.error]);

  let predHtml = '';
  if (info.pred_scores) {
    predHtml = '<h3>zeroshot 予測</h3><div class="pred">';
    for (const part of info.pred_scores.split('|')) {
      const [k, v] = part.split(':');
      const p = parseFloat(v) * 100;
      const hi = k === info.pred;
      predHtml += `<div class="bar"><span class="pk${hi?' hi':''}">${k}</span>
        <span class="bt"><i style="width:${p.toFixed(1)}%"></i></span><span>${v}</span></div>`;
    }
    predHtml += '</div>';
  }

  let tissueHtml = '';
  if (info.tissue) {
    const t = info.tissue;
    const pc = v => v == null || v === '' ? '-' : (parseFloat(v) * 100).toFixed(1) + ' %';
    tissueHtml = '<h3>組織構成・免疫浸潤 <span class="muted">(暫定判定)</span></h3><table>';
    tissueHtml += `<tr><th>リンパ球濃厚</th><td>${pc(t.frac_lymphoid)}</td></tr>`;
    tissueHtml += `<tr><th>腫瘍/上皮</th><td>${pc(t.frac_tumor)}</td></tr>`;
    tissueHtml += `<tr><th>線維間質</th><td>${pc(t.frac_stroma)}</td></tr>`;
    tissueHtml += `<tr><th>肺胞腔/空白</th><td>${pc(t.frac_empty)}</td></tr>`;
    tissueHtml += `<tr><th>腫瘍 patch にリンパ球隣接</th><td>${pc(t.tumor_adj_lymph)}</td></tr>`;
    tissueHtml += `<tr><th>リンパ球→腫瘍 最短距離(中央値)</th><td>${t.lymph_to_tumor_med_um || '-'} µm</td></tr>`;
    tissueHtml += `<tr><th>浸潤パターン</th><td>${t.pattern || '-'}</td></tr>`;
    tissueHtml += '</table>';
  }

  let conchHtml = '';
  if (info.conch) {
    const t = info.conch;
    const pc = v => v == null || v === '' ? '-' : (parseFloat(v) * 100).toFixed(1) + ' %';
    conchHtml = '<h3>CONCH zero-shot <span class="muted">(テキスト照合)</span></h3><table>';
    conchHtml += `<tr><th>肺腺癌</th><td>${pc(t.frac_LUAD)}</td></tr>`;
    conchHtml += `<tr><th>扁平上皮癌</th><td>${pc(t.frac_LUSC)}</td></tr>`;
    conchHtml += `<tr><th>リンパ球/炎症</th><td>${pc(t.frac_LYMPH)}</td></tr>`;
    conchHtml += `<tr><th>線維間質</th><td>${pc(t.frac_STROMA)}</td></tr>`;
    conchHtml += `<tr><th>肺胞腔/背景</th><td>${pc(t.frac_ALV)}</td></tr>`;
    conchHtml += `<tr><th>壊死</th><td>${pc(t.frac_NECR)}</td></tr>`;
    conchHtml += `<tr><th>血液</th><td>${pc(t.frac_BLD)}</td></tr>`;
    conchHtml += `<tr><th>腫瘍 (LUAD+LUSC)</th><td>${pc(t.frac_tumor)}</td></tr>`;
    conchHtml += `<tr><th>腫瘍 patch にリンパ球隣接</th><td>${pc(t.tumor_adj_lymph)}</td></tr>`;
    conchHtml += `<tr><th>リンパ球→腫瘍 最短距離(中央値)</th><td>${t.lymph_to_tumor_med_um || '-'} µm</td></tr>`;
    conchHtml += `<tr><th>浸潤パターン</th><td>${t.pattern || '-'}</td></tr>`;
    conchHtml += `<tr><th>LUAD/LUSC 判別</th><td>LUAD p=${(t.luad_prob || 0).toFixed ? parseFloat(t.luad_prob).toFixed(2) : t.luad_prob}</td></tr>`;
    conchHtml += '</table>';
  }
  $('infoBody').innerHTML =
    '<table>' + rows.map(([k, v]) => `<tr><th>${k}</th><td>${v}</td></tr>`).join('') +
    '</table>' + tissueHtml + conchHtml + predHtml + renderAtlas(info);
  refreshAtlasButtons();
}

/* ---- パッチオーバーレイ ---------------------------------------------- */
function applyOverlay() {
  if (!viewer) return;
  if (overlayEl) { viewer.removeOverlay(overlayEl); overlayEl = null; }
  if (!$('togglePatches').checked || !patchData || !patchData.coords.length) return;

  const w = current.levels[0].width, h = current.levels[0].height;
  const sc = Math.min(1, 8192 / Math.max(w, h));
  const cv = document.createElement('canvas');
  cv.width = Math.ceil(w * sc);
  cv.height = Math.ceil(h * sc);
  const ctx = cv.getContext('2d');
  const mode = $('patchMode').value;
  let pd = patchData, ps = patchData ? patchData.patch_size * sc : 0;
  if (mode === 'cluster') {
    if (patchCluster && patchCluster.coords.length) pd = patchCluster;
    else pd = null; // 解析対象外: 何も描画しない (凡例に注記)
  } else if (mode === 'conch') {
    if (patchConch && patchConch.coords.length) pd = patchConch;
    else pd = null;
  } else if (mode === 'atlas') {
    const m = atlasFor(current);
    const sel = $('atlasSel');
    const f = m && sel && m.files.find((x) => x.file === sel.value);
    if (f) {
      ctx.strokeStyle = 'rgba(0,200,255,.95)';
      ctx.lineWidth = Math.max(2, sc * 4);
      ctx.strokeRect(f.x * sc, f.y * sc, f.size * sc, f.size * sc);
    }
    renderLegend(mode);
    overlayEl = cv;
    viewer.addOverlay({ element: cv, location: new OpenSeadragon.Rect(0, 0, w, h) });
    viewer.setOverlayOpacity(cv, 1);
    return;
  }
  if (pd) {
    ps = pd.patch_size * sc;
    pd.coords.forEach(([x, y], i) => {
      if (mode === 'outline') {
        ctx.strokeStyle = 'rgba(255,0,0,.9)';
        ctx.lineWidth = 1;
        ctx.strokeRect(x * sc, y * sc, ps, ps);
      } else {
        const [r, g, b] = pd.colors[i];
        ctx.fillStyle = `rgb(${r},${g},${b})`;
        ctx.fillRect(x * sc, y * sc, ps, ps);
      }
    });
  }
  renderLegend(mode);
  overlayEl = cv;
  viewer.addOverlay({
    element: cv,
    location: new OpenSeadragon.Rect(0, 0, w, h),
  });
  viewer.setOverlayOpacity(cv, $('opacity').value / 100);
}

function renderLegend(mode) {
  const lg = $('legend');
  if (mode === 'conch') {
    let html = '';
    if (patchConch && !patchConch.coords.length) {
      html = `<div class="lgNote">${patchConch.note || 'CONCH 解析の対象外です'}</div>`;
    } else if (CONCH && CONCH.classes && CONCH.classes.length && patchConch) {
      const tot = (patchConch.counts || []).reduce((a, b) => a + b, 0) || 1;
      html = CONCH.classes.map((c, i) => {
        const n = (patchConch.counts || [])[i] || 0;
        return `<span class="lgItem"><i style="background:rgb(${c.color.join(',')})"></i>` +
          `${c.ja} ${(100 * n / tot).toFixed(1)}% (${n})</span>`;
      }).join('');
    }
    lg.innerHTML = html;
    lg.hidden = !html;
    return;
  }
  if (mode === 'atlas') {
    const m = current && atlasFor(current);
    const sel = m && $('atlasSel');
    const f = m && sel && m.files.find((x) => x.file === sel.value);
    lg.innerHTML = f
      ? `<span class="lgItem"><i style="background:rgb(0,200,255)"></i>` +
        `${f.cls_ja} patch #${f.patch_index} (${(f.score * 100).toFixed(1)}%)</span>`
      : '<div class="lgNote">アトラス patch がありません</div>';
    lg.hidden = false;
    return;
  }
  if (mode !== 'cluster' || !current) { lg.hidden = true; return; }
  let html = '';
  if (patchCluster && !patchCluster.coords.length) {
    html = `<div class="lgNote">${patchCluster.note || '組織クラスタ解析の対象外です'}</div>`;
  } else if (TISSUE && TISSUE.clusters && TISSUE.clusters.length) {
    html = TISSUE.clusters.map(c =>
      `<span class="lgItem"><i style="background:rgb(${c.color.join(',')})"></i>` +
      `C${c.cluster} ${c.cls_ja} (${c.n})</span>`).join('');
  }
  lg.innerHTML = html;
  lg.hidden = !html;
}

$('togglePatches').onchange = applyOverlay;
$('patchMode').onchange = applyOverlay;
$('opacity').oninput = () => { if (overlayEl) viewer.setOverlayOpacity(overlayEl, $('opacity').value / 100); };
$('fit').onclick = () => viewer && viewer.viewport.goHome();
