<!doctype html>
<html lang="tr">
<head>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width,initial-scale=1">
    <title>OtoKantar V7 | Canli Izleme</title>
    <style>
        @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;600&family=Syne:wght@400;600;700&display=swap');
        *{box-sizing:border-box;margin:0;padding:0}body{min-height:100vh;background:radial-gradient(circle at top right,rgba(33,209,159,.09),transparent 24%),linear-gradient(160deg,#0d1015,#151b25);color:#eef3f8;font:14px 'Syne',sans-serif}
        :root{--bg:#111722;--card:rgba(24,31,43,.92);--line:rgba(255,255,255,.08);--line2:rgba(255,255,255,.15);--text:#eef3f8;--muted:#94a0b1;--dim:#5d6877;--acc:#21d19f;--acc2:rgba(33,209,159,.12);--warn:#f3b341;--warn2:rgba(243,179,65,.12);--danger:#ff6178;--danger2:rgba(255,97,120,.12);--mono:'JetBrains Mono',monospace}
        header{position:sticky;top:0;z-index:20;display:flex;justify-content:space-between;align-items:center;gap:1rem;padding:1.1rem 1.4rem;background:rgba(10,14,20,.86);backdrop-filter:blur(14px);border-bottom:1px solid var(--line)}
        .brand{display:flex;align-items:center;gap:.9rem}.mark{width:38px;height:38px;border-radius:12px;display:grid;place-items:center;background:linear-gradient(135deg,rgba(33,209,159,.25),rgba(62,133,255,.18));border:1px solid rgba(33,209,159,.35)}.mark svg{width:18px;height:18px}.brand h1{font-size:1rem;letter-spacing:.08em;text-transform:uppercase}.brand p,.clock,.sub,.mini{font-family:var(--mono);color:var(--muted)}.brand p{font-size:.73rem;margin-top:.14rem}
        .pill{display:inline-flex;align-items:center;gap:.55rem;padding:.42rem .82rem;border-radius:999px;border:1px solid rgba(33,209,159,.3);background:var(--acc2);color:var(--acc);font:11px var(--mono);letter-spacing:.08em;text-transform:uppercase}.dot{width:.45rem;height:.45rem;border-radius:50%;background:currentColor}
        .side{display:flex;align-items:center;gap:.9rem}.clock{font-size:.76rem}
        main{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:1rem;padding:1.2rem}.card{background:var(--card);border:1px solid var(--line);border-radius:18px;padding:1rem;box-shadow:0 20px 48px rgba(0,0,0,.24)}.hero{grid-column:1/-1;display:flex;justify-content:space-between;gap:1rem;flex-wrap:wrap;background:linear-gradient(135deg,rgba(24,31,43,.96),rgba(18,25,35,.94));border-color:rgba(33,209,159,.16)}
        .view-switch{display:flex;gap:.6rem;padding:1rem 1.2rem 0 1.2rem;flex-wrap:wrap}.view-btn{padding:.55rem .95rem;border-radius:999px;border:1px solid var(--line2);background:rgba(255,255,255,.02);color:var(--muted);font:11px var(--mono);letter-spacing:.08em;text-transform:uppercase;cursor:pointer}.view-btn.active{background:var(--acc2);border-color:rgba(33,209,159,.3);color:var(--acc)}.tab-panel{display:none}.tab-panel.active{display:grid}
        .grow{flex:1 1 320px}.eyebrow,.title,.info-k{font-size:.67rem;letter-spacing:.14em;text-transform:uppercase;color:var(--muted)}.weight{font:700 clamp(2.1rem,5vw,3.4rem) var(--mono);line-height:.95;color:var(--acc);margin-top:.45rem}.status-text{margin-top:.55rem}.hero-grid{display:grid;gap:.7rem;min-width:240px}.panel{padding:.8rem .9rem;border:1px solid var(--line);border-radius:14px;background:rgba(255,255,255,.02)}.panel .v,.metric-v{font:600 1.25rem var(--mono);margin-top:.35rem}
        .metric-v.acc{color:var(--acc)}.metric-v.warn{color:var(--warn)}.metric-s{margin-top:.35rem;font:11px var(--mono);color:var(--muted)}.bar{margin-top:.7rem;height:4px;background:rgba(255,255,255,.08);border-radius:99px;overflow:hidden}.bar>span{display:block;height:100%;width:0;background:linear-gradient(90deg,var(--acc),var(--warn));transition:width 1s linear}
        .span2{grid-column:span 2}.span4{grid-column:1/-1}.frame{width:100%;aspect-ratio:16/9;object-fit:cover;border-radius:14px;background:#091018;border:1px solid var(--line);display:block}.note{margin-top:.65rem;font:11px var(--mono);color:var(--muted)}
        .plate{margin-top:.85rem;min-height:114px;border-radius:16px;display:grid;place-items:center;border:1px solid rgba(33,209,159,.18);background:linear-gradient(135deg,rgba(33,209,159,.08),rgba(255,255,255,.02))}.plate b{font:600 clamp(1.7rem,4vw,2.5rem) var(--mono);letter-spacing:.18em;color:var(--acc)}.plate .empty{color:var(--dim);font-size:1rem;letter-spacing:.12em}.flash{outline:2px solid rgba(33,209,159,.34)}
        .track{display:flex;gap:.45rem;margin-top:.9rem}.seg{flex:1;height:5px;border-radius:99px;background:rgba(255,255,255,.08)}.seg.wait{background:var(--warn)}.seg.on{background:var(--acc)}.verify{margin-top:.55rem;text-align:center;font:11px var(--mono);color:var(--muted)}
        .btns{display:flex;gap:.65rem;flex-wrap:wrap;margin-top:.95rem}.btn{flex:1 1 140px;display:inline-flex;justify-content:center;align-items:center;padding:.78rem .9rem;border-radius:12px;border:1px solid var(--line2);background:transparent;color:var(--muted);text-decoration:none;cursor:pointer;font:12px 'Syne',sans-serif}.btn:hover{background:rgba(255,255,255,.04);color:var(--text)}.btn.primary{background:var(--acc2);border-color:rgba(33,209,159,.28);color:var(--acc)}
        .head{display:flex;justify-content:space-between;align-items:center;gap:.7rem;margin-bottom:.85rem}.badge{padding:.18rem .58rem;border-radius:999px;border:1px solid var(--line);background:rgba(255,255,255,.03);font:10px var(--mono);color:var(--muted)}
        .log{height:210px;overflow:auto;border:1px solid var(--line);border-radius:14px;background:rgba(6,10,16,.42);padding:.7rem;display:flex;flex-direction:column;gap:.42rem}.row{display:grid;grid-template-columns:58px 48px 1fr;gap:.55rem;font:11px var(--mono)}.time{color:var(--dim)}.info{color:var(--acc)}.warn{color:var(--warn)}.error{color:var(--danger)}.msg{color:var(--muted)}
        .mini-chart{height:118px;display:flex;align-items:flex-end;gap:.35rem;margin-top:.8rem;padding-top:.7rem;border-top:1px solid var(--line)}.col{flex:1;display:flex;flex-direction:column;align-items:center;justify-content:flex-end;gap:.25rem;height:100%}.count,.label{font:10px var(--mono)}.count{color:var(--muted)}.label{color:var(--dim)}.stick{width:100%;min-height:4px;border-radius:8px 8px 0 0;background:rgba(33,209,159,.18);border-top:1px solid rgba(33,209,159,.28)}.stick.now{background:rgba(33,209,159,.28);border-color:var(--acc)}
        .info-grid{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:.8rem 1rem}.info-v{font:11px var(--mono);color:var(--muted);margin-top:.28rem}
        table{width:100%;border-collapse:collapse}th{padding:.45rem .72rem;text-align:left;font-size:10px;letter-spacing:.14em;text-transform:uppercase;color:var(--dim);border-bottom:1px solid var(--line)}td{padding:.78rem .72rem;border-bottom:1px solid rgba(255,255,255,.05);font:11px var(--mono);color:var(--muted)}tr:last-child td{border-bottom:none}.plate-td{color:var(--text);font-size:12px;letter-spacing:.08em;font-weight:600}
        .tag{display:inline-flex;align-items:center;padding:.2rem .55rem;border-radius:999px;font-size:10px;font-weight:600;letter-spacing:.08em}.giris{background:var(--acc2);color:var(--acc);border:1px solid rgba(33,209,159,.24)}.cikis{background:var(--warn2);color:var(--warn);border:1px solid rgba(243,179,65,.22)}.alarm{background:var(--danger2);color:var(--danger);border:1px solid rgba(255,97,120,.22)}
        .conf{display:flex;align-items:center;gap:.42rem}.conf-track{width:68px;height:4px;border-radius:99px;background:rgba(255,255,255,.08);overflow:hidden}.conf-fill{height:100%;background:var(--acc)}.conf-fill.mid{background:var(--warn)}.conf-fill.low{background:var(--danger)}.empty-row{padding:1.2rem .72rem;color:var(--dim)}
        @media(max-width:1080px){main{grid-template-columns:repeat(2,minmax(0,1fr))}.span4,.hero{grid-column:1/-1}.span2{grid-column:span 2}}
        @media(max-width:760px){header{flex-direction:column;align-items:flex-start}main{grid-template-columns:1fr;padding:1rem}.span2,.span4{grid-column:span 1}.info-grid{grid-template-columns:1fr}.row{grid-template-columns:54px 44px 1fr}}
    </style>
</head>
<body>
<header>
    <div class="brand">
        <div class="mark">
            <svg viewBox="0 0 16 16" fill="none" xmlns="http://www.w3.org/2000/svg">
                <rect x="1.1" y="5" width="13.8" height="6.8" rx="1.4" stroke="#21d19f" stroke-width="1.2"/>
                <circle cx="4.2" cy="11.3" r="1.3" fill="#21d19f"/>
                <circle cx="11.8" cy="11.3" r="1.3" fill="#21d19f"/>
                <path d="M4.4 5V3.7C4.4 2.76 5.16 2 6.1 2H9.9C10.84 2 11.6 2.76 11.6 3.7V5" stroke="#21d19f" stroke-width="1.1"/>
            </svg>
        </div>
        <div><h1>OtoKantar</h1><p>Canli panel | MySQL + JSON durum + JPG</p></div>
    </div>
    <div class="side">
        <div class="pill" id="pill"><span class="dot"></span><span id="pill-text">BEKLENIYOR</span></div>
        <div class="clock" id="clock">--:--:--</div>
    </div>
</header>

<div class="view-switch">
    <button class="view-btn active" type="button" data-tab="genel">Genel</button>
    <button class="view-btn" type="button" data-tab="canli">Canli</button>
    <button class="view-btn" type="button" data-tab="kayitlar">Kayitlar</button>
</div>

<main class="tab-panel active" data-panel="genel">
    <section class="card hero">
        <div class="grow">
            <div class="eyebrow">Anlik kantar</div>
            <div class="weight" id="kg">--</div>
            <div class="metric-s status-text" id="kg-status">Agirlik verisi bekleniyor (/canli/api)</div>
            <div class="bar"><span id="stale-bar"></span></div>
        </div>
        <div class="hero-grid">
            <div class="panel"><div class="title">Plaka tampon</div><div class="v" id="buffer">--</div></div>
            <div class="panel"><div class="title">Canli veri yasi</div><div class="v" id="fresh">--</div><div class="metric-s" id="fresh-sub">guncelleme bekleniyor</div></div>
        </div>
    </section>

    <section class="card">
        <div class="title">Bugun kayit</div>
        <div class="metric-v acc" id="m1">0</div>
        <div class="metric-s" id="m1s">MySQL tabanli gunluk toplam</div>
    </section>
    <section class="card">
        <div class="title">Aktif seans</div>
        <div class="metric-v" id="m2">0</div>
        <div class="metric-s" id="m2s">tamamlanan seans: 0</div>
    </section>
    <section class="card">
        <div class="title">Ortalama guven</div>
        <div class="metric-v warn" id="m3">--</div>
        <div class="metric-s">% OCR skoru</div>
    </section>
    <section class="card">
        <div class="title">Son 1 saat</div>
        <div class="metric-v" id="m4">0</div>
        <div class="metric-s">kayit hareketi</div>
    </section>
</main>

<main class="tab-panel" data-panel="canli">
    <section class="card span2">
        <div class="head"><div class="title">Canli kare</div><div class="badge">/canli/kare</div></div>
        <img class="frame" id="cam" src="" alt="Canli kare">
        <div class="note" id="cam-note">Kamera karesi bekleniyor...</div>
    </section>

    <section class="card span2">
        <div class="head"><div class="title">Aktif tespit</div><div class="badge">dogrulama</div></div>
        <div class="plate" id="plate"><b class="empty">BEKLENIYOR</b></div>
        <div class="track"><div class="seg" id="vd1"></div><div class="seg" id="vd2"></div><div class="seg" id="vd3"></div><div class="seg" id="vd4"></div></div>
        <div class="verify" id="verify">Dogrulama bekleniyor</div>
        <div class="btns">
            <button class="btn" id="refresh" type="button">Simdi yenile</button>
            <button class="btn primary" id="demo" type="button">Demo modu</button>
            <a class="btn" href="{{ route('canli.csv') }}" id="csv">CSV indir</a>
        </div>
    </section>

    <section class="card span2">
        <div class="head"><div class="title">Panel olay akisi</div><div class="badge" id="log-count">0 satir</div></div>
        <div class="log" id="log"></div>
    </section>
</main>

<main class="tab-panel" data-panel="kayitlar">
    <section class="card span2">
        <div class="head"><div class="title">Sistem bilgisi</div><div class="badge">uretim paneli</div></div>
        <div class="info-grid">
            <div><div class="info-k">AI yigin</div><div class="info-v" id="ai">YOLOv8 + OCR</div></div>
            <div><div class="info-k">Calisma modu</div><div class="info-v" id="mode">Bekleniyor</div></div>
            <div><div class="info-k">Esik</div><div class="info-v" id="esik">4 / canli durum dosyasi</div></div>
            <div><div class="info-k">Rapor</div><div class="info-v">MySQL / otokantar</div></div>
            <div><div class="info-k">OCR kare atlama</div><div class="info-v" id="ocr">Dinamik</div></div>
            <div><div class="info-k">Mimari</div><div class="info-v" id="arch">MySQL + JSON + JPG</div></div>
        </div>
        <div class="mini-chart" id="chart"></div>
    </section>
    <section class="card span4">
        <div class="head"><div class="title">Son kayitlar</div><div class="badge" id="table-count">0 kayit</div></div>
        <table>
            <thead><tr><th>Plaka</th><th>Agirlik / Net</th><th>Tarih</th><th>Saat</th><th>Tip</th><th>Guven</th></tr></thead>
            <tbody id="tbody"><tr><td class="empty-row" colspan="6">Henuz kayit yok. Sistem dosya akisina baglanmayi bekliyor.</td></tr></tbody>
        </table>
    </section>
</main>

<script>
const Config = {
  plates: ['06ABC123', '34TR574', '35ZK882', '16BRS61', '41KLM99', '27FRT20', '06ANK80', '34ED5728', '24TR123', '79SAA001'],
  pollMs: 1000,
  camMs: 1400,
  verifyThreshold: 4,
  maxLog: 80,
  tableLimit: 15,
  chartHours: 12,
};

const State = {
  records: [],
  bars: new Array(Config.chartHours).fill(0),
  total: 0,
  lastSignature: '',
  status: 'offline',
  lastUpdateMs: null,
  demoOn: false,
  demoPlate: null,
  demoStep: 0,
  intervals: { poll: null, cam: null, demo: null },
};

const Utils = {
  el(id) { return document.getElementById(id); },
  now() { return new Date().toLocaleTimeString('tr-TR', { hour: '2-digit', minute: '2-digit', second: '2-digit' }); },
  toNum(value) { return Number.isFinite(Number(value)) ? Number(value) : null; },
  kg(value) { const n = this.toNum(value); return n === null ? '--' : n.toLocaleString('tr-TR', { maximumFractionDigits: 1 }); },
  escapeHtml(value) {
    return String(value ?? '')
      .replaceAll('&', '&amp;')
      .replaceAll('<', '&lt;')
      .replaceAll('>', '&gt;')
      .replaceAll('"', '&quot;')
      .replaceAll("'", '&#39;');
  },
  recordTs(record) {
    const v = Date.parse(`${record.giris_tarih || ''}T${record.giris_saat || ''}`);
    return Number.isFinite(v) ? v : null;
  },
  normalizeRecord(record = {}) {
    const raw = String(record.durum || record.tip || 'ICERIDE').toUpperCase();
    const tip = raw.includes('TAMAMLANDI') || raw.includes('CIKIS')
      ? 'CIKIS'
      : raw.includes('KARA') || raw.includes('ALARM')
        ? 'ALARM'
        : 'GIRIS';
    return {
      plaka: String(record.plaka || '').trim(),
      tip,
      durum: raw,
      giris_tarih: String(record.giris_tarih || record.tarih || '').trim(),
      giris_saat: String(record.giris_saat || record.saat || '').trim(),
      giris_agirlik: this.toNum(record.giris_agirlik),
      cikis_tarih: String(record.cikis_tarih || '').trim(),
      cikis_saat: String(record.cikis_saat || '').trim(),
      cikis_agirlik: this.toNum(record.cikis_agirlik),
      net_agirlik: this.toNum(record.net_agirlik),
      guven: this.toNum(record.guven) || 0,
    };
  },
  recordStamp(record) { if (!record || !record.plaka) return ''; return [record.plaka, record.giris_tarih, record.giris_saat, record.tip].join('|'); },
};

const UI = {
  log(level, message) {
    const box = Utils.el('log');
    const row = document.createElement('div');
    row.className = 'row';
    row.innerHTML = `<span class="time">${Utils.now()}</span><span class="${Utils.escapeHtml(level)}">${Utils.escapeHtml(level.toUpperCase())}</span><span class="msg">${Utils.escapeHtml(message)}</span>`;
    box.appendChild(row);
    while (box.children.length > Config.maxLog) box.removeChild(box.firstChild);
    box.scrollTop = box.scrollHeight;
    Utils.el('log-count').textContent = `${box.children.length} satir`;
  },
  setStatus(next) {
    State.status = next;
    const map = {
      live: ['CANLI', 'var(--acc)', 'rgba(33,209,159,.12)', 'rgba(33,209,159,.3)'],
      stale: ['YAVAS', 'var(--warn)', 'rgba(243,179,65,.12)', 'rgba(243,179,65,.3)'],
      offline: ['BEKLENIYOR', 'var(--muted)', 'rgba(255,255,255,.04)', 'rgba(255,255,255,.12)'],
      demo: ['DEMO', 'var(--acc)', 'rgba(33,209,159,.18)', 'rgba(33,209,159,.36)'],
    };
    const s = map[next] || map.offline;
    const pill = Utils.el('pill');
    pill.style.color = s[1];
    pill.style.background = s[2];
    pill.style.borderColor = s[3];
    Utils.el('pill-text').textContent = s[0];
  },
  updateFresh(seconds) {
    const n = Utils.toNum(seconds);
    if (n === null) {
      Utils.el('fresh').textContent = '--';
      Utils.el('fresh-sub').textContent = 'guncelleme bekleniyor';
      Utils.el('stale-bar').style.width = '0%';
      State.lastUpdateMs = null;
      return;
    }
    State.lastUpdateMs = Date.now() - n * 1000;
    Utils.el('fresh').textContent = `${Math.floor(n)} sn`;
    Utils.el('fresh-sub').textContent = n <= 2 ? 'dosya yeni guncellendi' : 'son canli durum yazimi';
    Utils.el('stale-bar').style.width = `${Math.min(100, (n / 15) * 100)}%`;
  },
  resetPlate() {
    Utils.el('plate').className = 'plate';
    Utils.el('plate').innerHTML = '<b class="empty">BEKLENIYOR</b>';
    Utils.el('verify').textContent = 'Dogrulama bekleniyor';
    ['vd1', 'vd2', 'vd3', 'vd4'].forEach((id) => (Utils.el(id).className = 'seg'));
  },
  showPlate(text, step, done, msg) {
    Utils.el('plate').className = `plate${done ? ' flash' : ''}`;
    Utils.el('plate').innerHTML = `<b>${Utils.escapeHtml(text)}</b>`;
    ['vd1', 'vd2', 'vd3', 'vd4'].forEach((id, i) => {
      const el = Utils.el(id);
      el.className = 'seg';
      if (i < step && step < Config.verifyThreshold) el.classList.add('wait');
      else if (i < step) el.classList.add('on');
    });
    Utils.el('verify').textContent = msg || (done ? 'Kayit tamamlandi' : `Dogrulaniyor: ${step} / ${Config.verifyThreshold}`);
  },
  drawTable() {
    Utils.el('table-count').textContent = `${State.total} kayit`;
    if (!State.records.length) {
      Utils.el('tbody').innerHTML = '<tr><td class="empty-row" colspan="6">Henuz kayit yok. Sistem dosya akisina baglanmayi bekliyor.</td></tr>';
      return;
    }
    Utils.el('tbody').innerHTML = State.records.slice(0, Config.tableLimit).map((r) => {
      const p = Math.max(0, Math.min(100, Math.round((r.guven || 0) * 100)));
      const cls = p >= 80 ? '' : (p >= 60 ? ' mid' : ' low');
      const tag = r.tip === 'CIKIS' ? 'cikis' : (r.tip === 'ALARM' ? 'alarm' : 'giris');
      const weight = r.tip === 'CIKIS' ? (r.net_agirlik ?? r.cikis_agirlik ?? r.giris_agirlik) : r.giris_agirlik;
      const date = r.tip === 'CIKIS' && r.cikis_tarih ? r.cikis_tarih : r.giris_tarih;
      const time = r.tip === 'CIKIS' && r.cikis_saat ? r.cikis_saat : r.giris_saat;
      return `<tr><td class="plate-td">${Utils.escapeHtml(r.plaka)}</td><td>${Utils.escapeHtml(Utils.kg(weight))}</td><td>${Utils.escapeHtml(date || '--')}</td><td>${Utils.escapeHtml(time || '--')}</td><td><span class="tag ${tag}">${Utils.escapeHtml(r.tip)}</span></td><td><div class="conf"><div class="conf-track"><div class="conf-fill${cls}" style="width:${p}%"></div></div><span>%${p}</span></div></td></tr>`;
    }).join('');
  },
  drawChart() {
    const max = Math.max(...State.bars, 1);
    const hour = new Date().getHours();
    Utils.el('chart').innerHTML = State.bars.map((v, i) => (
      `<div class="col"><span class="count">${v || ''}</span><div class="stick ${i === 11 ? 'now' : ''}" style="height:${Math.max(4, Math.round((v / max) * 92))}px"></div><span class="label">${String(((hour - 11 + i) + 24) % 24).padStart(2, '0')}</span></div>`
    )).join('');
  },
  setInfo(durum) {
    const s = durum?.sistem || {};
    const fallback = s.ocr_fallback ? ` / ${s.ocr_fallback}` : '';
    Utils.el('ai').textContent = `YOLOv8 + ${s.ocr_backend || 'OCR'}${fallback}`;
    Utils.el('mode').textContent = s.simulasyon_modu ? 'SIMULASYON' : 'CANLI';
    Utils.el('ocr').textContent = s.ocr_kare_atlama ? `Her ${s.ocr_kare_atlama}. kare` : 'Dinamik';
    Utils.el('arch').textContent = s?.mimari || 'MySQL + JSON + JPG';
  },
  setMetrics(summary, durum) {
    Utils.el('m1').textContent = String(Number(summary?.bugun_kayit ?? 0));
    Utils.el('m1s').textContent = `${Number(summary?.son_saat_kayit ?? 0)} kayit son 1 saatte`;
    Utils.el('m2').textContent = String(Number(summary?.aktif_seans ?? 0));
    Utils.el('m2s').textContent = `tamamlanan seans: ${Number(summary?.tamamlanan ?? 0)}`;
    Utils.el('m3').textContent = Utils.toNum(summary?.ortalama_guven_yuzde) === null ? '--' : `%${Number(summary.ortalama_guven_yuzde)}`;
    Utils.el('m4').textContent = String(Number(summary?.son_saat_kayit ?? 0));
    this.updateFresh(durum?._durum_yasi_saniye ?? null);
  },
  setScale(durum) {
    const k = Utils.toNum(durum?.kantar_kg);
    const buffer = durum?.plaka_buffer_detay?.plaka || durum?.plaka_buffer || '';
    Utils.el('buffer').textContent = buffer || '--';
    if (k === null) {
      Utils.el('kg').textContent = '--';
      Utils.el('kg-status').textContent = 'Kantar verisi yok. Python sureci veya COM akisi bekleniyor.';
      return;
    }
    Utils.el('kg').textContent = `${Utils.kg(k)} kg`;
    Utils.el('kg-status').textContent = durum?.seans_kilitli
      ? 'Seans kilitli. Arac cikisi ve sifirlama guardi bekleniyor.'
      : (durum?.kantar_sabit ? 'Olcu sabit. Kantar karar vermeye hazir.' : 'Olcu degisiyor. Kantarin sabitlenmesi bekleniyor.');
  },
  refreshCam() {
    if (State.demoOn) return;
    const img = Utils.el('cam');
    const note = Utils.el('cam-note');
    let retry = false;
    img.onerror = () => {
      if (!retry) {
        retry = true;
        img.src = `/canli/kare?t=${Date.now()}`;
        return;
      }
      note.textContent = 'Canli kare yok. Otokantar calisinca canli_kare.jpg guncellenecek.';
    };
    img.onload = () => {
      retry = false;
      note.textContent = 'Canli kare yenilendi. Tartidaki arac goruntusu izleniyor.';
    };
    img.src = `/canli/kare?t=${Date.now()}`;
  },
};

const Store = {
  calcBars() {
    State.bars = new Array(Config.chartHours).fill(0);
    const nowMs = Date.now();
    State.records.forEach((record) => {
      const ts = Utils.recordTs(record);
      if (ts === null) return;
      const h = Math.floor((nowMs - ts) / 3600000);
      if (h >= 0 && h < Config.chartHours) State.bars[11 - h] += 1;
    });
  },
};

const Panel = {
  detectState(durum) {
    if (State.demoOn) return 'demo';
    const age = Utils.toNum(durum?._durum_yasi_saniye);
    if (age === null) return 'offline';
    if (age <= 2) return 'live';
    if (age <= 10) return 'stale';
    return 'offline';
  },
  updateState(durum) {
    const next = this.detectState(durum);
    if (next === State.status) return;
    UI.setStatus(next);
    if (next === 'live') UI.log('info', 'Canli veri akisi kuruldu: /canli/api');
    else if (next === 'stale') UI.log('warn', 'Canli dosya akisi yavasladi');
    else if (next === 'offline' && !State.demoOn) UI.log('warn', 'Canli veri yok, dosya akisi bekleniyor');
    else if (next === 'demo') UI.log('info', 'Demo modu aktif');
  },
  latestEvent(durum) {
    if (!durum?.son_kayit?.plaka) return false;
    const record = Utils.normalizeRecord(durum.son_kayit);
    const signature = Utils.recordStamp(record);
    if (!signature || signature === State.lastSignature) return false;
    State.lastSignature = signature;
    const weight = record.tip === 'CIKIS'
      ? (record.net_agirlik ?? record.cikis_agirlik ?? record.giris_agirlik)
      : record.giris_agirlik;
    UI.log('info', `Son kayit: ${record.plaka} / ${record.tip} / ${Utils.kg(weight)} kg`);
    return true;
  },
  setDetection(durum, isNewRecord) {
    const bufferDetail = durum?.plaka_buffer_detay || null;
    const lastRecord = durum?.son_kayit ? Utils.normalizeRecord(durum.son_kayit) : null;
    if (bufferDetail?.plaka) {
      const step = durum?.seans_kilitli ? 4 : (durum?.kantar_sabit ? 3 : 2);
      const message = durum?.seans_kilitli
        ? 'Kayit kilitlendi, seans temizligi bekleniyor'
        : (durum?.kantar_sabit ? 'Plaka tamponlandi, agirlik karari bekleniyor' : 'Plaka goruldu, agirlik sabitleniyor');
      UI.showPlate(bufferDetail.plaka, step, isNewRecord, message);
      return;
    }
    if (lastRecord?.plaka) {
      UI.showPlate(
        lastRecord.plaka, 4, isNewRecord,
        durum?.seans_kilitli ? 'Son basarili okuma kilitli seans olarak tutuluyor' : 'Son basarili okuma'
      );
      return;
    }
    UI.resetPlate();
  },
  apply(data) {
    const durum = data?.durum || {};
    State.records = Array.isArray(data?.kayitlar) ? data.kayitlar.map((r) => Utils.normalizeRecord(r)) : [];
    State.total = Number(data?.toplam ?? State.records.length);
    Store.calcBars();
    const isNewRecord = this.latestEvent(durum);
    this.updateState(durum);
    UI.setScale(durum);
    this.setDetection(durum, isNewRecord);
    UI.setMetrics(data?.ozet || {}, durum);
    UI.setInfo(durum);
    UI.drawTable();
    UI.drawChart();
  },
};

const Api = {
  async poll() {
    if (State.demoOn) return;
    try {
      const r = await fetch(`/canli/api?action=panel&limit=40&t=${Date.now()}`, { cache: 'no-store' });
      if (!r.ok) throw new Error(`HTTP ${r.status}`);
      const payload = await r.json();
      if (payload?.hata) throw new Error(payload.hata);
      Panel.apply(payload);
    } catch (e) {
      if (State.status !== 'offline') {
        UI.setStatus('offline');
        UI.log('warn', 'Canli veri okunamadi, /canli/api bekleniyor');
      }
      Utils.el('kg-status').textContent = 'Panel baglantisi bekleniyor. MySQL veya durum kaynagi yanit vermiyor.';
    }
  },
};

const Demo = {
  summary() {
    const today = new Date().toISOString().slice(0, 10);
    const hourAgo = Date.now() - 3600000;
    const g = State.records.map((r) => r.guven).filter((v) => Number.isFinite(v));
    return {
      bugun_kayit: State.records.filter((r) => r.giris_tarih === today).length,
      son_saat_kayit: State.records.filter((r) => {
        const t = Utils.recordTs(r);
        return t !== null && t >= hourAgo;
      }).length,
      aktif_seans: State.records.filter((r) => r.tip === 'GIRIS').length,
      tamamlanan: State.records.filter((r) => r.tip === 'CIKIS').length,
      ortalama_guven_yuzde: g.length ? Math.round((g.reduce((a, b) => a + b, 0) / g.length) * 100) : null,
    };
  },
  addRecord(plaka, guven, tip) {
    const d = new Date();
    const tarih = d.toISOString().slice(0, 10);
    const saat = d.toTimeString().slice(0, 8);
    State.records.unshift({
      plaka, tip, durum: tip, giris_tarih: tarih, giris_saat: saat,
      giris_agirlik: tip === 'CIKIS' ? 12000 : 42000,
      cikis_tarih: tip === 'CIKIS' ? tarih : '',
      cikis_saat: tip === 'CIKIS' ? saat : '',
      cikis_agirlik: tip === 'CIKIS' ? 42000 : null,
      net_agirlik: tip === 'CIKIS' ? 30000 : null,
      guven,
    });
    State.total += 1;
    Store.calcBars();
    UI.drawTable();
    UI.drawChart();
    UI.setMetrics(this.summary(), { _durum_yasi_saniye: 0 });
    UI.log('info', `Demo kaydi: ${plaka} / ${tip} / %${Math.round(guven * 100)}`);
  },
  tick() {
    const plaka = Config.plates[Math.floor(Math.random() * Config.plates.length)];
    if (State.demoPlate !== plaka) {
      State.demoPlate = plaka;
      State.demoStep = 0;
    }
    State.demoStep += 1;
    UI.showPlate(
      State.demoPlate, State.demoStep, false,
      State.demoStep >= 3 ? 'Demo plaka tamponlandi' : `Demo dogrulama: ${State.demoStep} / ${Config.verifyThreshold}`
    );
    UI.setScale({
      kantar_kg: State.demoStep >= 3 ? 42000 : 18000 + State.demoStep * 2800,
      kantar_sabit: State.demoStep >= 3,
      plaka_buffer: State.demoStep >= 2 ? State.demoPlate : null,
      plaka_buffer_detay: State.demoStep >= 2 ? { plaka: State.demoPlate } : null,
      seans_kilitli: false,
    });
    if (State.demoStep >= Config.verifyThreshold) {
      const tip = Math.random() > 0.55 ? 'CIKIS' : 'GIRIS';
      const guven = 0.68 + Math.random() * 0.28;
      this.addRecord(State.demoPlate, guven, tip);
      State.demoPlate = null;
      State.demoStep = 0;
      setTimeout(() => { if (State.demoOn) UI.resetPlate(); }, 1600);
    }
  },
  start() {
    if (State.demoOn) return;
    State.demoOn = true;
    clearInterval(State.intervals.poll);
    clearInterval(State.intervals.cam);
    clearInterval(State.intervals.demo);
    UI.setStatus('demo');
    Utils.el('demo').textContent = 'Canli moda don';
    Utils.el('cam-note').textContent = 'Demo modu aktif. Gercek kamera yerine simulasyon gosteriliyor.';
    UI.log('info', 'Demo modu manuel olarak baslatildi');
    this.tick();
    State.intervals.demo = setInterval(() => this.tick(), 1000);
  },
  stop() {
    State.demoOn = false;
    clearInterval(State.intervals.demo);
    Utils.el('demo').textContent = 'Demo modu';
    UI.setStatus('offline');
    UI.resetPlate();
    UI.log('warn', 'Demo modu durduruldu, canli dosya akisina donuluyor');
    Api.poll();
    UI.refreshCam();
    State.intervals.poll = setInterval(() => Api.poll(), Config.pollMs);
    State.intervals.cam = setInterval(() => UI.refreshCam(), Config.camMs);
  },
};

const App = {
  bindTabs() {
    const buttons = Array.from(document.querySelectorAll('[data-tab]'));
    const panels = Array.from(document.querySelectorAll('[data-panel]'));
    const activate = (tab) => {
      buttons.forEach((btn) => btn.classList.toggle('active', btn.dataset.tab === tab));
      panels.forEach((panel) => panel.classList.toggle('active', panel.dataset.panel === tab));
    };
    buttons.forEach((btn) => btn.addEventListener('click', () => activate(btn.dataset.tab)));
    activate('genel');
  },
  bindEvents() {
    Utils.el('refresh').addEventListener('click', () => {
      Api.poll();
      UI.refreshCam();
      UI.log('info', 'Panel verisi manuel yenilendi');
    });
    Utils.el('demo').addEventListener('click', () => {
      if (State.demoOn) Demo.stop();
      else Demo.start();
    });
    Utils.el('csv').addEventListener('click', () => UI.log('info', 'CSV raporu indiriliyor'));
  },
  startClock() {
    setInterval(() => {
      Utils.el('clock').textContent = Utils.now();
      if (State.lastUpdateMs !== null && !State.demoOn) {
        UI.updateFresh((Date.now() - State.lastUpdateMs) / 1000);
      }
    }, 1000);
  },
  init() {
    this.bindTabs();
    this.bindEvents();
    this.startClock();
    UI.resetPlate();
    UI.drawChart();
    UI.setStatus('offline');
    UI.log('info', 'OtoKantar paneli yuklendi');
    UI.log('info', 'Kaynak: MySQL + /canli/api + /canli/kare');
    Api.poll();
    UI.refreshCam();
    State.intervals.poll = setInterval(() => Api.poll(), Config.pollMs);
    State.intervals.cam = setInterval(() => UI.refreshCam(), Config.camMs);
  },
};

App.init();
</script>
</body>
</html>

