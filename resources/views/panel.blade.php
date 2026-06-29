<!doctype html>
<html lang="tr">
<head>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width,initial-scale=1">
    <title>OtoKantar V7 | Canli Operasyon</title>
    @vite(['resources/css/panel.css'])
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
        <div><h1>OtoKantar</h1><p>Canli operasyon merkezi</p></div>
    </div>
    <div class="side">
        <a class="toplink primary-link" href="{{ route('dashboard') }}">Ozet Sayfasi</a>
        @if (Auth::user()->isAdmin())
            <a class="toplink" href="{{ route('admin.users.index') }}">Kullanicilar</a>
        @endif
        <a class="toplink" href="{{ route('vehicle-profiles.index') }}">Arac Kartlari</a>
        <a class="toplink" href="{{ route('companies.index') }}">Firma Kartlari</a>
        <div class="pill" id="pill"><span class="dot"></span><span id="pill-text">BEKLENIYOR</span></div>
        <div class="clock" id="clock">--:--:--</div>
    </div>
</header>

<div class="view-switch">
    <x-panel.tab-button tab="genel" active>Genel</x-panel.tab-button>
    <x-panel.tab-button tab="kayitlar">Kayitlar</x-panel.tab-button>
</div>

<main class="tab-panel active" data-panel="genel">
    <section class="card hero span2">
        <div class="grow">
            <div class="eyebrow">Anlik kantar</div>
            <div class="weight" id="kg">--</div>
            <div class="metric-s status-text" id="kg-status">Agirlik verisi bekleniyor</div>
            <div class="bar"><span id="stale-bar"></span></div>
        </div>
        <div class="hero-grid">
            <x-panel.hero-stat title="Plaka tampon" value-id="buffer" />
            <x-panel.hero-stat title="Son kayit" value-id="fresh" subtitle="gecis bekleniyor" subtitle-id="fresh-sub" />
        </div>
    </section>

    <section class="card span2">
        <x-panel.card-head title="Arac goruntusu" badge="canli" />
        <img class="frame" id="cam" src="" alt="Canli kare">
        <div class="note" id="cam-note">Kamera karesi bekleniyor...</div>
    </section>

    <section class="card span2">
        <x-panel.card-head title="Okunan plaka" badge="aktif" />
        <div class="plate" id="plate"><b class="empty">BEKLENIYOR</b></div>
        <div class="track"><div class="seg" id="vd1"></div><div class="seg" id="vd2"></div><div class="seg" id="vd3"></div><div class="seg" id="vd4"></div></div>
        <div class="verify" id="verify">Dogrulama bekleniyor</div>
        <div class="btns">
            <button class="btn" id="refresh" type="button">Simdi yenile</button>
            @if (Auth::user()->isAdmin())
                <a class="btn" href="{{ route('canli.csv') }}" id="csv">CSV indir</a>
            @endif
        </div>
    </section>

    <details class="card drawer span4">
        <summary>
            <span>
                <b>Gunluk Ozet</b>
                <small>Icerdeki arac, bugunku kayit ve tamamlanan cikislar</small>
            </span>
            <i>Ac / Kapat</i>
        </summary>
        <div class="drawer-body metrics-grid">
            <div class="metric-tile">
                <div class="title">Icerdeki arac sayisi</div>
                <div class="metric-v acc" id="m2">0</div>
                <div class="metric-s" id="m2s">cikisi bekleyen arac</div>
            </div>
            <div class="metric-tile">
                <div class="title">Bugun kayit</div>
                <div class="metric-v" id="m1">0</div>
                <div class="metric-s" id="m1s">gunluk toplam</div>
            </div>
            <div class="metric-tile">
                <div class="title">Son 1 saat</div>
                <div class="metric-v warn" id="m4">0</div>
                <div class="metric-s">kayit hareketi</div>
            </div>
            <div class="metric-tile">
                <div class="title">Tamamlanan</div>
                <div class="metric-v" id="m3">0</div>
                <div class="metric-s">bugunku cikis</div>
            </div>
        </div>
    </details>

    <details class="card drawer span4">
        <summary>
            <span>
                <b>Saatlik Hareket</b>
                <small>Son 12 saatin yogunlugu</small>
            </span>
            <i>Ac / Kapat</i>
        </summary>
        <div class="drawer-body">
            <div class="mini-chart" id="chart"></div>
        </div>
    </details>
</main>

<main class="tab-panel" data-panel="kayitlar">
    <details class="card drawer span4" open>
        <summary>
            <span>
                <b>Kayitlari Goster</b>
                <small id="records-summary">Filtre secip gecisleri goruntule</small>
            </span>
            <i>Ac / Kapat</i>
        </summary>
        <div class="drawer-body">
        <div class="records-head">
            <x-panel.card-head title="Kayitlar" badge="0 kayit" badge-id="table-count" />
            <div class="record-tools">
                <label class="field">
                    <span>Donem</span>
                    <select id="period">
                        <option value="all">Tum kayitlar</option>
                        <option value="day">Gunluk</option>
                        <option value="month">Aylik</option>
                        <option value="year">Yillik</option>
                    </select>
                </label>
                <label class="field period-field" data-period-field="day">
                    <span>Gun</span>
                    <input id="filter-date" type="date" value="{{ date('Y-m-d') }}">
                </label>
                <label class="field period-field" data-period-field="month">
                    <span>Ay</span>
                    <input id="filter-month" type="month" value="{{ date('Y-m') }}">
                </label>
                <label class="field period-field" data-period-field="year">
                    <span>Yil</span>
                    <input id="filter-year" type="number" min="2000" max="2100" step="1" value="{{ date('Y') }}">
                </label>
                <label class="field">
                    <span>Plaka</span>
                    <input id="filter-plate" type="search" placeholder="Tum plakalar">
                </label>
                <button class="btn compact" id="apply-record-filter" type="button">Uygula</button>
                @if (Auth::user()->isAdmin())
                    <a class="btn compact primary" href="{{ route('canli.csv') }}" id="records-csv">CSV indir</a>
                @endif
            </div>
        </div>
        <div class="records-list" id="records-list">
            <div class="empty-row">Henuz kayit yok. Canli gecis bekleniyor.</div>
        </div>
        <div class="pagination" id="records-pagination"></div>
        </div>
    </details>
</main>

<script>
const Config = {
  plates: ['06ABC123', '34TR574', '35ZK882', '16BRS61', '41KLM99', '27FRT20', '06ANK80', '34ED5728', '24TR123', '79SAA001'],
  eventCheckMs: 5000,
  scaleFreshMaxSeconds: 15,
  verifyThreshold: 4,
  maxLog: 80,
  tableLimit: 200,
  chartHours: 12,
  isAdmin: @json(Auth::user()->isAdmin()),
};

const State = {
  records: [],
  archivePage: 1,
  pagination: { current_page: 1, per_page: 50, total: 0, last_page: 1 },
  activeTab: 'genel',
  bars: new Array(Config.chartHours).fill(0),
  total: 0,
  lastSignature: '',
  hasReceivedPanel: false,
  status: 'offline',
  lastUpdateMs: null,
  demoOn: false,
  demoPlate: null,
  demoStep: 0,
  intervals: { event: null, demo: null },
  filters: {
    period: 'all',
    date: @json(date('Y-m-d')),
    month: @json(date('Y-m')),
    year: @json(date('Y')),
    plate: '',
  },
};

const Utils = {
  el(id) { return document.getElementById(id); },
  now() { return new Date().toLocaleTimeString('tr-TR', { hour: '2-digit', minute: '2-digit', second: '2-digit' }); },
  toNum(value) { return Number.isFinite(Number(value)) ? Number(value) : null; },
  kg(value) { const n = this.toNum(value); return n === null ? '--' : n.toLocaleString('tr-TR', { maximumFractionDigits: 1 }); },
  formatAge(seconds) {
    const n = Math.max(0, Math.floor(Number(seconds) || 0));
    if (n < 60) return `${n} sn`;
    if (n < 3600) return `${Math.floor(n / 60)} dk`;
    if (n < 86400) return `${Math.floor(n / 3600)} sa`;
    return `${Math.floor(n / 86400)} gun`;
  },
  escapeHtml(value) {
    return String(value ?? '')
      .replaceAll('&', '&amp;')
      .replaceAll('<', '&lt;')
      .replaceAll('>', '&gt;')
      .replaceAll('"', '&quot;')
      .replaceAll("'", '&#39;');
  },
  recordTs(record) {
    if (record.gecis_zamani) {
      const direct = Date.parse(String(record.gecis_zamani).replace(' ', 'T'));
      if (Number.isFinite(direct)) return direct;
    }
    const date = record.tip === 'CIKIS' && record.cikis_tarih ? record.cikis_tarih : record.giris_tarih;
    const time = record.tip === 'CIKIS' && record.cikis_saat ? record.cikis_saat : record.giris_saat;
    const v = Date.parse(`${date || ''}T${time || ''}`);
    return Number.isFinite(v) ? v : null;
  },
  recordAgeSeconds(record) {
    const ts = this.recordTs(this.normalizeRecord(record || {}));
    if (ts === null) return null;
    return Math.max(0, (Date.now() - ts) / 1000);
  },
  latestRecordAgeSeconds(durum) {
    if (!durum?.son_kayit?.plaka) return null;
    return this.recordAgeSeconds(durum.son_kayit);
  },
  splitDateTime(value) {
    const parsed = Date.parse(String(value || '').replace(' ', 'T'));
    if (!Number.isFinite(parsed)) return { date: '', time: '' };
    const d = new Date(parsed);
    const y = d.getFullYear();
    const m = String(d.getMonth() + 1).padStart(2, '0');
    const day = String(d.getDate()).padStart(2, '0');
    const hh = String(d.getHours()).padStart(2, '0');
    const mm = String(d.getMinutes()).padStart(2, '0');
    const ss = String(d.getSeconds()).padStart(2, '0');
    return {
      date: `${y}-${m}-${day}`,
      time: `${hh}:${mm}:${ss}`,
    };
  },
  normalizeRecord(record = {}) {
    const raw = String(record.durum || record.tip || 'ICERIDE').toUpperCase();
    const tip = raw.includes('TAMAMLANDI') || raw.includes('CIKIS')
      ? 'CIKIS'
      : raw.includes('KARA') || raw.includes('ALARM')
        ? 'ALARM'
        : 'GIRIS';
    const fallback = this.splitDateTime(record.gecis_zamani);
    const girisTarih = String(record.giris_tarih || record.tarih || (tip === 'GIRIS' ? fallback.date : '') || '').trim();
    const girisSaat = String(record.giris_saat || record.saat || (tip === 'GIRIS' ? fallback.time : '') || '').trim();
    const cikisTarih = String(record.cikis_tarih || (tip === 'CIKIS' ? (record.tarih || fallback.date) : '') || '').trim();
    const cikisSaat = String(record.cikis_saat || (tip === 'CIKIS' ? (record.saat || fallback.time) : '') || '').trim();
    return {
      plaka: String(record.plaka || '').trim(),
      tip,
      durum: raw,
      giris_tarih: girisTarih,
      giris_saat: girisSaat,
      giris_agirlik: this.toNum(record.giris_agirlik),
      cikis_tarih: cikisTarih,
      cikis_saat: cikisSaat,
      cikis_agirlik: this.toNum(record.cikis_agirlik),
      net_agirlik: this.toNum(record.net_agirlik),
      arac_agirlik: this.toNum(record.arac_agirlik),
      malzeme_agirlik: this.toNum(record.malzeme_agirlik),
      guven: this.toNum(record.guven) || 0,
      gecis_zamani: String(record.gecis_zamani || '').trim(),
    };
  },
  recordStamp(record) {
    if (!record || !record.plaka) return '';
    const date = record.tip === 'CIKIS' && record.cikis_tarih ? record.cikis_tarih : record.giris_tarih;
    const time = record.tip === 'CIKIS' && record.cikis_saat ? record.cikis_saat : record.giris_saat;
    return [record.plaka, record.tip, date, time, record.gecis_zamani || ''].join('|');
  },
};

const UI = {
  log(level, message) {
    const box = Utils.el('log');
    if (!box) return;
    const row = document.createElement('div');
    row.className = 'row';
    row.innerHTML = `<span class="time">${Utils.now()}</span><span class="${Utils.escapeHtml(level)}">${Utils.escapeHtml(level.toUpperCase())}</span><span class="msg">${Utils.escapeHtml(message)}</span>`;
    box.appendChild(row);
    while (box.children.length > Config.maxLog) box.removeChild(box.firstChild);
    box.scrollTop = box.scrollHeight;
    const count = Utils.el('log-count');
    if (count) count.textContent = `${box.children.length} satir`;
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
      Utils.el('fresh-sub').textContent = 'gecis bekleniyor';
      Utils.el('stale-bar').style.width = '0%';
      State.lastUpdateMs = null;
      return;
    }
    State.lastUpdateMs = Date.now() - n * 1000;
    Utils.el('fresh').textContent = Utils.formatAge(n);
    Utils.el('fresh-sub').textContent = n <= 2 ? 'az once kaydedildi' : 'son gecis zamani';
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
  confidenceClass(percent) {
    if (percent >= 80) return '';
    return percent >= 60 ? ' mid' : ' low';
  },
  recordTag(tip) {
    if (tip === 'CIKIS') return 'cikis';
    return tip === 'ALARM' ? 'alarm' : 'giris';
  },
  recordWeight(record) {
    return record.tip === 'CIKIS'
      ? (record.malzeme_agirlik ?? record.net_agirlik ?? record.cikis_agirlik ?? record.giris_agirlik)
      : record.giris_agirlik;
  },
  recordWeightLabel(record) {
    return record.tip === 'CIKIS' ? 'Malzeme' : 'Tartim';
  },
  recordDateTime(record) {
    return {
      date: record.tip === 'CIKIS' && record.cikis_tarih ? record.cikis_tarih : record.giris_tarih,
      time: record.tip === 'CIKIS' && record.cikis_saat ? record.cikis_saat : record.giris_saat,
    };
  },
  emptyTableRow() {
    return '<div class="empty-row">Henuz kayit yok. Canli gecis bekleniyor.</div>';
  },
  renderRecordRow(record) {
    const percent = Math.max(0, Math.min(100, Math.round((record.guven || 0) * 100)));
    const { date, time } = this.recordDateTime(record);
    const weight = this.recordWeight(record);
    const weightLabel = this.recordWeightLabel(record);
    const materialWeight = record.malzeme_agirlik ?? record.net_agirlik;
    return `<details class="record-item"><summary class="record-summary"><span class="plate-td">${Utils.escapeHtml(record.plaka || '--')}</span><span>${Utils.escapeHtml(date || '--')}</span><span><small>${Utils.escapeHtml(weightLabel)}</small>${Utils.escapeHtml(Utils.kg(weight))} kg</span><span><span class="tag ${this.recordTag(record.tip)}">${Utils.escapeHtml(record.tip)}</span></span></summary><div class="record-detail"><div><span>Giris</span><b>${Utils.escapeHtml(record.giris_tarih || '--')} ${Utils.escapeHtml(record.giris_saat || '--')}</b></div><div><span>Giris kg</span><b>${Utils.escapeHtml(Utils.kg(record.giris_agirlik))}</b></div><div><span>Cikis</span><b>${Utils.escapeHtml(record.cikis_tarih || '--')} ${Utils.escapeHtml(record.cikis_saat || '--')}</b></div><div><span>Cikis kg</span><b>${Utils.escapeHtml(Utils.kg(record.cikis_agirlik))}</b></div><div><span>Arac / Dara kg</span><b>${Utils.escapeHtml(Utils.kg(record.arac_agirlik))}</b></div><div><span>Malzeme / Net kg</span><b>${Utils.escapeHtml(Utils.kg(materialWeight))}</b></div><div><span>Guven</span><b class="conf"><span class="conf-track"><span class="conf-fill${this.confidenceClass(percent)}" style="width:${percent}%"></span></span><span>%${percent}</span></b></div></div></details>`;
  },
  renderChartColumn(value, index, max, hour) {
    const height = Math.max(4, Math.round((value / max) * 92));
    const label = String(((hour - 11 + index) + 24) % 24).padStart(2, '0');
    return `<div class="col"><span class="count">${value || ''}</span><div class="stick ${index === 11 ? 'now' : ''}" style="height:${height}px"></div><span class="label">${label}</span></div>`;
  },
  drawTable() {
    Utils.el('table-count').textContent = `${State.total} kayit`;
    if (!State.records.length) {
      Utils.el('records-list').innerHTML = this.emptyTableRow();
      this.drawPagination();
      return;
    }
    Utils.el('records-list').innerHTML = State.records
      .slice(0, Config.tableLimit)
      .map((record) => this.renderRecordRow(record))
      .join('');
    this.drawPagination();
  },
  drawPagination() {
    const box = Utils.el('records-pagination');
    if (!box) return;
    const meta = State.pagination || {};
    const last = Math.max(1, Number(meta.last_page || 1));
    const current = Math.max(1, Number(meta.current_page || 1));
    if (last <= 1) {
      box.innerHTML = '';
      return;
    }
    const pages = [];
    for (let i = 1; i <= last; i += 1) {
      if (i === 1 || i === last || Math.abs(i - current) <= 2) {
        pages.push(`<button class="page-btn${i === current ? ' active' : ''}" type="button" data-page="${i}">${i}</button>`);
      } else if (pages[pages.length - 1] !== '<span class="page-gap">...</span>') {
        pages.push('<span class="page-gap">...</span>');
      }
    }
    box.innerHTML = pages.join('');
  },
  drawChart() {
    const chart = Utils.el('chart');
    if (!chart) return;
    const max = Math.max(...State.bars, 1);
    const hour = new Date().getHours();
    chart.innerHTML = State.bars
      .map((value, index) => this.renderChartColumn(value, index, max, hour))
      .join('');
  },
  setInfo(durum) {
    const s = durum?.sistem || {};
    const fallback = s.ocr_fallback ? ` / ${s.ocr_fallback}` : '';
    const ai = Utils.el('ai');
    const mode = Utils.el('mode');
    const ocr = Utils.el('ocr');
    const arch = Utils.el('arch');
    if (ai) ai.textContent = `Otomatik okuma${fallback}`;
    if (mode) mode.textContent = s.simulasyon_modu ? 'SIMULASYON' : 'CANLI';
    if (ocr) ocr.textContent = 'Aktif';
    if (arch) arch.textContent = 'Canli kayit';
    const archCard = Utils.el('arch-card');
    if (archCard) archCard.textContent = 'Canli kayit';
  },
  setMetrics(summary, durum) {
    Utils.el('m1').textContent = String(Number(summary?.bugun_kayit ?? 0));
    Utils.el('m1s').textContent = `${Number(summary?.son_saat_kayit ?? 0)} kayit son 1 saatte`;
    Utils.el('m2').textContent = String(Number(summary?.aktif_seans ?? 0));
    Utils.el('m2s').textContent = 'cikisi bekleyen arac';
    Utils.el('m3').textContent = String(Number(summary?.tamamlanan ?? 0));
    Utils.el('m4').textContent = String(Number(summary?.son_saat_kayit ?? 0));
  },
  setScale(durum) {
    const k = Utils.toNum(durum?.kantar_kg);
    const age = Utils.toNum(durum?._durum_yasi_saniye);
    const isFresh = age !== null && age <= Config.scaleFreshMaxSeconds;
    const buffer = durum?.plaka_buffer_detay?.plaka || durum?.plaka_buffer || '';
    Utils.el('buffer').textContent = isFresh && buffer ? buffer : '--';
    if (k === null || !isFresh) {
      Utils.el('kg').textContent = '--';
      Utils.el('kg-status').textContent = age === null
        ? 'Kantar verisi bekleniyor.'
        : 'Kantar sinyali bekleniyor. Son deger gosterilmiyor.';
      return;
    }
    Utils.el('kg').textContent = `${Utils.kg(k)} kg`;
    Utils.el('kg-status').textContent = durum?.seans_kilitli
      ? 'Kayit alindi. Aracin hareketi bekleniyor.'
      : (durum?.kantar_sabit ? 'Olcu sabit. Kayit icin hazir.' : 'Olcu degisiyor. Sabitlenmesi bekleniyor.');
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
      note.textContent = 'Canli kare bekleniyor.';
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
    if (next === 'live') UI.log('info', 'Canli veri akisi kuruldu');
    else if (next === 'stale') UI.log('warn', 'Canli dosya akisi yavasladi');
    else if (next === 'offline' && !State.demoOn) UI.log('warn', 'Canli veri bekleniyor');
    else if (next === 'demo') UI.log('info', 'Demo modu aktif');
  },
  latestEvent(durum, canTreatAsNew) {
    if (!durum?.son_kayit?.plaka) return false;
    const record = Utils.normalizeRecord(durum.son_kayit);
    const signature = Utils.recordStamp(record);
    if (!signature) return false;
    if (!State.lastSignature) {
      State.lastSignature = signature;
      return Boolean(canTreatAsNew);
    }
    if (signature === State.lastSignature) return false;
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
    const hadPanelData = State.hasReceivedPanel;
    State.records = Array.isArray(data?.kayitlar) ? data.kayitlar.map((r) => Utils.normalizeRecord(r)) : [];
    State.total = Number(data?.toplam ?? State.records.length);
    Store.calcBars();
    const isNewRecord = this.latestEvent(durum, hadPanelData);
    State.hasReceivedPanel = true;
    this.updateState(durum);
    UI.setScale(durum);
    this.setDetection(durum, isNewRecord);
    if (isNewRecord) UI.refreshCam();
    UI.setMetrics(data?.ozet || {}, durum);
    UI.updateFresh(Utils.latestRecordAgeSeconds(durum));
    UI.setInfo(durum);
    UI.drawTable();
    UI.drawChart();
  },
};

const Api = {
  liveTickerUrl() {
    const params = new URLSearchParams({
      t: String(Date.now()),
    });
    return `/canli/live-ticker?${params.toString()}`;
  },
  archiveUrl(page = State.archivePage) {
    const params = new URLSearchParams({
      period: State.filters.period,
      date: State.filters.date,
      month: State.filters.month,
      year: State.filters.year,
      plate: State.filters.plate,
      page: String(page),
      t: String(Date.now()),
    });
    return `/canli/archive?${params.toString()}`;
  },
  durumUrl() {
    const params = new URLSearchParams({
      action: 'durum',
      t: String(Date.now()),
    });
    return `/canli/api?${params.toString()}`;
  },
  csvUrl() {
    const params = new URLSearchParams({
      period: State.filters.period,
      date: State.filters.date,
      month: State.filters.month,
      year: State.filters.year,
      plate: State.filters.plate,
    });
    return `/canli/csv?${params.toString()}`;
  },
  async poll() {
    if (State.demoOn) return;
    try {
      const r = await fetch(this.liveTickerUrl(), { cache: 'no-store' });
      if (!r.ok) throw new Error(`HTTP ${r.status}`);
      const payload = await r.json();
      if (payload?.hata) throw new Error(payload.hata);
      Panel.apply(payload);
    } catch (e) {
      if (State.status !== 'offline') {
        UI.setStatus('offline');
        UI.log('warn', 'Canli veri okunamadi');
      }
      Utils.el('kg-status').textContent = 'Panel baglantisi bekleniyor.';
    }
  },
  async loadArchive(page = 1) {
    State.archivePage = Math.max(1, Number(page) || 1);
    try {
      const r = await fetch(this.archiveUrl(State.archivePage), { cache: 'no-store' });
      if (!r.ok) throw new Error(`HTTP ${r.status}`);
      const payload = await r.json();
      if (payload?.hata) throw new Error(payload.hata);
      State.records = Array.isArray(payload?.kayitlar) ? payload.kayitlar.map((record) => Utils.normalizeRecord(record)) : [];
      State.total = Number(payload?.toplam ?? State.records.length);
      State.pagination = payload?.pagination || {
        current_page: State.archivePage,
        per_page: 50,
        total: State.total,
        last_page: 1,
      };
      const recordsSummary = Utils.el('records-summary');
      if (recordsSummary) recordsSummary.textContent = `${State.total} kayit listeleniyor`;
      Store.calcBars();
      UI.drawTable();
      UI.drawChart();
    } catch (e) {
      UI.log('warn', 'Kayitlar okunamadi');
      Utils.el('records-list').innerHTML = '<div class="empty-row">Kayitlar su anda okunamiyor.</div>';
      Utils.el('records-pagination').innerHTML = '';
    }
  },
  async checkEvent() {
    if (State.demoOn) return;
    try {
      const r = await fetch(this.durumUrl(), { cache: 'no-store' });
      if (!r.ok) throw new Error(`HTTP ${r.status}`);
      const durum = await r.json();
      if (durum?.hata) throw new Error(durum.hata);
      const isNewRecord = Panel.latestEvent(durum, State.hasReceivedPanel);
      Panel.updateState(durum);
      UI.setScale(durum);
      UI.updateFresh(Utils.latestRecordAgeSeconds(durum));
      UI.setInfo(durum);
      if (isNewRecord) {
        UI.refreshCam();
        await this.poll();
      }
    } catch (e) {
      if (State.status !== 'offline') {
        UI.setStatus('offline');
        UI.log('warn', 'Canli durum kontrolu okunamadi');
      }
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
      arac_agirlik: tip === 'CIKIS' ? 12000 : null,
      malzeme_agirlik: tip === 'CIKIS' ? 30000 : null,
      guven,
    });
    State.total += 1;
    Store.calcBars();
    UI.drawTable();
    UI.drawChart();
    UI.setMetrics(this.summary(), {});
    UI.updateFresh(0);
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
    clearInterval(State.intervals.event);
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
    App.startLivePolling();
  },
};

const App = {
  startLivePolling() {
    if (State.demoOn || State.activeTab === 'kayitlar') return;
    clearInterval(State.intervals.event);
    Api.poll();
    State.intervals.event = setInterval(() => Api.poll(), Config.eventCheckMs);
  },
  stopLivePolling() {
    clearInterval(State.intervals.event);
    State.intervals.event = null;
  },
  bindTabs() {
    const buttons = Array.from(document.querySelectorAll('[data-tab]'));
    const panels = Array.from(document.querySelectorAll('[data-panel]'));
    const activate = (tab) => {
      State.activeTab = tab;
      buttons.forEach((btn) => btn.classList.toggle('active', btn.dataset.tab === tab));
      panels.forEach((panel) => panel.classList.toggle('active', panel.dataset.panel === tab));
      if (tab === 'kayitlar') {
        this.stopLivePolling();
        Api.loadArchive(State.archivePage);
      } else {
        this.startLivePolling();
      }
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
    const demo = Utils.el('demo');
    if (demo) {
      demo.addEventListener('click', () => {
        if (State.demoOn) Demo.stop();
        else Demo.start();
      });
    }
    const csv = Utils.el('csv');
    if (csv) {
      csv.addEventListener('click', () => {
        csv.href = Api.csvUrl();
        UI.log('info', 'CSV raporu indiriliyor');
      });
    }
    const recordsCsv = Utils.el('records-csv');
    if (recordsCsv) {
      recordsCsv.addEventListener('click', () => {
        recordsCsv.href = Api.csvUrl();
        UI.log('info', 'Filtreli CSV raporu indiriliyor');
      });
    }
    Utils.el('apply-record-filter').addEventListener('click', () => {
      this.syncRecordFilters();
      Api.loadArchive(1);
      UI.log('info', 'Kayit filtresi guncellendi');
    });
    Utils.el('period').addEventListener('change', () => {
      this.syncRecordFilters();
      this.updateFilterFields();
      Api.loadArchive(1);
    });
    ['filter-date', 'filter-month', 'filter-year', 'filter-plate'].forEach((id) => {
      Utils.el(id).addEventListener('change', () => {
        this.syncRecordFilters();
        Api.loadArchive(1);
      });
    });
    Utils.el('records-pagination').addEventListener('click', (event) => {
      const btn = event.target.closest('[data-page]');
      if (!btn) return;
      Api.loadArchive(Number(btn.dataset.page || 1));
    });
  },
  syncRecordFilters() {
    State.filters.period = Utils.el('period').value || 'all';
    State.filters.date = Utils.el('filter-date').value || State.filters.date;
    State.filters.month = Utils.el('filter-month').value || State.filters.month;
    State.filters.year = Utils.el('filter-year').value || State.filters.year;
    State.filters.plate = Utils.el('filter-plate').value.trim();
    const url = Api.csvUrl();
    const csv = Utils.el('csv');
    const recordsCsv = Utils.el('records-csv');
    if (csv) csv.href = url;
    if (recordsCsv) recordsCsv.href = url;
  },
  updateFilterFields() {
    const period = Utils.el('period').value || 'all';
    document.querySelectorAll('[data-period-field]').forEach((field) => {
      field.classList.toggle('hidden', field.dataset.periodField !== period);
    });
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
    this.updateFilterFields();
    this.syncRecordFilters();
    this.bindEvents();
    this.startClock();
    UI.resetPlate();
    UI.drawChart();
    UI.setStatus('offline');
    UI.log('info', 'OtoKantar paneli yuklendi');
    UI.refreshCam();
  },
};

App.init();
</script>
</body>
</html>
