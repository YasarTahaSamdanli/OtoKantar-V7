<!doctype html>
<html lang="tr">
<head>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <title>{{ $company->name }} | Firma Hareket Dökümü</title>
    <style>
        :root {
            color: #172033;
            font-family: Arial, Helvetica, sans-serif;
            font-size: 13px;
        }

        body {
            margin: 0;
            background: #eef2f7;
        }

        .page {
            box-sizing: border-box;
            width: min(1120px, calc(100% - 32px));
            margin: 24px auto;
            background: #ffffff;
            border: 1px solid #d8dee8;
            box-shadow: 0 18px 50px rgba(23, 32, 51, .12);
        }

        .toolbar {
            display: flex;
            justify-content: flex-end;
            gap: 8px;
            padding: 14px 18px;
            border-bottom: 1px solid #e3e8f0;
            background: #f8fafc;
        }

        .btn {
            border: 1px solid #b7c3d4;
            border-radius: 6px;
            background: #ffffff;
            color: #172033;
            cursor: pointer;
            font-size: 13px;
            font-weight: 700;
            padding: 8px 12px;
            text-decoration: none;
        }

        .btn.primary {
            border-color: #0f766e;
            background: #0f766e;
            color: #ffffff;
        }

        .report {
            padding: 28px;
        }

        .header {
            display: flex;
            align-items: flex-start;
            justify-content: space-between;
            gap: 24px;
            border-bottom: 2px solid #172033;
            padding-bottom: 18px;
        }

        .brand {
            color: #0f766e;
            font-size: 12px;
            font-weight: 800;
            letter-spacing: .14em;
            text-transform: uppercase;
        }

        h1 {
            margin: 8px 0 0;
            color: #111827;
            font-size: 26px;
            line-height: 1.15;
        }

        .meta {
            min-width: 260px;
            text-align: right;
        }

        .meta div,
        .info div {
            margin-bottom: 6px;
        }

        .label {
            color: #64748b;
            font-size: 11px;
            font-weight: 700;
            letter-spacing: .08em;
            text-transform: uppercase;
        }

        .value {
            color: #172033;
            font-weight: 700;
        }

        .summary {
            display: grid;
            grid-template-columns: repeat(3, 1fr);
            gap: 12px;
            margin: 22px 0;
        }

        .metric {
            border: 1px solid #dce3ee;
            border-radius: 6px;
            padding: 14px;
        }

        .metric .value {
            display: block;
            margin-top: 7px;
            font-size: 22px;
        }

        .section-title {
            margin: 22px 0 10px;
            color: #172033;
            font-size: 15px;
            font-weight: 800;
        }

        table {
            width: 100%;
            border-collapse: collapse;
            table-layout: fixed;
        }

        th {
            background: #172033;
            color: #ffffff;
            font-size: 11px;
            padding: 9px 7px;
            text-align: left;
        }

        td {
            border-bottom: 1px solid #e5eaf2;
            color: #172033;
            padding: 8px 7px;
            vertical-align: top;
        }

        th.num,
        td.num {
            text-align: right;
        }

        .empty {
            border: 1px dashed #cbd5e1;
            color: #64748b;
            padding: 26px;
            text-align: center;
        }

        .footer {
            border-top: 1px solid #dce3ee;
            color: #64748b;
            font-size: 11px;
            margin-top: 20px;
            padding-top: 12px;
        }

        @media print {
            @page {
                margin: 12mm;
                size: A4 landscape;
            }

            body {
                background: #ffffff;
            }

            .page {
                width: 100%;
                margin: 0;
                border: 0;
                box-shadow: none;
            }

            .toolbar {
                display: none;
            }

            .report {
                padding: 0;
            }

            table {
                font-size: 11px;
            }

            th,
            td {
                padding: 6px 5px;
            }
        }
    </style>
</head>
<body>
    <main class="page">
        <div class="toolbar">
            <a class="btn" href="{{ route('companies.show', $company) }}">Karta Dön</a>
            <button class="btn primary" type="button" onclick="window.print()">Yazdır / PDF Kaydet</button>
        </div>

        <section class="report">
            <header class="header">
                <div>
                    <div class="brand">OtoKantar V7</div>
                    <h1>Firma Hareket Dökümü</h1>
                    <div class="info" style="margin-top: 14px;">
                        <div><span class="label">Firma</span><br><span class="value">{{ $company->name }}</span></div>
                        <div><span class="label">Filtre</span><br><span class="value">{{ $filterSummary }}</span></div>
                    </div>
                </div>

                <div class="meta">
                    <div><span class="label">Oluşturma</span><br><span class="value">{{ $generatedAt->format('d.m.Y H:i') }}</span></div>
                    <div><span class="label">Yetkili</span><br><span class="value">{{ $company->contact_name ?: '-' }}</span></div>
                    <div><span class="label">Telefon</span><br><span class="value">{{ $company->phone ?: '-' }}</span></div>
                </div>
            </header>

            <section class="summary" aria-label="Özet">
                <div class="metric">
                    <span class="label">Toplam Geçiş</span>
                    <span class="value">{{ number_format($summary['pass_count']) }}</span>
                </div>
                <div class="metric">
                    <span class="label">Tekil Plaka</span>
                    <span class="value">{{ number_format($summary['vehicle_count']) }}</span>
                </div>
                <div class="metric">
                    <span class="label">Toplam Net Ton</span>
                    <span class="value">{{ number_format($summary['net_weight_kg'] / 1000, 2, ',', '.') }}</span>
                </div>
            </section>

            <div class="section-title">Hareketler</div>

            @if ($passes->isEmpty())
                <div class="empty">Bu filtrelerde firma hareketi yok.</div>
            @else
                <table>
                    <thead>
                        <tr>
                            <th style="width: 40px;">No</th>
                            <th style="width: 90px;">Tarih</th>
                            <th style="width: 72px;">Saat</th>
                            <th style="width: 90px;">Plaka</th>
                            <th style="width: 70px;">İşlem</th>
                            <th>Malzeme</th>
                            <th class="num" style="width: 90px;">Giriş Kg</th>
                            <th class="num" style="width: 90px;">Çıkış Kg</th>
                            <th class="num" style="width: 90px;">Net Kg</th>
                            <th style="width: 100px;">İrsaliye</th>
                            <th style="width: 115px;">Şoför</th>
                        </tr>
                    </thead>
                    <tbody>
                        @foreach ($passes as $index => $pass)
                            <tr>
                                <td>{{ $index + 1 }}</td>
                                <td>{{ optional($pass->passed_at)->format('d.m.Y') ?: '-' }}</td>
                                <td>{{ optional($pass->passed_at)->format('H:i:s') ?: '-' }}</td>
                                <td><strong>{{ $pass->plate }}</strong></td>
                                <td>{{ $pass->direction === 'CIKIS' ? 'Çıkış' : 'Giriş' }}</td>
                                <td>{{ $pass->material_type ?: '-' }}</td>
                                <td class="num">{{ $pass->entry_weight_kg !== null ? number_format((float) $pass->entry_weight_kg, 0, ',', '.') : '-' }}</td>
                                <td class="num">{{ $pass->exit_weight_kg !== null ? number_format((float) $pass->exit_weight_kg, 0, ',', '.') : '-' }}</td>
                                <td class="num">{{ $pass->net_weight_kg !== null ? number_format((float) $pass->net_weight_kg, 0, ',', '.') : '-' }}</td>
                                <td>{{ $pass->dispatch_no ?: '-' }}</td>
                                <td>{{ $pass->driver_name ?: '-' }}</td>
                            </tr>
                        @endforeach
                    </tbody>
                </table>
            @endif

            <div class="footer">
                Bu döküm OtoKantar V7 tarafından oluşturulmuştur. Görüntülenen kayıtlar seçili filtrelere göre hazırlanmıştır.
            </div>
        </section>
    </main>
</body>
</html>
