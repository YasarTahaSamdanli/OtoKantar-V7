<?php

namespace App\Services;

class CanliAgirlikService
{
    public function parseAgirlik(mixed $value): ?float
    {
        if ($value === null) {
            return null;
        }
        $text = trim((string) $value);
        if ($text === '') {
            return null;
        }
        $text = str_replace(',', '.', $text);
        return is_numeric($text) ? (float) $text : null;
    }

    public function jsonAgirlikIndexiGetir(string $jsonDurumDosya): array
    {
        if (!is_file($jsonDurumDosya)) {
            return [];
        }
        $raw = file_get_contents($jsonDurumDosya);
        if ($raw === false) {
            return [];
        }
        $data = json_decode($raw, true);
        if (!is_array($data) || !isset($data['son_10']) || !is_array($data['son_10'])) {
            return [];
        }

        $index = [];
        foreach ($data['son_10'] as $row) {
            if (!is_array($row)) {
                continue;
            }
            $plaka = strtoupper(trim((string) ($row['plaka'] ?? '')));
            if ($plaka === '') {
                continue;
            }
            $durumRaw = strtoupper(trim((string) ($row['durum'] ?? $row['tip'] ?? 'GIRIS')));
            $tip = ($durumRaw === 'TAMAMLANDI' || $durumRaw === 'CIKIS') ? 'CIKIS' : 'GIRIS';
            $tarih = trim((string) ($row['giris_tarih'] ?? $row['tarih'] ?? ''));
            $saat = trim((string) ($row['giris_saat'] ?? $row['saat'] ?? ''));
            if ($tip === 'CIKIS' && trim((string) ($row['cikis_tarih'] ?? '')) !== '') {
                $tarih = trim((string) $row['cikis_tarih']);
            }
            if ($tip === 'CIKIS' && trim((string) ($row['cikis_saat'] ?? '')) !== '') {
                $saat = trim((string) $row['cikis_saat']);
            }
            if ($tarih === '' || $saat === '') {
                continue;
            }

            $key = $plaka.'|'.$tip.'|'.$tarih.'|'.$saat;
            $index[$key] = [
                'giris_agirlik' => $this->parseAgirlik($row['giris_agirlik'] ?? null),
                'cikis_agirlik' => $this->parseAgirlik($row['cikis_agirlik'] ?? null),
                'net_agirlik' => $this->parseAgirlik($row['net_agirlik'] ?? null),
            ];
        }
        return $index;
    }

    public function csvAgirlikIndexiGetir(string $csvDosya): array
    {
        $exact = [];
        $minute = [];
        if (!is_file($csvDosya)) {
            return ['exact' => $exact, 'minute' => $minute];
        }
        $lines = file($csvDosya, FILE_IGNORE_NEW_LINES | FILE_SKIP_EMPTY_LINES);
        if (!is_array($lines)) {
            return ['exact' => $exact, 'minute' => $minute];
        }

        foreach ($lines as $line) {
            $row = str_getcsv((string) $line, ';');
            if (!is_array($row) || count($row) < 10) {
                continue;
            }
            $plaka = strtoupper(trim((string) ($row[0] ?? '')));
            $durum = strtoupper(trim((string) ($row[1] ?? '')));
            if ($plaka === '' || ($durum !== 'ICERIDE' && $durum !== 'TAMAMLANDI' && $durum !== 'GIRIS' && $durum !== 'CIKIS')) {
                continue;
            }
            $tip = ($durum === 'TAMAMLANDI' || $durum === 'CIKIS') ? 'CIKIS' : 'GIRIS';
            $tarih = trim((string) (($tip === 'CIKIS' ? ($row[5] ?? '') : ($row[2] ?? ''))));
            $saat = trim((string) (($tip === 'CIKIS' ? ($row[6] ?? '') : ($row[3] ?? ''))));
            if ($tarih === '' || $saat === '') {
                continue;
            }

            $weights = [
                'giris_agirlik' => $this->parseAgirlik($row[4] ?? null),
                'cikis_agirlik' => $this->parseAgirlik($row[7] ?? null),
                'net_agirlik' => $this->parseAgirlik($row[8] ?? null),
            ];
            $exact[$plaka.'|'.$tip.'|'.$tarih.'|'.$saat] = $weights;
            $minute[$plaka.'|'.$tip.'|'.$tarih.'|'.substr($saat, 0, 5)] = $weights;
        }

        return ['exact' => $exact, 'minute' => $minute];
    }

    public function agirlikBul(string $plaka, string $tip, string $tarih, string $saat, array $jsonIndex, array $csvIndex): ?array
    {
        $exactKey = $plaka.'|'.$tip.'|'.$tarih.'|'.$saat;
        if (isset($jsonIndex[$exactKey]) && is_array($jsonIndex[$exactKey])) {
            return $jsonIndex[$exactKey];
        }
        if (isset($csvIndex['exact'][$exactKey]) && is_array($csvIndex['exact'][$exactKey])) {
            return $csvIndex['exact'][$exactKey];
        }

        $minuteKey = $plaka.'|'.$tip.'|'.$tarih.'|'.substr($saat, 0, 5);
        if (isset($csvIndex['minute'][$minuteKey]) && is_array($csvIndex['minute'][$minuteKey])) {
            return $csvIndex['minute'][$minuteKey];
        }
        return null;
    }
}
