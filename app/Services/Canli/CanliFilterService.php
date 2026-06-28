<?php

namespace App\Services\Canli;

class CanliFilterService
{
    public function normalize(array $filters): array
    {
        $period = strtolower((string) ($filters['period'] ?? 'all'));
        if (! in_array($period, ['all', 'day', 'month', 'year'], true)) {
            $period = 'all';
        }

        return [
            'period' => $period,
            'date' => preg_match('/^\d{4}-\d{2}-\d{2}$/', (string) ($filters['date'] ?? '')) === 1
                ? (string) $filters['date']
                : date('Y-m-d'),
            'month' => preg_match('/^\d{4}-\d{2}$/', (string) ($filters['month'] ?? '')) === 1
                ? (string) $filters['month']
                : date('Y-m'),
            'year' => preg_match('/^\d{4}$/', (string) ($filters['year'] ?? '')) === 1
                ? (string) $filters['year']
                : date('Y'),
            'plate' => strtoupper(trim((string) ($filters['plate'] ?? $filters['plaka'] ?? ''))),
        ];
    }

    public function slug(array $filters): string
    {
        $filters = $this->normalize($filters);

        return match ($filters['period']) {
            'day' => 'gunluk_'.$filters['date'],
            'month' => 'aylik_'.$filters['month'],
            'year' => 'yillik_'.$filters['year'],
            default => 'tum_kayitlar',
        };
    }

    public function matchesRecord(array $row, array $filters): bool
    {
        $filters = $this->normalize($filters);
        if ($filters['plate'] !== '' && ! str_contains(strtoupper((string) ($row['plaka'] ?? '')), $filters['plate'])) {
            return false;
        }
        if ($filters['period'] === 'all') {
            return true;
        }

        $date = $this->recordDate($row);
        if ($date === null) {
            return false;
        }

        return $this->dateMatchesPeriod($date, $filters);
    }

    public function matchesCsvRow(array $row, array $filters): bool
    {
        $filters = $this->normalize($filters);
        if ($filters['plate'] !== '' && ! str_contains(strtoupper((string) ($row['Plaka'] ?? $row['plaka'] ?? '')), $filters['plate'])) {
            return false;
        }
        if ($filters['period'] === 'all') {
            return true;
        }

        $date = $this->csvRowDate($row);
        if ($date === null) {
            return false;
        }

        return $this->dateMatchesPeriod($date, $filters);
    }

    public function recordDate(array $row): ?string
    {
        $type = $this->recordType($row);
        $value = $type === 'CIKIS'
            ? ($row['cikis_tarih'] ?? $row['tarih'] ?? $row['giris_tarih'] ?? null)
            : ($row['giris_tarih'] ?? $row['tarih'] ?? $row['cikis_tarih'] ?? null);

        if ($value === null || trim((string) $value) === '') {
            $value = $row['gecis_zamani'] ?? $row['GecisZamani'] ?? null;
        }

        if ($value === null || trim((string) $value) === '') {
            return null;
        }

        $timestamp = strtotime((string) $value);

        return $timestamp === false ? null : date('Y-m-d', $timestamp);
    }

    public function recordType(array $row): string
    {
        $raw = strtoupper(trim((string) ($row['tip'] ?? $row['durum'] ?? $row['yon'] ?? 'GIRIS')));

        return str_contains($raw, 'CIKIS') || str_contains($raw, 'TAMAMLANDI') ? 'CIKIS' : 'GIRIS';
    }

    public function csvRowDate(array $row): ?string
    {
        $type = $this->recordType([
            'tip' => $row['Tip'] ?? null,
            'durum' => $row['Durum'] ?? $row['Yon'] ?? null,
        ]);
        $value = $type === 'CIKIS'
            ? ($row['CikisTarih'] ?? $row['Tarih'] ?? $row['GecisZamani'] ?? $row['GirisTarih'] ?? null)
            : ($row['GirisTarih'] ?? $row['Tarih'] ?? $row['GecisZamani'] ?? $row['CikisTarih'] ?? null);

        if ($value === null || trim((string) $value) === '') {
            return null;
        }

        $timestamp = strtotime((string) $value);

        return $timestamp === false ? null : date('Y-m-d', $timestamp);
    }

    private function dateMatchesPeriod(string $date, array $filters): bool
    {
        return match ($filters['period']) {
            'day' => $date === $filters['date'],
            'month' => str_starts_with($date, $filters['month'].'-'),
            'year' => str_starts_with($date, $filters['year'].'-'),
            default => true,
        };
    }
}
