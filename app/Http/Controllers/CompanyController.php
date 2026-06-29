<?php

namespace App\Http\Controllers;

use App\Models\Company;
use App\Models\VehiclePass;
use Illuminate\Database\Eloquent\Builder;
use Illuminate\Http\RedirectResponse;
use Illuminate\Http\Request;
use Illuminate\Validation\Rule;
use Illuminate\View\View;
use Symfony\Component\HttpFoundation\StreamedResponse;

class CompanyController extends Controller
{
    public function index(Request $request): View
    {
        $search = trim((string) $request->query('q', ''));
        $type = (string) $request->query('type', 'all');

        $query = Company::query()
            ->withCount('vehicleProfiles')
            ->withSum('vehiclePasses', 'net_weight_kg');

        if ($search !== '') {
            $query->where(function (Builder $query) use ($search): void {
                $query->where('name', 'like', '%'.$search.'%')
                    ->orWhere('tax_number', 'like', '%'.$search.'%')
                    ->orWhere('contact_name', 'like', '%'.$search.'%')
                    ->orWhere('phone', 'like', '%'.$search.'%');
            });
        }

        if (array_key_exists($type, Company::TYPES)) {
            $query->where('type', $type);
        }

        return view('companies.index', [
            'companies' => $query->orderBy('name')->paginate(20)->withQueryString(),
            'search' => $search,
            'type' => $type,
            'types' => Company::TYPES,
        ]);
    }

    public function create(): View
    {
        return view('companies.create', [
            'company' => new Company(['type' => 'supplier', 'is_active' => true]),
            'types' => Company::TYPES,
        ]);
    }

    public function store(Request $request): RedirectResponse
    {
        $company = Company::create($this->validatedCompany($request));

        return redirect()
            ->route('companies.show', $company)
            ->with('status', 'Firma karti olusturuldu.');
    }

    public function show(Request $request, Company $company): View
    {
        $filters = $this->filters($request);
        $passesQuery = $this->passesQuery($company, $filters);

        $summaryQuery = clone $passesQuery;

        return view('companies.show', [
            'company' => $company,
            'filters' => $filters,
            'passes' => $passesQuery->latest('passed_at')->paginate(20)->withQueryString(),
            'summary' => [
                'pass_count' => (clone $summaryQuery)->count(),
                'net_weight_kg' => (float) ((clone $summaryQuery)->sum('net_weight_kg') ?? 0),
                'vehicle_count' => (clone $summaryQuery)->distinct('plate')->count('plate'),
            ],
        ]);
    }

    public function edit(Company $company): View
    {
        return view('companies.edit', [
            'company' => $company,
            'types' => Company::TYPES,
        ]);
    }

    public function update(Request $request, Company $company): RedirectResponse
    {
        $company->update($this->validatedCompany($request, $company));

        return redirect()
            ->route('companies.show', $company)
            ->with('status', 'Firma karti guncellendi.');
    }

    public function csv(Request $request, Company $company): StreamedResponse
    {
        $filters = $this->filters($request);
        $filename = 'firma-'.$company->id.'-hareket-dokumu.csv';

        return response()->streamDownload(function () use ($company, $filters): void {
            $out = fopen('php://output', 'w');
            fputcsv($out, ['Firma', $company->name]);
            fputcsv($out, ['Olusturma', now()->format('Y-m-d H:i:s')]);
            fputcsv($out, []);
            fputcsv($out, [
                'Tarih',
                'Saat',
                'Plaka',
                'Durum',
                'Malzeme',
                'GirisKg',
                'CikisKg',
                'NetKg',
                'IrsaliyeNo',
                'Sofor',
            ]);

            $this->passesQuery($company, $filters)
                ->orderBy('passed_at')
                ->chunk(200, function ($passes) use ($out): void {
                    foreach ($passes as $pass) {
                        fputcsv($out, [
                            optional($pass->passed_at)->format('Y-m-d'),
                            optional($pass->passed_at)->format('H:i:s'),
                            $pass->plate,
                            $pass->direction,
                            $pass->material_type,
                            $pass->entry_weight_kg,
                            $pass->exit_weight_kg,
                            $pass->net_weight_kg,
                            $pass->dispatch_no,
                            $pass->driver_name,
                        ]);
                    }
                });

            fclose($out);
        }, $filename, [
            'Content-Type' => 'text/csv; charset=UTF-8',
        ]);
    }

    private function validatedCompany(Request $request, ?Company $company = null): array
    {
        return $request->validate([
            'name' => ['required', 'string', 'max:255', Rule::unique('companies', 'name')->ignore($company)],
            'type' => ['required', 'string', 'in:'.implode(',', array_keys(Company::TYPES))],
            'tax_number' => ['nullable', 'string', 'max:50'],
            'contact_name' => ['nullable', 'string', 'max:255'],
            'phone' => ['nullable', 'string', 'max:50'],
            'email' => ['nullable', 'email', 'max:255'],
            'is_active' => ['nullable', 'boolean'],
            'notes' => ['nullable', 'string', 'max:5000'],
        ]) + ['is_active' => false];
    }

    private function filters(Request $request): array
    {
        return [
            'date_from' => trim((string) $request->query('date_from', '')),
            'date_to' => trim((string) $request->query('date_to', '')),
            'plate' => trim((string) $request->query('plate', '')),
            'material' => trim((string) $request->query('material', '')),
        ];
    }

    private function passesQuery(Company $company, array $filters): Builder
    {
        return VehiclePass::query()
            ->where(function (Builder $query) use ($company): void {
                $query->where('company_id', $company->id)
                    ->orWhere('company_name', $company->name);
            })
            ->when($filters['date_from'] !== '', fn (Builder $query) => $query->whereDate('passed_at', '>=', $filters['date_from']))
            ->when($filters['date_to'] !== '', fn (Builder $query) => $query->whereDate('passed_at', '<=', $filters['date_to']))
            ->when($filters['plate'] !== '', fn (Builder $query) => $query->where('plate', 'like', '%'.$filters['plate'].'%'))
            ->when($filters['material'] !== '', fn (Builder $query) => $query->where('material_type', 'like', '%'.$filters['material'].'%'));
    }
}
