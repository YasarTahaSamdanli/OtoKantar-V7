<x-app-layout>
    <x-slot name="header">
        <h2 class="font-semibold text-xl text-gray-800 leading-tight">
            Yonetim Paneli
        </h2>
    </x-slot>

    <div class="py-12">
        <div class="max-w-7xl mx-auto sm:px-6 lg:px-8">
            <div class="grid gap-6 md:grid-cols-3">
                <a href="{{ route('canli.view') }}"
                   class="block bg-white overflow-hidden shadow-sm sm:rounded-lg border border-gray-100 hover:border-gray-300 transition">
                    <div class="p-6 text-gray-900">
                        <div class="text-sm font-semibold text-gray-500 uppercase tracking-wide">Canli operasyon</div>
                        <div class="mt-3 text-2xl font-semibold">Canli Panel</div>
                        <p class="mt-2 text-sm text-gray-600">Kantar durumu, son gecisler, canli kare ve CSV raporuna gir.</p>
                    </div>
                </a>

                <a href="{{ route('admin.users.index') }}"
                   class="block bg-white overflow-hidden shadow-sm sm:rounded-lg border border-gray-100 hover:border-gray-300 transition">
                    <div class="p-6 text-gray-900">
                        <div class="text-sm font-semibold text-gray-500 uppercase tracking-wide">Hesap yonetimi</div>
                        <div class="mt-3 text-2xl font-semibold">Kullanicilar</div>
                        <p class="mt-2 text-sm text-gray-600">Admin ve calisan hesaplarini listele, kimlerin giris yapabilecegini gor.</p>
                    </div>
                </a>

                <a href="{{ route('admin.users.create') }}"
                   class="block bg-white overflow-hidden shadow-sm sm:rounded-lg border border-gray-100 hover:border-gray-300 transition">
                    <div class="p-6 text-gray-900">
                        <div class="text-sm font-semibold text-gray-500 uppercase tracking-wide">Yeni hesap</div>
                        <div class="mt-3 text-2xl font-semibold">Kullanici Olustur</div>
                        <p class="mt-2 text-sm text-gray-600">Yeni calisan veya yeni admin hesabi ac.</p>
                    </div>
                </a>
            </div>

            <div class="mt-6 bg-white overflow-hidden shadow-sm sm:rounded-lg">
                <div class="p-6 text-gray-900">
                    <h3 class="text-lg font-semibold">Roller nasil calisir?</h3>
                    <div class="mt-4 grid gap-4 md:grid-cols-2">
                        <div class="rounded-md border border-gray-200 p-4">
                            <div class="font-semibold">Admin</div>
                            <p class="mt-1 text-sm text-gray-600">Canli paneli acar, kullanicilari gorur, yeni admin veya calisan olusturur.</p>
                        </div>
                        <div class="rounded-md border border-gray-200 p-4">
                            <div class="font-semibold">Calisan</div>
                            <p class="mt-1 text-sm text-gray-600">Su an canli panel ve yonetim alanlari admin rolune kapali oldugu icin calisan hesaplari giriste yetkisiz kalir.</p>
                        </div>
                    </div>
                </div>
            </div>
        </div>
    </div>
</x-app-layout>
