<x-app-layout>
    <x-slot name="header">
        <h2 class="font-semibold text-xl text-gray-800 leading-tight">
            Kullanıcılar
        </h2>
    </x-slot>

    <div class="py-12">
        <div class="max-w-7xl mx-auto sm:px-6 lg:px-8">
            <div class="bg-white overflow-hidden shadow-sm sm:rounded-lg">
                <div class="p-6 text-gray-900">
                    <div class="flex items-center justify-between gap-4">
                        <div class="text-sm text-gray-600">
                            Çalışan hesaplarını buradan oluşturabilirsiniz.
                        </div>
                        <a href="{{ route('admin.users.create') }}"
                           class="inline-flex items-center rounded-md bg-gray-900 px-4 py-2 text-sm font-semibold text-white hover:bg-gray-800">
                            Yeni Çalışan
                        </a>
                    </div>

                    <div class="mt-6 overflow-x-auto">
                        <table class="min-w-full text-sm">
                            <thead class="text-left text-gray-600">
                            <tr class="border-b">
                                <th class="py-2 pr-4">ID</th>
                                <th class="py-2 pr-4">Ad</th>
                                <th class="py-2 pr-4">Email</th>
                                <th class="py-2 pr-4">Rol</th>
                                <th class="py-2 pr-4">Oluşturma</th>
                            </tr>
                            </thead>
                            <tbody>
                            @foreach($users as $user)
                                <tr class="border-b">
                                    <td class="py-2 pr-4">{{ $user->id }}</td>
                                    <td class="py-2 pr-4">{{ $user->name }}</td>
                                    <td class="py-2 pr-4">{{ $user->email }}</td>
                                    <td class="py-2 pr-4">
                                        <span class="inline-flex items-center rounded-full bg-gray-100 px-2 py-0.5 text-xs font-semibold text-gray-700">
                                            {{ $user->role }}
                                        </span>
                                    </td>
                                    <td class="py-2 pr-4">{{ $user->created_at?->format('Y-m-d H:i') }}</td>
                                </tr>
                            @endforeach
                            </tbody>
                        </table>
                    </div>

                    <div class="mt-6">
                        {{ $users->links() }}
                    </div>
                </div>
            </div>
        </div>
    </div>
</x-app-layout>
